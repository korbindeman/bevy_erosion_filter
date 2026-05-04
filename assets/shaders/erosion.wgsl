// =====================================================================================
// bevy_erosion_filter — WGSL port of Rune Skovbo Johansen's gully-erosion noise.
//
// Upstream: https://www.shadertoy.com/view/wXcfWn
//
// The `gullies` and `erosion` functions are derivative works of Rune Skovbo
// Johansen's GLSL implementation, ported to WGSL under the Mozilla Public
// License v2. The IQ gradient noise (`noised`) and IQ hash (`hash2`) are
// bundled here because the algorithm requires them; they are reused under MIT.
//
// Erosion-filter upstream copyright:
//
//   Copyright 2025 Rune Skovbo Johansen
//
// Earlier technique lineage:
//
//   Copyright 2023 Fewes
//   Copyright 2020 Clay John
//
// WGSL port additions copyright (c) 2026 the bevy_erosion_filter authors.
// See LICENSE-MPL-2.0, LICENSE-MIT, and THIRD-PARTY-NOTICES.md.
// =====================================================================================

#define_import_path bevy_erosion_filter::erosion

const PI: f32 = 3.14159265358979;

// Inigo Quilez 2D hash. Returns a vec2 in [-1, 1].
fn hash2(x_in: vec2<f32>) -> vec2<f32> {
    let k = vec2<f32>(0.3183099, 0.3678794);
    let x = x_in * k + k.yx;
    return -1.0 + 2.0 * fract(16.0 * k * fract(x.x * x.y * (x.x + x.y)));
}

// Inigo Quilez gradient noise with analytical derivatives.
// Returns vec3(value, d/dx, d/dy). https://www.shadertoy.com/view/XdXBRH
fn noised(p: vec2<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let ga = hash2(i + vec2<f32>(0.0, 0.0));
    let gb = hash2(i + vec2<f32>(1.0, 0.0));
    let gc = hash2(i + vec2<f32>(0.0, 1.0));
    let gd = hash2(i + vec2<f32>(1.0, 1.0));

    let va = dot(ga, f - vec2<f32>(0.0, 0.0));
    let vb = dot(gb, f - vec2<f32>(1.0, 0.0));
    let vc = dot(gc, f - vec2<f32>(0.0, 1.0));
    let vd = dot(gd, f - vec2<f32>(1.0, 1.0));

    let value = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd);
    let deriv = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
        + du * (u.yx * (va - vb - vc + vd) + vec2<f32>(vb, vc) - va);

    return vec3<f32>(value, deriv.x, deriv.y);
}

// Standard fBm built from `noised`, returning vec3(value, d/dx, d/dy) so the
// gradient can be passed straight into `erosion` as the input slope.
fn fbm(p: vec2<f32>, frequency: f32, octaves: i32, lacunarity: f32, gain: f32) -> vec3<f32> {
    var n = vec3<f32>(0.0);
    var freq = frequency;
    var amp = 1.0;
    for (var i: i32 = 0; i < octaves; i++) {
        n += noised(p * freq) * amp * vec3<f32>(1.0, freq, freq);
        amp *= gain;
        freq *= lacunarity;
    }
    return n;
}

// Geometric series 1 + g + g^2 + ... + g^(N-1). Used to compute the total
// magnitude of the layered erosion contribution.
fn magnitude_sum(octaves: i32, gain: f32) -> f32 {
    if (octaves <= 0) {
        return 0.0;
    }
    if (abs(1.0 - gain) < 0.000001) {
        return f32(octaves);
    }
    return (1.0 - pow(gain, f32(octaves))) / (1.0 - gain);
}

// Single octave of gully noise.
//
// `slope` is the local terrain slope (analytical gradient) that the gullies
// will run along. Returns vec3(height, d/dx, d/dy) with amplitude bounded by 1.
fn gullies(p: vec2<f32>, slope: vec2<f32>) -> vec3<f32> {
    // Side direction = perpendicular to slope, pre-multiplied by 2π so every
    // dot-product call site gets the right phase increment for free.
    let side_dir = slope.yx * vec2<f32>(-1.0, 1.0) * 2.0 * PI;

    let p_int = floor(p);
    let p_frac = fract(p);

    var acc = vec3<f32>(0.0);
    var weight_sum = 0.0;

    // 4×4 cell window around p. Random per-cell jitter of up to ±0.5 means
    // any cell outside this window is at least 1.5 units away, where the bell
    // weight has decayed to zero.
    for (var i: i32 = -1; i <= 2; i++) {
        for (var j: i32 = -1; j <= 2; j++) {
            let grid_offset = vec2<f32>(f32(i), f32(j));
            let grid_point = p_int + grid_offset;
            let random_offset = hash2(grid_point) * 0.5;
            let v = p_frac - grid_offset - random_offset;

            let sqr_dist = dot(v, v);
            // exp(-2r²) bell, biased so it actually hits 0 at r = 1.5
            // (avoids subtle grid-edge banding).
            let weight = max(0.0, exp(-sqr_dist * 2.0) - 0.01111);
            weight_sum += weight;

            // Wave runs along the slope (not across it). cos() at the cell
            // center → zero-slope inputs evaluate to wave maximum, so peaks
            // and ridges naturally see no erosion.
            let wave_input = dot(v, side_dir);
            let s = -sin(wave_input);
            acc += vec3<f32>(cos(wave_input), s * side_dir.x, s * side_dir.y) * weight;
        }
    }

    return acc / weight_sum;
}

struct ErosionParams {
    // Overall horizontal scale of the gully pattern. Try (mountain width / 5..10).
    scale: f32,
    // Gully depth, relative to scale. 0..0.5 is the useful range.
    strength: f32,
    // Power applied to the slope length before sampling gullies.
    // 1.0 = smooth peaks/ridges. 0.5 = sharper.
    slope_power: f32,
    // Cell size relative to scale. ≈1 is nominal; smaller is grainier,
    // larger is curvier and eventually chaotic.
    cell_scale: f32,
    // Octaves of gullies to stack. 3..8 typical.
    octaves: i32,
    // Per-octave amplitude multiplier.
    gain: f32,
    // Per-octave frequency multiplier.
    lacunarity: f32,
    // -1 = pure carve, +1 = pure raise, 0 ≈ neutral (slight raise in practice).
    height_offset: f32,
};

// Default parameters matching the Shadertoy reference.
fn erosion_params_default() -> ErosionParams {
    return ErosionParams(
        0.08333,  // scale
        0.16,     // strength
        0.6,      // slope_power
        1.0,      // cell_scale
        5,        // octaves
        0.5,      // gain
        2.0,      // lacunarity
        -0.5,     // height_offset
    );
}

// Multi-octave erosion filter. Returns the *delta* (height + slope) to add to
// the input `base_height_and_slope` to produce the eroded surface.
//
// `base_height_and_slope` is vec3(height, d/dx, d/dy) of the input terrain.
// Return value is vec3(delta_height, delta_d/dx, delta_d/dy).
//
// Each octave reads the running slope (modified by previous octaves) so smaller
// gullies branch off larger ones at angles. The slope is rescaled by
// pow(|slope|, slope_power) before sampling, *without* affecting the final
// gradient — this controls peak/ridge sharpness.
fn erosion(p: vec2<f32>, base_height_and_slope: vec3<f32>, params: ErosionParams) -> vec3<f32> {
    let input_h_and_s = base_height_and_slope;
    var h_and_s = base_height_and_slope;

    var freq = 1.0 / (params.scale * params.cell_scale);
    var strength = params.strength * params.scale;

    for (var i: i32 = 0; i < params.octaves; i++) {
        let sqr_len = dot(h_and_s.yz, h_and_s.yz);
        // Equivalent to slope * pow(|slope|, slope_power - 1).
        var input_slope = vec2<f32>(0.0);
        if (sqr_len > 0.0) {
            input_slope = h_and_s.yz * pow(sqr_len, 0.5 * (params.slope_power - 1.0));
        }

        let g = gullies(p * freq, input_slope * params.cell_scale);
        h_and_s += g * strength * vec3<f32>(1.0, freq, freq);

        strength *= params.gain;
        freq *= params.lacunarity;
    }

    return h_and_s - input_h_and_s;
}

// Convenience: apply the height_offset bias and add the delta to the base.
// This is the textbook way to consume `erosion` when you don't care about
// keeping the delta separate.
fn apply_erosion(p: vec2<f32>, base_height_and_slope: vec3<f32>, params: ErosionParams) -> vec3<f32> {
    let delta = erosion(p, base_height_and_slope, params);
    let total_strength = params.scale * params.strength * magnitude_sum(params.octaves, params.gain);
    let offset = total_strength * params.height_offset;
    return base_height_and_slope + delta + vec3<f32>(offset, 0.0, 0.0);
}
