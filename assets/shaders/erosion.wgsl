// =====================================================================================
// bevy_erosion_filter — WGSL port of Rune Skovbo Johansen's gully-erosion noise.
//
// Upstream: https://www.shadertoy.com/view/wXcfWn
//
// The erosion functions are derivative works of Rune Skovbo Johansen's GLSL
// implementation, ported to WGSL under the Mozilla Public License v2. The IQ
// gradient noise (`noised`) and IQ hash (`hash2`) are bundled here because the
// algorithm requires them; they are reused under MIT.
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
// gradient can be passed straight into `erosion_filter` as the input slope.
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

// =====================================================================================
//  Raw advanced erosion filter.
//
//  Direct port of Rune Skovbo Johansen's `ErosionFilter` from the Shadertoy
//  reference (https://www.shadertoy.com/view/wXcfWn). It takes per-octave
//  masking by slope (`onset`), separate rounding for ridges and creases
//  (`rounding`), an `assumed_slope` override for the input gradient, a fade
//  target that lets each octave bias toward a desired terrain mean, and it
//  returns a **ridge map** for drainage paint and tree placement.
// =====================================================================================

const TAU: f32 = 6.28318530717959;

fn pow_inv(t: f32, power: f32) -> f32 {
    return 1.0 - pow(1.0 - clamp(t, 0.0, 1.0), power);
}

fn ease_out(t: f32) -> f32 {
    let v = 1.0 - clamp(t, 0.0, 1.0);
    return 1.0 - v * v;
}

fn smooth_start(t: f32, smoothing: f32) -> f32 {
    if (t >= smoothing) {
        return t - 0.5 * smoothing;
    }
    return 0.5 * t * t / smoothing;
}

fn safe_normalize(n: vec2<f32>) -> vec2<f32> {
    let l = length(n);
    if (abs(l) > 1e-10) {
        return n / l;
    }
    return n;
}

// Phacelle Noise (Rune Skovbo Johansen, MPL-2.0). A "phase + cell" stripe
// pattern aligned with `norm_dir`, normalised to roughly unit amplitude. The
// cosine wave is in `.x`, the sine wave in `.y`, and `.zw` is the side-direction
// vector pre-multiplied by `freq * TAU` so callers can multiply with the sine
// to get the cosine's analytical derivative. `freq` is best kept near 1; values
// well above produce distortion. `offset` is a phase shift in cycles.
// `normalization` (in [0,1]) is how aggressively to renormalise small
// magnitudes — high values produce loop-shaped meeting points.
fn phacelle_noise(
    p: vec2<f32>,
    norm_dir: vec2<f32>,
    freq: f32,
    offset_cycles: f32,
    normalization: f32,
) -> vec4<f32> {
    let side_dir = norm_dir.yx * vec2<f32>(-1.0, 1.0) * freq * TAU;
    let offset = offset_cycles * TAU;

    let p_int = floor(p);
    let p_frac = fract(p);

    var phase_dir = vec2<f32>(0.0);
    var weight_sum = 0.0;

    // 4×4 cell window. Random per-cell jitter of up to ±0.5 means any cell
    // outside this window is at least 1.5 units away, where the bell weight has
    // decayed to zero.
    for (var i: i32 = -1; i <= 2; i++) {
        for (var j: i32 = -1; j <= 2; j++) {
            let grid_offset = vec2<f32>(f32(i), f32(j));
            let grid_point = p_int + grid_offset;
            let random_offset = hash2(grid_point) * 0.5;
            let v = p_frac - grid_offset - random_offset;

            let sqr_dist = dot(v, v);
            let w = max(0.0, exp(-sqr_dist * 2.0) - 0.01111);
            weight_sum += w;

            let wave_input = dot(v, side_dir) + offset;
            phase_dir += vec2<f32>(cos(wave_input), sin(wave_input)) * w;
        }
    }

    let interpolated = phase_dir / weight_sum;
    let mag_raw = sqrt(dot(interpolated, interpolated));
    let magnitude = max(1.0 - normalization, mag_raw);
    return vec4<f32>(interpolated / magnitude, side_dir);
}

struct ErosionFilterParams {
    // Overall horizontal scale of the gully pattern (mountain width / 5..10).
    scale: f32,
    // Overall vertical strength multiplier. Pre-multiplied by `scale` inside
    // the filter, so changing scale alone keeps gully depths sane.
    strength: f32,
    // Magnitude of the gully contribution inside the slope band, [0..1].
    // 0 still sharpens peaks/valleys but leaves no gullies; 1 gives full
    // gullies but leaves peaks/valleys rounded.
    gully_weight: f32,
    // Detail mask falloff. Lower values keep high-frequency gullies confined
    // to steeper slopes; higher values let them bleed onto flatter areas.
    detail: f32,
    // (ridges, creases, init_mult, octave_mult). Per-octave rounding strengths
    // for ridges (.x) and creases (.y), with .z scaling rounding on the
    // initial heightfield, and .w the per-octave decay.
    rounding: vec4<f32>,
    // (init, octave, ridgemap_init, ridgemap_octave). Slope-onset thresholds
    // controlling how steep terrain has to be before erosion (.xy) and the
    // ridge map (.zw) kick in.
    onset: vec4<f32>,
    // (slope_value, blend). The actual input gradient often shapes erosion
    // poorly; this lets you mix in an idealised slope magnitude.
    assumed_slope: vec2<f32>,
    // Voronoi cell size relative to scale. ≈1 nominal; smaller is grainier;
    // values >1 produce curvier flow lines and eventually chaotic output.
    cell_scale: f32,
    // Phacelle normalisation, [0..1]. 0 leaves stripes weak in flat
    // regions; values close to 1 force unit amplitude everywhere and can
    // create unnatural loop-shaped meeting points.
    normalization: f32,
    // Octaves of gully noise. 3..8 typical; cost is roughly linear.
    octaves: i32,
    // Per-octave frequency multiplier.
    lacunarity: f32,
    // Per-octave amplitude multiplier.
    gain: f32,
};

struct ErosionFilterResult {
    // (delta_height, delta_d/dx, delta_d/dy) to add to the input.
    delta: vec3<f32>,
    // Sum of per-octave strengths actually applied. Useful for normalising the
    // height offset bias on the caller side.
    magnitude: f32,
    // Approximate ridge map: ≈ +1 on ridges, ≈ -1 in creases, ≈ 0 on flats.
    // Drives drainage paint and tree placement in the reference demo.
    ridge_map: f32,
    // Final per-pixel fade target (debug — exposes the inner state of the
    // last octave's mix-toward-mean).
    debug: f32,
};

// Defaults matching the Shadertoy demo. Animated parameters are pinned to
// their middle values.
fn erosion_filter_params_default() -> ErosionFilterParams {
    return ErosionFilterParams(
        0.15,                              // scale
        0.22,                              // strength
        0.5,                               // gully_weight
        1.5,                               // detail
        vec4<f32>(0.1, 0.0, 0.1, 2.0),     // rounding
        vec4<f32>(1.25, 1.25, 2.8, 1.5),   // onset
        vec2<f32>(0.7, 1.0),               // assumed_slope
        0.7,                               // cell_scale
        0.5,                               // normalization
        5,                                 // octaves
        2.0,                               // lacunarity
        0.5,                               // gain
    );
}

// Multi-octave erosion filter with masking, rounding, and a ridge-map output.
// Direct WGSL port of Rune Skovbo Johansen's reference `ErosionFilter`.
//
// `base_height_and_slope` is vec3(height, d/dx, d/dy) of the input terrain.
// `fade_target` is the target the filter biases toward in flat masked-out
// regions — typically clamp(height / mean_amp, -1, 1) so peaks bias up and
// troughs bias down, preserving the input's extrema through the filter.
fn erosion_filter(
    p: vec2<f32>,
    base_height_and_slope: vec3<f32>,
    fade_target_in: f32,
    params: ErosionFilterParams,
) -> ErosionFilterResult {
    var strength = params.strength * params.scale;
    var fade_target = clamp(fade_target_in, -1.0, 1.0);

    let input_h_and_s = base_height_and_slope;
    var h_and_s = base_height_and_slope;

    var freq = 1.0 / (params.scale * params.cell_scale);
    let slope_length = max(length(h_and_s.yz), 1e-10);
    var magnitude_total = 0.0;
    var rounding_mult = 1.0;

    let rounding_for_input = mix(
        params.rounding.y,
        params.rounding.x,
        clamp(fade_target + 0.5, 0.0, 1.0),
    ) * params.rounding.z;
    var combi_mask = ease_out(smooth_start(
        slope_length * params.onset.x,
        rounding_for_input * params.onset.x,
    ));

    var ridge_map_combi_mask = ease_out(slope_length * params.onset.z);
    var ridge_map_fade_target = fade_target;

    // Optionally substitute a fixed slope magnitude — the actual input slope
    // often produces sub-optimal gully directions on steep features.
    var gully_slope = mix(
        h_and_s.yz,
        h_and_s.yz / slope_length * params.assumed_slope.x,
        params.assumed_slope.y,
    );

    for (var i: i32 = 0; i < params.octaves; i++) {
        let phacelle = phacelle_noise(
            p * freq,
            safe_normalize(gully_slope),
            params.cell_scale,
            0.25,
            params.normalization,
        );
        // Negate to keep slope directions pointing downhill, and pre-multiply
        // by freq because `p` was scaled by freq before sampling.
        let p_zw = phacelle.zw * (-freq);
        let sloping = abs(phacelle.y);

        // Subsequent octaves see a gully slope updated by the current one, so
        // smaller gullies branch off larger ones at angles.
        gully_slope += sign(phacelle.y) * p_zw * strength * params.gully_weight;

        // (height, d/dx, d/dy) of the current gully octave.
        let octave_h_and_s = vec3<f32>(phacelle.x, phacelle.y * p_zw.x, phacelle.y * p_zw.y);
        let faded = mix(
            vec3<f32>(fade_target, 0.0, 0.0),
            octave_h_and_s * params.gully_weight,
            combi_mask,
        );
        h_and_s += faded * strength;
        magnitude_total += strength;

        fade_target = faded.x;

        let rounding_for_octave = mix(
            params.rounding.y,
            params.rounding.x,
            clamp(phacelle.x + 0.5, 0.0, 1.0),
        ) * rounding_mult;
        let new_mask = ease_out(smooth_start(
            sloping * params.onset.y,
            rounding_for_octave * params.onset.y,
        ));
        combi_mask = pow_inv(combi_mask, params.detail) * new_mask;

        ridge_map_fade_target = mix(ridge_map_fade_target, octave_h_and_s.x, ridge_map_combi_mask);
        let new_ridge_mask = ease_out(sloping * params.onset.w);
        ridge_map_combi_mask = ridge_map_combi_mask * new_ridge_mask;

        strength *= params.gain;
        freq *= params.lacunarity;
        rounding_mult *= params.rounding.w;
    }

    let ridge_map = ridge_map_fade_target * (1.0 - ridge_map_combi_mask);
    let delta = h_and_s - input_h_and_s;
    return ErosionFilterResult(delta, magnitude_total, ridge_map, fade_target);
}
