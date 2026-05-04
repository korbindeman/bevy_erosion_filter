//! Pure-Rust port of the erosion filter, mirroring the WGSL implementation in
//! `assets/shaders/erosion.wgsl` numerically. Useful for offline baking,
//! parity tests, and applications that need to evaluate the same field on
//! both CPU and GPU.
//!
//! Mathematical procedure derived from Rune Skovbo Johansen's Shadertoy
//! (see the WGSL header and third-party notices for attribution).

use glam::{Vec2, Vec2Swizzles, Vec3, Vec4};

const TAU: f32 = std::f32::consts::TAU;

#[inline]
fn fract1(v: f32) -> f32 {
    v - v.floor()
}

#[inline]
fn fract2(v: Vec2) -> Vec2 {
    Vec2::new(fract1(v.x), fract1(v.y))
}

#[inline]
fn floor2(v: Vec2) -> Vec2 {
    Vec2::new(v.x.floor(), v.y.floor())
}

/// Inigo Quilez 2D hash. Returns a vec2 in roughly [-1, 1].
#[allow(clippy::approx_constant)]
pub fn hash2(x_in: Vec2) -> Vec2 {
    let k = Vec2::new(0.318_309_9, 0.367_879_4);
    let x = x_in * k + k.yx();
    let scalar = fract1(x.x * x.y * (x.x + x.y));
    let v = Vec2::splat(16.0) * k * Vec2::splat(scalar);
    -Vec2::ONE + 2.0 * fract2(v)
}

/// Inigo Quilez gradient noise with analytical derivatives.
/// Returns `Vec3(value, d/dx, d/dy)`.
pub fn noised(p: Vec2) -> Vec3 {
    let i = floor2(p);
    let f = fract2(p);

    let u = f * f * f * (f * (f * 6.0 - Vec2::splat(15.0)) + Vec2::splat(10.0));
    let du = Vec2::splat(30.0) * f * f * (f * (f - Vec2::splat(2.0)) + Vec2::ONE);

    let ga = hash2(i + Vec2::new(0.0, 0.0));
    let gb = hash2(i + Vec2::new(1.0, 0.0));
    let gc = hash2(i + Vec2::new(0.0, 1.0));
    let gd = hash2(i + Vec2::new(1.0, 1.0));

    let va = ga.dot(f - Vec2::new(0.0, 0.0));
    let vb = gb.dot(f - Vec2::new(1.0, 0.0));
    let vc = gc.dot(f - Vec2::new(0.0, 1.0));
    let vd = gd.dot(f - Vec2::new(1.0, 1.0));

    let value = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd);
    let deriv = ga
        + u.x * (gb - ga)
        + u.y * (gc - ga)
        + u.x * u.y * (ga - gb - gc + gd)
        + du * (u.yx() * (va - vb - vc + vd) + Vec2::new(vb, vc) - Vec2::splat(va));

    Vec3::new(value, deriv.x, deriv.y)
}

/// Standard fBm built from `noised`, returning `Vec3(value, d/dx, d/dy)`.
pub fn fbm(p: Vec2, frequency: f32, octaves: i32, lacunarity: f32, gain: f32) -> Vec3 {
    let mut n = Vec3::ZERO;
    let mut freq = frequency;
    let mut amp = 1.0;
    for _ in 0..octaves {
        n += noised(p * freq) * amp * Vec3::new(1.0, freq, freq);
        amp *= gain;
        freq *= lacunarity;
    }
    n
}

#[inline]
fn mix_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn clamp01(t: f32) -> f32 {
    t.clamp(0.0, 1.0)
}

#[inline]
fn pow_inv(t: f32, power: f32) -> f32 {
    1.0 - (1.0 - clamp01(t)).powf(power)
}

#[inline]
fn ease_out(t: f32) -> f32 {
    let v = 1.0 - clamp01(t);
    1.0 - v * v
}

#[inline]
fn smooth_start(t: f32, smoothing: f32) -> f32 {
    if t >= smoothing {
        return t - 0.5 * smoothing;
    }
    0.5 * t * t / smoothing
}

#[inline]
fn safe_normalize(n: Vec2) -> Vec2 {
    let l = n.length();
    if l.abs() > 1e-10 { n / l } else { n }
}

#[inline]
fn glsl_sign(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Rune Skovbo Johansen's Phacelle Noise.
///
/// Returns `Vec4(cos_wave, sin_wave, side_dir_x, side_dir_y)`. The direction
/// vector is pre-multiplied by `freq * TAU`, matching the WGSL implementation,
/// so callers can multiply it by the sine wave to get the approximate
/// derivative of the cosine wave.
pub fn phacelle_noise(
    p: Vec2,
    norm_dir: Vec2,
    freq: f32,
    offset_cycles: f32,
    normalization: f32,
) -> Vec4 {
    let side_dir = norm_dir.yx() * Vec2::new(-1.0, 1.0) * freq * TAU;
    let offset = offset_cycles * TAU;

    let p_int = floor2(p);
    let p_frac = fract2(p);

    let mut phase_dir = Vec2::ZERO;
    let mut weight_sum = 0.0;

    for i in -1..=2 {
        for j in -1..=2 {
            let grid_offset = Vec2::new(i as f32, j as f32);
            let grid_point = p_int + grid_offset;
            let random_offset = hash2(grid_point) * 0.5;
            let v = p_frac - grid_offset - random_offset;

            let sqr_dist = v.dot(v);
            let weight = ((-sqr_dist * 2.0).exp() - 0.011_11).max(0.0);
            weight_sum += weight;

            let wave_input = v.dot(side_dir) + offset;
            phase_dir += Vec2::new(wave_input.cos(), wave_input.sin()) * weight;
        }
    }

    let interpolated = phase_dir / weight_sum;
    let mag_raw = interpolated.length();
    let magnitude = (1.0 - normalization).max(mag_raw);
    let normalized = interpolated / magnitude;
    Vec4::new(normalized.x, normalized.y, side_dir.x, side_dir.y)
}

/// Raw advanced erosion parameters matching `ErosionFilterParams` in
/// `assets/shaders/erosion.wgsl`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErosionFilterParams {
    pub scale: f32,
    pub strength: f32,
    pub gully_weight: f32,
    pub detail: f32,
    pub rounding: Vec4,
    pub onset: Vec4,
    pub assumed_slope: Vec2,
    pub cell_scale: f32,
    pub normalization: f32,
    pub octaves: i32,
    pub lacunarity: f32,
    pub gain: f32,
}

impl Default for ErosionFilterParams {
    fn default() -> Self {
        Self {
            scale: 0.15,
            strength: 0.22,
            gully_weight: 0.5,
            detail: 1.5,
            rounding: Vec4::new(0.1, 0.0, 0.1, 2.0),
            onset: Vec4::new(1.25, 1.25, 2.8, 1.5),
            assumed_slope: Vec2::new(0.7, 1.0),
            cell_scale: 0.7,
            normalization: 0.5,
            octaves: 5,
            lacunarity: 2.0,
            gain: 0.5,
        }
    }
}

/// Result of [`erosion_filter`], matching the WGSL `ErosionFilterResult`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErosionFilterResult {
    pub delta: Vec3,
    pub magnitude: f32,
    pub ridge_map: f32,
    pub debug: f32,
}

/// Raw advanced erosion filter.
///
/// This is the CPU mirror of the WGSL `erosion_filter` function. It returns the
/// height/slope delta, the accumulated magnitude, a ridge map useful for
/// drainage-style masks, and the final debug fade target.
pub fn erosion_filter(
    p: Vec2,
    base_height_and_slope: Vec3,
    fade_target_in: f32,
    params: &ErosionFilterParams,
) -> ErosionFilterResult {
    let mut strength = params.strength * params.scale;
    let mut fade_target = fade_target_in.clamp(-1.0, 1.0);

    let input_h_and_s = base_height_and_slope;
    let mut h_and_s = base_height_and_slope;

    let mut freq = 1.0 / (params.scale * params.cell_scale);
    let slope = Vec2::new(h_and_s.y, h_and_s.z);
    let slope_length = slope.length().max(1e-10);
    let mut magnitude = 0.0;
    let mut rounding_mult = 1.0;

    let rounding_for_input = mix_f32(
        params.rounding.y,
        params.rounding.x,
        clamp01(fade_target + 0.5),
    ) * params.rounding.z;
    let mut combi_mask = ease_out(smooth_start(
        slope_length * params.onset.x,
        rounding_for_input * params.onset.x,
    ));

    let mut ridge_map_combi_mask = ease_out(slope_length * params.onset.z);
    let mut ridge_map_fade_target = fade_target;

    let mut gully_slope = slope.lerp(
        slope / slope_length * params.assumed_slope.x,
        params.assumed_slope.y,
    );

    for _ in 0..params.octaves {
        let phacelle = phacelle_noise(
            p * freq,
            safe_normalize(gully_slope),
            params.cell_scale,
            0.25,
            params.normalization,
        );
        let p_zw = Vec2::new(phacelle.z, phacelle.w) * -freq;
        let sloping = phacelle.y.abs();

        gully_slope += glsl_sign(phacelle.y) * p_zw * strength * params.gully_weight;

        let octave_h_and_s = Vec3::new(phacelle.x, phacelle.y * p_zw.x, phacelle.y * p_zw.y);
        let faded =
            Vec3::new(fade_target, 0.0, 0.0).lerp(octave_h_and_s * params.gully_weight, combi_mask);
        h_and_s += faded * strength;
        magnitude += strength;

        fade_target = faded.x;

        let rounding_for_octave = mix_f32(
            params.rounding.y,
            params.rounding.x,
            clamp01(phacelle.x + 0.5),
        ) * rounding_mult;
        let new_mask = ease_out(smooth_start(
            sloping * params.onset.y,
            rounding_for_octave * params.onset.y,
        ));
        combi_mask = pow_inv(combi_mask, params.detail) * new_mask;

        ridge_map_fade_target = mix_f32(
            ridge_map_fade_target,
            octave_h_and_s.x,
            ridge_map_combi_mask,
        );
        let new_ridge_mask = ease_out(sloping * params.onset.w);
        ridge_map_combi_mask *= new_ridge_mask;

        strength *= params.gain;
        freq *= params.lacunarity;
        rounding_mult *= params.rounding.w;
    }

    ErosionFilterResult {
        delta: h_and_s - input_h_and_s,
        magnitude,
        ridge_map: ridge_map_fade_target * (1.0 - ridge_map_combi_mask),
        debug: fade_target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same property for the full `noised` building block — guards against
    /// regressions in the IQ noise port.
    #[test]
    fn noised_analytical_gradient_matches_finite_difference() {
        let h = 1e-4_f32;
        for &p in &[
            Vec2::new(0.13, 0.27),
            Vec2::new(1.7, -3.2),
            Vec2::new(50.5, 77.25),
        ] {
            let n = noised(p);
            let dx_num = (noised(p + Vec2::new(h, 0.0)).x - n.x) / h;
            let dy_num = (noised(p + Vec2::new(0.0, h)).x - n.x) / h;
            assert!(
                (n.y - dx_num).abs() < 1e-2,
                "noised dx mismatch at {p:?}: {} vs {dx_num}",
                n.y
            );
            assert!(
                (n.z - dy_num).abs() < 1e-2,
                "noised dy mismatch at {p:?}: {} vs {dy_num}",
                n.z
            );
        }
    }

    #[test]
    fn phacelle_noise_default_path_is_finite() {
        for &p in &[
            Vec2::new(0.13, 0.27),
            Vec2::new(1.7, -3.2),
            Vec2::new(50.5, 77.25),
        ] {
            let n = phacelle_noise(p, Vec2::new(0.6, -0.8), 0.7, 0.25, 0.5);
            assert!(n.x.is_finite());
            assert!(n.y.is_finite());
            assert!(n.z.is_finite());
            assert!(n.w.is_finite());
        }
    }

    #[test]
    fn erosion_filter_default_result_is_finite() {
        let params = ErosionFilterParams::default();
        for &p in &[
            Vec2::new(0.42, 0.31),
            Vec2::new(0.7, 0.85),
            Vec2::new(0.13, 0.66),
        ] {
            let base = fbm(p, 3.0, 3, 2.0, 0.1) * 0.125 * Vec3::new(0.5, 0.5, 0.5)
                + Vec3::new(0.5, 0.0, 0.0);
            let fade_target = ((base.x - 0.5) / (0.125 * 0.6)).clamp(-1.0, 1.0);
            let result = erosion_filter(p, base, fade_target, &params);

            assert!(result.delta.x.is_finite());
            assert!(result.delta.y.is_finite());
            assert!(result.delta.z.is_finite());
            assert!(result.magnitude.is_finite());
            assert!(result.ridge_map.is_finite());
            assert!(result.debug.is_finite());
        }
    }

    #[test]
    fn erosion_filter_magnitude_tracks_octave_strength_sum() {
        let params = ErosionFilterParams::default();
        let result = erosion_filter(
            Vec2::new(0.42, 0.31),
            Vec3::new(0.5, 0.1, -0.2),
            0.0,
            &params,
        );
        let mut expected = 0.0;
        let mut strength = params.strength * params.scale;
        for _ in 0..params.octaves {
            expected += strength;
            strength *= params.gain;
        }
        assert!((result.magnitude - expected).abs() < 1e-6);
    }

    #[test]
    fn erosion_filter_flat_input_is_finite() {
        let params = ErosionFilterParams::default();
        let result = erosion_filter(Vec2::new(0.25, 0.75), Vec3::ZERO, 0.0, &params);
        assert!(result.delta.x.is_finite());
        assert!(result.delta.y.is_finite());
        assert!(result.delta.z.is_finite());
        assert!(result.magnitude.is_finite());
        assert!(result.ridge_map.is_finite());
        assert!(result.debug.is_finite());
    }
}
