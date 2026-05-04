//! Pure-Rust port of the erosion filter, mirroring the WGSL implementation in
//! `assets/shaders/erosion.wgsl` numerically. Useful for offline baking,
//! parity tests, and applications that need to evaluate the same field on
//! both CPU and GPU.
//!
//! Mathematical procedure derived from Rune Skovbo Johansen's Shadertoy
//! (see the WGSL header and third-party notices for attribution).

use glam::{Vec2, Vec2Swizzles, Vec3};

const PI: f32 = std::f32::consts::PI;

/// Erosion parameters. Field meanings mirror `ErosionParams` in the WGSL
/// library; defaults match the Shadertoy reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErosionParams {
    pub scale: f32,
    pub strength: f32,
    pub slope_power: f32,
    pub cell_scale: f32,
    pub octaves: i32,
    pub gain: f32,
    pub lacunarity: f32,
    pub height_offset: f32,
}

impl Default for ErosionParams {
    fn default() -> Self {
        Self {
            scale: 0.08333,
            strength: 0.16,
            slope_power: 0.6,
            cell_scale: 1.0,
            octaves: 5,
            gain: 0.5,
            lacunarity: 2.0,
            height_offset: -0.5,
        }
    }
}

#[inline]
fn fract2(v: Vec2) -> Vec2 {
    Vec2::new(v.x - v.x.floor(), v.y - v.y.floor())
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
    let scalar = (x.x * x.y * (x.x + x.y)).fract();
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

/// Geometric series 1 + g + g² + … + g^(N-1).
pub fn magnitude_sum(octaves: i32, gain: f32) -> f32 {
    if octaves <= 0 {
        return 0.0;
    }
    if (1.0 - gain).abs() < f32::EPSILON {
        return octaves as f32;
    }
    (1.0 - gain.powi(octaves)) / (1.0 - gain)
}

/// Single octave of gully noise. Returns `Vec3(height, d/dx, d/dy)`,
/// amplitude bounded by 1.
pub fn gullies(p: Vec2, slope: Vec2) -> Vec3 {
    let side_dir = Vec2::new(-slope.y, slope.x) * 2.0 * PI;

    let p_int = floor2(p);
    let p_frac = fract2(p);

    let mut acc = Vec3::ZERO;
    let mut weight_sum = 0.0;

    for i in -1..=2 {
        for j in -1..=2 {
            let grid_offset = Vec2::new(i as f32, j as f32);
            let grid_point = p_int + grid_offset;
            let random_offset = hash2(grid_point) * 0.5;
            let v = p_frac - grid_offset - random_offset;

            let sqr_dist = v.dot(v);
            let weight = (((-sqr_dist * 2.0).exp()) - 0.011_11).max(0.0);
            weight_sum += weight;

            let wave_input = v.dot(side_dir);
            let s = -wave_input.sin();
            acc += Vec3::new(wave_input.cos(), s * side_dir.x, s * side_dir.y) * weight;
        }
    }

    acc / weight_sum
}

/// Multi-octave erosion filter. Returns the *delta* `Vec3(dh, d/dx, d/dy)`
/// to add to `base_height_and_slope`.
pub fn erosion(p: Vec2, base_height_and_slope: Vec3, params: &ErosionParams) -> Vec3 {
    let input_h_and_s = base_height_and_slope;
    let mut h_and_s = base_height_and_slope;

    let mut freq = 1.0 / (params.scale * params.cell_scale);
    let mut strength = params.strength * params.scale;

    for _ in 0..params.octaves {
        let slope_yz = Vec2::new(h_and_s.y, h_and_s.z);
        let sqr_len = slope_yz.dot(slope_yz);
        let input_slope = if sqr_len > 0.0 {
            let factor = sqr_len.powf(0.5 * (params.slope_power - 1.0));
            slope_yz * factor
        } else {
            Vec2::ZERO
        };

        let g = gullies(p * freq, input_slope * params.cell_scale);
        h_and_s += g * strength * Vec3::new(1.0, freq, freq);

        strength *= params.gain;
        freq *= params.lacunarity;
    }

    h_and_s - input_h_and_s
}

/// Convenience: compute the eroded height + slope, including the
/// `height_offset` bias applied uniformly to lift or sink the surface.
pub fn apply_erosion(p: Vec2, base_height_and_slope: Vec3, params: &ErosionParams) -> Vec3 {
    let delta = erosion(p, base_height_and_slope, params);
    let total_strength =
        params.scale * params.strength * magnitude_sum(params.octaves, params.gain);
    let offset = total_strength * params.height_offset;
    base_height_and_slope + delta + Vec3::new(offset, 0.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At zero slope, `gullies` evaluates to wave maximum (cos(0) = 1)
    /// regardless of position. This is the property that prevents erosion at
    /// peaks/ridges.
    #[test]
    fn gullies_at_zero_slope_returns_unit_height() {
        for &p in &[
            Vec2::new(0.0, 0.0),
            Vec2::new(1.7, -3.2),
            Vec2::new(100.0, 50.0),
        ] {
            let g = gullies(p, Vec2::ZERO);
            assert!(
                (g.x - 1.0).abs() < 1e-5,
                "expected height ≈ 1 at zero slope, got {} at p={p:?}",
                g.x
            );
            assert!(g.y.abs() < 1e-5 && g.z.abs() < 1e-5);
        }
    }

    /// `gullies` height is bounded by 1 — the wave's amplitude is exactly cos
    /// of a phase, and the weighted average is convex over per-cell unit-amplitude
    /// values. The 1.05 slack absorbs the bell-weight bias term.
    #[test]
    fn gullies_height_amplitude_bounded() {
        let mut rng_state: u32 = 0x9E37_79B9;
        for _ in 0..2000 {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let px = (rng_state as f32 / u32::MAX as f32) * 100.0 - 50.0;
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let py = (rng_state as f32 / u32::MAX as f32) * 100.0 - 50.0;
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let sx = (rng_state as f32 / u32::MAX as f32) * 4.0 - 2.0;
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let sy = (rng_state as f32 / u32::MAX as f32) * 4.0 - 2.0;

            let g = gullies(Vec2::new(px, py), Vec2::new(sx, sy));
            assert!(
                g.x.abs() <= 1.05,
                "amplitude bound violated: |{}| > 1.05 at p=({px}, {py}), slope=({sx}, {sy})",
                g.x
            );
            assert!(g.x.is_finite() && g.y.is_finite() && g.z.is_finite());
        }
    }

    /// `gullies` analytical derivative is qualitatively coherent with the
    /// numerical one — same sign, similar magnitude — but it is *intentionally*
    /// approximate. The upstream code omits the chain-rule term coming from the
    /// position-dependent cell weights ("it seems to look worse" per the
    /// original Shadertoy comment). This test guards against gross divergence
    /// (wrong sign, order-of-magnitude mismatch); it does not assert equality.
    #[test]
    fn gullies_analytical_gradient_qualitatively_matches_finite_difference() {
        let h = 1e-4_f32;
        let slope = Vec2::new(0.7, -0.3);
        for &p in &[
            Vec2::new(0.5, 0.5),
            Vec2::new(2.31, -4.17),
            Vec2::new(13.0, 7.0),
        ] {
            let g = gullies(p, slope);
            let dx_num = (gullies(p + Vec2::new(h, 0.0), slope).x - g.x) / h;
            let dy_num = (gullies(p + Vec2::new(0.0, h), slope).x - g.x) / h;
            // Loose tolerance: chain-rule term from cell-weight gradient is
            // intentionally dropped upstream, so divergence of ~O(slope) is
            // expected. We only check the result is bounded.
            assert!(g.y.is_finite() && g.z.is_finite());
            assert!(
                (g.y - dx_num).abs() < 5.0,
                "dx wildly off at {p:?}: {} vs {dx_num}",
                g.y
            );
            assert!(
                (g.z - dy_num).abs() < 5.0,
                "dy wildly off at {p:?}: {} vs {dy_num}",
                g.z
            );
        }
    }

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

    /// Numerical gradient of the full eroded surface should match the analytical
    /// gradient returned by `apply_erosion`. This is the core consistency claim
    /// of Rune's port — derivatives must remain coherent across all octaves.
    #[test]
    fn apply_erosion_analytical_gradient_matches_finite_difference() {
        let h = 1e-4_f32;
        let params = ErosionParams::default();
        for &p in &[
            Vec2::new(0.42, 0.31),
            Vec2::new(0.7, 0.85),
            Vec2::new(0.13, 0.66),
        ] {
            let base = fbm(p, 3.0, 3, 2.0, 0.5) * 0.25;
            let eroded = apply_erosion(p, base, &params);

            let base_x = fbm(p + Vec2::new(h, 0.0), 3.0, 3, 2.0, 0.5) * 0.25;
            let base_y = fbm(p + Vec2::new(0.0, h), 3.0, 3, 2.0, 0.5) * 0.25;
            let eroded_x = apply_erosion(p + Vec2::new(h, 0.0), base_x, &params);
            let eroded_y = apply_erosion(p + Vec2::new(0.0, h), base_y, &params);

            let dx_num = (eroded_x.x - eroded.x) / h;
            let dy_num = (eroded_y.x - eroded.x) / h;

            // Looser tolerance: high-frequency octaves amplify finite-difference
            // truncation error. We only need the gradient direction to be
            // qualitatively right.
            assert!(
                (eroded.y - dx_num).abs() < 1.0,
                "eroded dx mismatch at {p:?}: analytical {} vs numerical {dx_num}",
                eroded.y
            );
            assert!(
                (eroded.z - dy_num).abs() < 1.0,
                "eroded dy mismatch at {p:?}: analytical {} vs numerical {dy_num}",
                eroded.z
            );
        }
    }

    /// `magnitude_sum` against the closed form for a few known cases.
    #[test]
    fn magnitude_sum_geometric_series() {
        assert!((magnitude_sum(1, 0.5) - 1.0).abs() < 1e-6);
        assert!((magnitude_sum(2, 0.5) - 1.5).abs() < 1e-6);
        assert!((magnitude_sum(5, 0.5) - 1.9375).abs() < 1e-6);
        assert!((magnitude_sum(5, 1.0) - 5.0).abs() < 1e-6);
        assert_eq!(magnitude_sum(0, 0.5), 0.0);
    }

    /// Exactly flat analytical inputs are common at peaks, troughs, and flat
    /// masks. The slope shaping exponent can be negative, so zero slopes need
    /// an explicit finite path.
    #[test]
    fn apply_erosion_flat_input_is_finite() {
        let params = ErosionParams::default();
        let eroded = apply_erosion(Vec2::new(0.25, 0.75), Vec3::ZERO, &params);
        assert!(eroded.x.is_finite());
        assert!(eroded.y.is_finite());
        assert!(eroded.z.is_finite());
    }
}
