//! GPU-friendly per-fragment erosion filter for Bevy.
//!
//! WGSL port of Rune Skovbo Johansen's erosion noise from
//! <https://www.shadertoy.com/view/wXcfWn>. The shader library can be
//! imported into your own Bevy materials, and this crate also provides a
//! pure-Rust CPU implementation for offline baking and parity testing.
//!
//! # WGSL usage
//!
//! Add [`ErosionFilterPlugin`] to your app, then in your shader:
//!
//! ```wgsl
//! #import bevy_erosion_filter::erosion::{
//!     fbm, erosion_filter, erosion_filter_params_default,
//! }
//!
//! // Get base height + analytical gradient from your own height function:
//! let base = fbm(uv, 3.0, 4, 2.0, 0.5);
//! let fade_target = clamp(base.x / 0.1, -1.0, 1.0);
//! let filtered = erosion_filter(uv, base, fade_target, erosion_filter_params_default());
//! let height = base.x + filtered.delta.x;
//! let grad = base.yz + filtered.delta.yz;
//! let ridge_map = filtered.ridge_map;
//! ```
//!
//! # CPU usage
//!
//! ```
//! use bevy_erosion_filter::cpu;
//! use glam::Vec2;
//!
//! let p = Vec2::new(1.0, 2.0);
//! let base = cpu::fbm(p, 3.0, 4, 2.0, 0.5);
//! let params = cpu::ErosionFilterParams::default();
//! let filtered = cpu::erosion_filter(p, base, base.x.clamp(-1.0, 1.0), &params);
//! println!("eroded height = {}", base.x + filtered.delta.x);
//! ```
//!
//! # Raw WGSL source
//!
//! [`EROSION_WGSL`] exposes the shader source as a `&'static str`, available
//! regardless of the `bevy` feature. Use this when you need to feed the WGSL
//! to `wgpu` (or another backend) directly, outside Bevy's asset/shader
//! loader — for example, offline bake CLIs that run compute pipelines without
//! a Bevy `App`.
//!
//! # Cargo features
//!
//! `bevy` (default) — pulls in Bevy and registers the WGSL shader library
//! plus the [`ErosionFilterParamsGpu`] uniform layout. Disable with
//! `default-features = false` to use only the pure-Rust [`cpu`] module and
//! [`EROSION_WGSL`] from non-Bevy crates.

pub mod cpu;

/// Raw WGSL source for the erosion shader library.
///
/// Identical to the file Bevy loads via `ErosionFilterPlugin` when the `bevy`
/// feature is enabled. Use this when you need to feed the shader to `wgpu`
/// directly, outside Bevy's asset pipeline.
pub const EROSION_WGSL: &str = include_str!("../assets/shaders/erosion.wgsl");

#[cfg(feature = "bevy")]
mod bevy_integration {
    use crate::cpu;
    use bevy::prelude::*;

    /// Bevy plugin that registers the erosion WGSL as a shader library.
    ///
    /// After adding this plugin you can import the library from your own shader as
    /// `bevy_erosion_filter::erosion`.
    pub struct ErosionFilterPlugin;

    impl Plugin for ErosionFilterPlugin {
        fn build(&self, app: &mut App) {
            bevy::shader::load_shader_library!(app, "../assets/shaders/erosion.wgsl");
        }
    }

    /// GPU uniform layout matching the WGSL `ErosionFilterParams` struct.
    ///
    /// Use this when driving the raw advanced filter from Bevy uniforms. The
    /// fields map directly to [`cpu::ErosionFilterParams`] and the WGSL
    /// `ErosionFilterParams` declaration.
    #[derive(Clone, Copy, Debug, PartialEq, bevy::render::render_resource::ShaderType)]
    #[repr(C)]
    pub struct ErosionFilterParamsGpu {
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

    impl Default for ErosionFilterParamsGpu {
        fn default() -> Self {
            Self::from_cpu(&cpu::ErosionFilterParams::default())
        }
    }

    impl ErosionFilterParamsGpu {
        pub fn from_cpu(p: &cpu::ErosionFilterParams) -> Self {
            Self {
                scale: p.scale,
                strength: p.strength,
                gully_weight: p.gully_weight,
                detail: p.detail,
                rounding: p.rounding,
                onset: p.onset,
                assumed_slope: p.assumed_slope,
                cell_scale: p.cell_scale,
                normalization: p.normalization,
                octaves: p.octaves,
                lacunarity: p.lacunarity,
                gain: p.gain,
            }
        }
    }
}

#[cfg(feature = "bevy")]
pub use bevy_integration::{ErosionFilterParamsGpu, ErosionFilterPlugin};
