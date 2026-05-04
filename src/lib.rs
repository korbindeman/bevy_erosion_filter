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
//!     erosion, apply_erosion, gullies, fbm,
//!     ErosionParams, erosion_params_default,
//! }
//!
//! // Get base height + analytical gradient from your own height function:
//! let base = fbm(uv, 3.0, 4, 2.0, 0.5);
//! let eroded = apply_erosion(uv, base, erosion_params_default());
//! let height = eroded.x;
//! let grad = eroded.yz;
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
//! let params = cpu::ErosionParams::default();
//! let eroded = cpu::apply_erosion(p, base, &params);
//! println!("eroded height = {}", eroded.x);
//! ```

use bevy::prelude::*;

pub mod cpu;

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

/// GPU uniform layout matching the WGSL `ErosionParams` struct, plus a
/// matching CPU-side struct for parity tests.
///
/// Layout matches the WGSL declaration field-for-field. WGSL uniform alignment
/// rules: 8 floats + 1 i32 = 36 bytes, padded to 48 by the std140 packing.
/// `ShaderType` derives the right padding when used as a Bevy uniform.
#[derive(Clone, Copy, Debug, PartialEq, bevy::render::render_resource::ShaderType)]
#[repr(C)]
pub struct ErosionParamsGpu {
    pub scale: f32,
    pub strength: f32,
    pub slope_power: f32,
    pub cell_scale: f32,
    pub octaves: i32,
    pub gain: f32,
    pub lacunarity: f32,
    pub height_offset: f32,
}

impl Default for ErosionParamsGpu {
    fn default() -> Self {
        Self::from_cpu(&cpu::ErosionParams::default())
    }
}

impl ErosionParamsGpu {
    pub fn from_cpu(p: &cpu::ErosionParams) -> Self {
        Self {
            scale: p.scale,
            strength: p.strength,
            slope_power: p.slope_power,
            cell_scale: p.cell_scale,
            octaves: p.octaves,
            gain: p.gain,
            lacunarity: p.lacunarity,
            height_offset: p.height_offset,
        }
    }
}
