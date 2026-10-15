# Changelog

## 0.3.0

- Updates to Bevy 0.20 (and glam 0.33). Requires consumers to be on Bevy 0.20.
- Shader library is now a WESL module at `src/erosion.wesl`. Bevy 0.20 no longer
  honors `#define_import_path`; the public import path remains
  `bevy_erosion_filter::erosion` because that is the embedded module path
  (`embedded://bevy_erosion_filter/erosion.wesl`).
- Updates the terrain demo to WESL (`import` / `constants::MATERIAL_BIND_GROUP`)
  and `bevy_egui` 0.43.
- MSRV is now Rust 1.97, matching Bevy 0.20.

## 0.2.0

- Updates to Bevy 0.19 (and glam 0.32). Requires consumers to be on Bevy 0.19.

## 0.1.2

- Exposes the erosion WGSL source as a public `EROSION_WGSL` constant so
  non-Bevy consumers (offline bake CLIs running `wgpu` directly, other engines)
  can use the shader without locating it in Cargo's registry cache.
  Available regardless of the `bevy` feature.

## 0.1.0 - Initial Release

- Adds `ErosionFilterPlugin`, registering `assets/shaders/erosion.wgsl` as
  `bevy_erosion_filter::erosion`.
- Exposes WGSL functions for analytical-gradient fBm, single-octave gullies,
  multi-octave erosion, and default erosion parameters.
- Adds a pure-Rust CPU mirror under `bevy_erosion_filter::cpu` for offline
  baking and parity tests.
- Includes an interactive Bevy 0.18 terrain demo.
