# Changelog

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
