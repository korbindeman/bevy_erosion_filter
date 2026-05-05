//! Real-geometry, shader-shaded terrain demo for `bevy_erosion_filter`.
//!
//! The mesh provides a dense terrain silhouette and side walls. The shader
//! evaluates low-frequency erosion per vertex for displacement and full erosion
//! per fragment for analytical normals, ridge-map coloring, and debug views.
//!
//! ```sh
//! cargo run --example terrain_demo
//! ```

use std::f32::consts::FRAC_PI_2;

use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::tonemapping::Tonemapping,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    mesh::Indices,
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, PrimitiveTopology, ShaderType},
    shader::ShaderRef,
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui, input::EguiWantsInput};
use bevy_erosion_filter::{ErosionFilterPlugin, cpu};

const TERRAIN_SHADER: &str = "shaders/terrain_demo.wgsl";
const TERRAIN_EXTENT: f32 = 22.0;
const TERRAIN_RESOLUTION: usize = 640;
const SIDE_LAYERS: usize = 28;
const BOTTOM_HEIGHT: f32 = -3.2;
const CAMERA_TARGET_HEIGHT: f32 = 1.1;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.45, 0.60, 0.70)))
        .insert_resource(GlobalAmbientLight::NONE)
        .insert_resource(DemoSettings::default())
        .insert_resource(OrbitCamera::default())
        .add_plugins((
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: "examples/assets".to_string(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy_erosion_filter terrain demo".into(),
                        resolution: (1280, 820).into(),
                        ..default()
                    }),
                    ..default()
                }),
            EguiPlugin::default(),
            ErosionFilterPlugin,
            MaterialPlugin::<TerrainMaterial>::default(),
            ErosionTerrainPlugin,
        ))
        .run();
}

struct ErosionTerrainPlugin;

impl Plugin for ErosionTerrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup)
            .add_systems(
                Update,
                (orbit_camera, keyboard_shortcuts, update_terrain_material),
            )
            .add_systems(EguiPrimaryContextPass, controls_ui);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewMode {
    Terrain,
    ErosionDelta,
    RidgeMap,
    DebugFade,
}

impl ViewMode {
    fn label(self) -> &'static str {
        match self {
            ViewMode::Terrain => "Terrain",
            ViewMode::ErosionDelta => "Erosion",
            ViewMode::RidgeMap => "Ridges",
            ViewMode::DebugFade => "Debug",
        }
    }

    fn shader_value(self) -> f32 {
        match self {
            ViewMode::Terrain => 0.0,
            ViewMode::ErosionDelta => 1.0,
            ViewMode::RidgeMap => 2.0,
            ViewMode::DebugFade => 3.0,
        }
    }

    fn next(self) -> Self {
        match self {
            ViewMode::Terrain => ViewMode::ErosionDelta,
            ViewMode::ErosionDelta => ViewMode::RidgeMap,
            ViewMode::RidgeMap => ViewMode::DebugFade,
            ViewMode::DebugFade => ViewMode::Terrain,
        }
    }
}

#[derive(Resource, Clone)]
struct DemoSettings {
    erosion: cpu::ErosionFilterParams,
    height_offset: f32,
    vertical_scale: f32,
    map_scale: f32,
    water_level: f32,
    erosion_enabled: bool,
    view_mode: ViewMode,
    dirty: bool,
}

impl Default for DemoSettings {
    fn default() -> Self {
        Self {
            erosion: cpu::ErosionFilterParams {
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
            },
            height_offset: -0.65,
            vertical_scale: 22.0,
            map_scale: 1.0,
            water_level: -1.54,
            erosion_enabled: true,
            view_mode: ViewMode::Terrain,
            dirty: true,
        }
    }
}

impl DemoSettings {
    fn material_uniform(&self) -> TerrainDemoUniform {
        TerrainDemoUniform {
            erosion0: Vec4::new(
                self.erosion.scale,
                self.erosion.strength,
                self.erosion.gully_weight,
                self.erosion.detail,
            ),
            rounding: self.erosion.rounding,
            onset: self.erosion.onset,
            erosion1: Vec4::new(
                self.erosion.assumed_slope.x,
                self.erosion.assumed_slope.y,
                self.erosion.cell_scale,
                self.erosion.normalization,
            ),
            erosion2: Vec4::new(
                self.erosion.octaves as f32,
                self.erosion.lacunarity,
                self.erosion.gain,
                self.height_offset,
            ),
            terrain0: Vec4::new(
                TERRAIN_EXTENT,
                self.vertical_scale,
                self.map_scale,
                BOTTOM_HEIGHT,
            ),
            terrain1: Vec4::new(
                if self.erosion_enabled { 1.0 } else { 0.0 },
                self.view_mode.shader_value(),
                0.62,
                self.water_level,
            ),
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct TerrainMaterial {
    #[uniform(0)]
    settings: TerrainDemoUniform,
}

#[derive(ShaderType, Debug, Clone, Copy)]
struct TerrainDemoUniform {
    erosion0: Vec4,
    rounding: Vec4,
    onset: Vec4,
    erosion1: Vec4,
    erosion2: Vec4,
    terrain0: Vec4,
    terrain1: Vec4,
}

impl Material for TerrainMaterial {
    fn vertex_shader() -> ShaderRef {
        TERRAIN_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        TERRAIN_SHADER.into()
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }
}

#[derive(Component)]
struct Terrain;

#[derive(Component)]
struct Water;

#[derive(Resource)]
struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    radius: f32,
    target: Vec3,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            yaw: -0.52,
            pitch: 0.16,
            radius: 15.0,
            target: Vec3::new(0.0, CAMERA_TARGET_HEIGHT * 0.6, -1.2),
        }
    }
}

fn setup(
    mut commands: Commands,
    settings: Res<DemoSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
) {
    let mesh = meshes.add(build_terrain_mesh());
    let material = terrain_materials.add(TerrainMaterial {
        settings: settings.material_uniform(),
    });

    commands.spawn((
        Terrain,
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::IDENTITY,
    ));

    commands.spawn((
        Name::new("Water"),
        Water,
        Mesh3d(
            meshes.add(
                Plane3d::default()
                    .mesh()
                    .size(TERRAIN_EXTENT * 0.94, TERRAIN_EXTENT * 0.94),
            ),
        ),
        MeshMaterial3d(standard_materials.add(StandardMaterial {
            base_color: Color::srgba(0.05, 0.24, 0.34, 0.72),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            perceptual_roughness: 0.18,
            reflectance: 0.68,
            ..default()
        })),
        Transform::from_xyz(0.0, settings.water_level, 0.0),
    ));

    commands.spawn((
        Camera3d::default(),
        camera_transform(&OrbitCamera::default()),
        Tonemapping::AcesFitted,
    ));
}

fn controls_ui(mut contexts: EguiContexts, mut settings: ResMut<DemoSettings>) -> Result {
    let ctx = contexts.ctx_mut()?;
    let settings = &mut *settings;
    let mut changed = false;

    egui::Window::new("Erosion filter")
        .default_width(310.0)
        .default_pos(egui::pos2(16.0, 16.0))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                changed |= ui
                    .checkbox(&mut settings.erosion_enabled, "enabled")
                    .changed();
                if ui.button("reset").clicked() {
                    *settings = DemoSettings::default();
                    changed = true;
                }
            });

            ui.separator();

            ui.label("View");
            ui.horizontal(|ui| {
                for mode in [
                    ViewMode::Terrain,
                    ViewMode::ErosionDelta,
                    ViewMode::RidgeMap,
                    ViewMode::DebugFade,
                ] {
                    changed |= ui
                        .selectable_value(&mut settings.view_mode, mode, mode.label())
                        .changed();
                }
            });

            ui.separator();
            ui.label("Filter");
            changed |= slider(ui, &mut settings.erosion.scale, 0.04..=0.32, "scale");
            changed |= slider(ui, &mut settings.erosion.strength, 0.0..=0.42, "strength");
            changed |= slider(
                ui,
                &mut settings.erosion.gully_weight,
                0.0..=1.0,
                "gully weight",
            );
            changed |= slider(ui, &mut settings.erosion.detail, 0.2..=4.0, "detail");
            changed |= slider(
                ui,
                &mut settings.erosion.rounding.x,
                0.0..=1.0,
                "ridge rounding",
            );
            changed |= slider(
                ui,
                &mut settings.erosion.rounding.y,
                0.0..=1.0,
                "crease rounding",
            );
            changed |= ui
                .add(egui::Slider::new(&mut settings.erosion.octaves, 1..=8).text("octaves"))
                .changed();

            ui.separator();
            ui.label("Terrain");
            changed |= slider(
                ui,
                &mut settings.height_offset,
                -1.1..=0.25,
                "height offset",
            );
            changed |= slider(ui, &mut settings.vertical_scale, 10.0..=40.0, "relief");
            changed |= slider(ui, &mut settings.map_scale, 0.8..=3.6, "range scale");
            changed |= slider(ui, &mut settings.water_level, -3.0..=1.0, "water level");
        });

    if changed {
        settings.dirty = true;
    }
    Ok(())
}

fn slider(
    ui: &mut egui::Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    text: &str,
) -> bool {
    ui.add(egui::Slider::new(value, range).text(text)).changed()
}

fn keyboard_shortcuts(
    keyboard: Res<ButtonInput<KeyCode>>,
    egui_wants: Res<EguiWantsInput>,
    mut settings: ResMut<DemoSettings>,
) {
    if egui_wants.wants_any_keyboard_input() {
        return;
    }

    if keyboard.just_pressed(KeyCode::Space) {
        settings.erosion_enabled = !settings.erosion_enabled;
        settings.dirty = true;
    }

    if keyboard.just_pressed(KeyCode::KeyM) {
        settings.view_mode = settings.view_mode.next();
        settings.dirty = true;
    }

    if keyboard.just_pressed(KeyCode::KeyR) {
        *settings = DemoSettings::default();
    }
}

fn orbit_camera(
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    egui_wants: Res<EguiWantsInput>,
    mut orbit: ResMut<OrbitCamera>,
    mut camera: Single<&mut Transform, With<Camera3d>>,
) {
    if !egui_wants.wants_any_pointer_input() {
        if mouse_scroll.delta.y != 0.0 {
            let zoom = 1.0 - mouse_scroll.delta.y * 0.08;
            orbit.radius = (orbit.radius * zoom).clamp(5.5, 34.0);
        }

        if mouse_buttons.pressed(MouseButton::Left) || mouse_buttons.pressed(MouseButton::Right) {
            let delta = mouse_motion.delta;
            orbit.yaw -= delta.x * 0.006;
            orbit.pitch = (orbit.pitch + delta.y * 0.004).clamp(-0.08, FRAC_PI_2 - 0.08);
        }
    }

    **camera = camera_transform(&orbit);
}

fn camera_transform(orbit: &OrbitCamera) -> Transform {
    let horizontal = orbit.pitch.cos();
    let offset = Vec3::new(
        orbit.yaw.sin() * horizontal,
        orbit.pitch.sin(),
        orbit.yaw.cos() * horizontal,
    ) * orbit.radius;
    Transform::from_translation(orbit.target + offset).looking_at(orbit.target, Vec3::Y)
}

fn update_terrain_material(
    mut settings: ResMut<DemoSettings>,
    terrain: Single<&MeshMaterial3d<TerrainMaterial>, With<Terrain>>,
    mut water: Single<&mut Transform, (With<Water>, Without<Terrain>)>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
) {
    if !settings.dirty {
        return;
    }

    if let Some(material) = materials.get_mut(&terrain.0) {
        material.settings = settings.material_uniform();
    }
    water.translation.y = settings.water_level;

    settings.dirty = false;
}

fn build_terrain_mesh() -> Mesh {
    let side = TERRAIN_RESOLUTION + 1;
    let mut positions = Vec::<[f32; 3]>::new();
    let mut normals = Vec::<[f32; 3]>::new();
    let mut uvs = Vec::<[f32; 2]>::new();
    let mut indices = Vec::<u32>::new();

    for z in 0..side {
        for x in 0..side {
            let uv = Vec2::new(
                x as f32 / TERRAIN_RESOLUTION as f32,
                z as f32 / TERRAIN_RESOLUTION as f32,
            );
            positions.push(top_position(uv, 1.0).to_array());
            normals.push(Vec3::Y.to_array());
            uvs.push(uv.to_array());
        }
    }

    for z in 0..TERRAIN_RESOLUTION {
        for x in 0..TERRAIN_RESOLUTION {
            let i00 = top_index(x, z, side);
            let i10 = top_index(x + 1, z, side);
            let i01 = top_index(x, z + 1, side);
            let i11 = top_index(x + 1, z + 1, side);
            if (x + z) % 2 == 0 {
                indices.extend_from_slice(&[i00, i01, i10, i10, i01, i11]);
            } else {
                indices.extend_from_slice(&[i00, i01, i11, i00, i11, i10]);
            }
        }
    }

    push_side(
        &mut positions,
        &mut normals,
        &mut uvs,
        &mut indices,
        (0..side).map(|x| Vec2::new(x as f32 / TERRAIN_RESOLUTION as f32, 0.0)),
        Vec3::NEG_Z,
    );
    push_side(
        &mut positions,
        &mut normals,
        &mut uvs,
        &mut indices,
        (0..side).map(|x| Vec2::new(x as f32 / TERRAIN_RESOLUTION as f32, 1.0)),
        Vec3::Z,
    );
    push_side(
        &mut positions,
        &mut normals,
        &mut uvs,
        &mut indices,
        (0..side).map(|z| Vec2::new(0.0, z as f32 / TERRAIN_RESOLUTION as f32)),
        Vec3::NEG_X,
    );
    push_side(
        &mut positions,
        &mut normals,
        &mut uvs,
        &mut indices,
        (0..side).map(|z| Vec2::new(1.0, z as f32 / TERRAIN_RESOLUTION as f32)),
        Vec3::X,
    );

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices))
}

fn push_side(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
    edge: impl Iterator<Item = Vec2>,
    normal: Vec3,
) {
    let edge: Vec<_> = edge.collect();
    let Some([first_a, first_b]) = edge.windows(2).next().map(|w| [w[0], w[1]]) else {
        return;
    };

    // Decide winding once for the whole wall: the default index order produces
    // a face normal of cross(a_t1 - a_t0, b_t0 - a_t0). If that disagrees with
    // the supplied outward normal, we'd be back-face culled, so flip the
    // triangles. Without this, walls on the +Z and -X sides vanish.
    let a_t0 = top_position(first_a, 0.0);
    let a_t1 = top_position(first_a, 1.0);
    let b_t0 = top_position(first_b, 0.0);
    let face_normal = (a_t1 - a_t0).cross(b_t0 - a_t0);
    let flip_winding = face_normal.dot(normal) < 0.0;

    for pair in edge.windows(2) {
        let [a, b] = pair else {
            continue;
        };

        for layer in 0..SIDE_LAYERS {
            let t0 = layer as f32 / SIDE_LAYERS as f32;
            let t1 = (layer + 1) as f32 / SIDE_LAYERS as f32;

            let base = positions.len() as u32;
            positions.extend_from_slice(&[
                top_position(*a, t0).to_array(),
                top_position(*b, t0).to_array(),
                top_position(*a, t1).to_array(),
                top_position(*b, t1).to_array(),
            ]);
            normals.extend_from_slice(&[normal.to_array(); 4]);
            uvs.extend_from_slice(&[a.to_array(), b.to_array(), a.to_array(), b.to_array()]);
            if flip_winding {
                indices.extend_from_slice(&[
                    base,
                    base + 1,
                    base + 2,
                    base + 1,
                    base + 3,
                    base + 2,
                ]);
            } else {
                indices.extend_from_slice(&[
                    base,
                    base + 2,
                    base + 1,
                    base + 1,
                    base + 2,
                    base + 3,
                ]);
            }
        }
    }
}

fn top_position(uv: Vec2, top_factor: f32) -> Vec3 {
    Vec3::new(
        (uv.x - 0.5) * TERRAIN_EXTENT,
        top_factor,
        (uv.y - 0.5) * TERRAIN_EXTENT,
    )
}

fn top_index(x: usize, z: usize, side: usize) -> u32 {
    (z * side + x) as u32
}
