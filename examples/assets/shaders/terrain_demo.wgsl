#import bevy_pbr::{
    forward_io::Vertex,
    mesh_functions,
    mesh_view_bindings::view,
    view_transformations::position_world_to_clip,
}

#import bevy_erosion_filter::erosion::{
    ErosionFilterParams,
    fbm,
    erosion_filter,
    noised,
}

const PI: f32 = 3.14159265358979;

// -----------------------------------------------------------------------------
// Reference colour palette and thresholds (from the Shadertoy demo).
// -----------------------------------------------------------------------------

const CLIFF_COLOR: vec3<f32> = vec3<f32>(0.22, 0.20, 0.20);
const DIRT_COLOR: vec3<f32> = vec3<f32>(0.60, 0.50, 0.40);
const TREE_COLOR: vec3<f32> = vec3<f32>(0.12, 0.26, 0.10);
const GRASS_COLOR1: vec3<f32> = vec3<f32>(0.15, 0.30, 0.10);
const GRASS_COLOR2: vec3<f32> = vec3<f32>(0.40, 0.50, 0.20);
const SAND_COLOR: vec3<f32> = vec3<f32>(0.80, 0.70, 0.60);

// SUN_COLOR = vec3(1.0, 0.98, 0.95) * 2.0
const SUN_COLOR: vec3<f32> = vec3<f32>(2.00, 1.96, 1.90);
// AMBIENT_COLOR = vec3(0.3, 0.5, 0.7) * 0.1
const AMBIENT_COLOR: vec3<f32> = vec3<f32>(0.030, 0.050, 0.070);

const GRASS_HEIGHT: f32 = 0.465;
const DRAINAGE_WIDTH: f32 = 0.3;
const GEOMETRY_EROSION_OCTAVES: i32 = 3;

struct TerrainDemoUniform {
    erosion0: vec4<f32>,
    rounding: vec4<f32>,
    onset: vec4<f32>,
    erosion1: vec4<f32>,
    erosion2: vec4<f32>,
    terrain0: vec4<f32>,
    terrain1: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> demo: TerrainDemoUniform;

struct TerrainEval {
    unit_height: f32,
    world_height: f32,
    slope: vec2<f32>,
    erosion_delta: f32,
    ridge_map: f32,
    debug: f32,
    tree_amount: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) top_factor: f32,
    @location(3) side_normal: vec3<f32>,
}

fn erosion_params_for_octaves(octaves: i32) -> ErosionFilterParams {
    return ErosionFilterParams(
        demo.erosion0.x,
        demo.erosion0.y,
        demo.erosion0.z,
        demo.erosion0.w,
        demo.rounding,
        demo.onset,
        demo.erosion1.xy,
        demo.erosion1.z,
        demo.erosion1.w,
        octaves,
        demo.erosion2.y,
        demo.erosion2.z,
    );
}

fn erosion_params() -> ErosionFilterParams {
    return erosion_params_for_octaves(i32(round(demo.erosion2.x)));
}

fn terrain_point(uv: vec2<f32>) -> vec2<f32> {
    return uv * demo.terrain0.z + vec2<f32>(0.17, 0.31);
}

fn saturate(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn smoothstep_down(edge0: f32, edge1: f32, x: f32) -> f32 {
    return 1.0 - smoothstep(edge0, edge1, x);
}

// -----------------------------------------------------------------------------
// BRDF helpers (Burley diffuse + GGX specular, port of the reference's `Shade`).
// -----------------------------------------------------------------------------

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

fn d_ggx(linear_roughness: f32, n_o_h: f32) -> f32 {
    let one_minus = 1.0 - n_o_h * n_o_h;
    let a = n_o_h * linear_roughness;
    let k = linear_roughness / (one_minus + a * a);
    return k * k * (1.0 / PI);
}

fn v_smith_ggx_correlated(linear_roughness: f32, n_o_v: f32, n_o_l: f32) -> f32 {
    let a2 = linear_roughness * linear_roughness;
    let ggxv = n_o_l * sqrt((n_o_v - a2 * n_o_v) * n_o_v + a2);
    let ggxl = n_o_v * sqrt((n_o_l - a2 * n_o_l) * n_o_l + a2);
    return 0.5 / (ggxv + ggxl);
}

fn f_schlick3(f0: vec3<f32>, v_o_h: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow5(1.0 - v_o_h);
}

fn f_schlick_scalar(f0: f32, f90: f32, v_o_h: f32) -> f32 {
    return f0 + (f90 - f0) * pow5(1.0 - v_o_h);
}

fn fd_burley(linear_roughness: f32, n_o_v: f32, n_o_l: f32, l_o_h: f32) -> f32 {
    let f90 = 0.5 + 2.0 * linear_roughness * l_o_h * l_o_h;
    let light_scatter = f_schlick_scalar(1.0, f90, n_o_l);
    let view_scatter = f_schlick_scalar(1.0, f90, n_o_v);
    return light_scatter * view_scatter * (1.0 / PI);
}

fn fd_lambert() -> f32 {
    return 1.0 / PI;
}

fn shade(
    diffuse: vec3<f32>,
    f0: vec3<f32>,
    smoothness: f32,
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    lc: vec3<f32>,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_o_v = abs(dot(n, v)) + 1e-5;
    let n_o_l = saturate(dot(n, l));
    let n_o_h = saturate(dot(n, h));
    let l_o_h = saturate(dot(l, h));
    let roughness = 1.0 - smoothness;
    let linear_roughness = roughness * roughness;
    let d = d_ggx(linear_roughness, n_o_h);
    let v_term = v_smith_ggx_correlated(linear_roughness, n_o_v, n_o_l);
    let f = f_schlick3(f0, l_o_h);
    let fr = (d * v_term) * f;
    let fd = diffuse * fd_burley(linear_roughness, n_o_v, n_o_l, l_o_h);
    return (fd + fr) * lc * n_o_l;
}

fn sky_color(rd: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let costh = dot(rd, sun);
    return AMBIENT_COLOR * PI * (1.0 - abs(costh) * 0.8);
}

// Ambient + sun + bounce, matching the reference's lighting accumulator.
fn lit(
    albedo: vec3<f32>,
    f0: vec3<f32>,
    smoothness: f32,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    sun: vec3<f32>,
    occlusion: f32,
) -> vec3<f32> {
    var color = albedo * sky_color(normal, sun) * fd_lambert() * occlusion;
    color += shade(albedo, f0, smoothness, normal, view_dir, sun, SUN_COLOR);
    let bounce_dir = sun * vec3<f32>(1.0, -1.0, 1.0);
    color += albedo * SUN_COLOR * (dot(normal, bounce_dir) * 0.5 + 0.5) * fd_lambert() / PI;
    return color;
}

// -----------------------------------------------------------------------------
// Tree mask. Direct port of `GetTreesAmount` from the reference.
// -----------------------------------------------------------------------------

fn get_trees_amount(
    height: f32,
    normal_y: f32,
    occlusion: f32,
    ridge_map: f32,
    water_unit: f32,
) -> f32 {
    let t = smoothstep_down(GRASS_HEIGHT + 0.01, GRASS_HEIGHT + 0.05, height + 0.01 + (occlusion - 0.8) * 0.05)
        * smoothstep(0.0, 0.4, occlusion)
        * smoothstep(0.95, 1.0, normal_y)
        * smoothstep(-1.4, 0.0, ridge_map)
        * smoothstep(water_unit + 0.000, water_unit + 0.007, height);
    return (t - 0.5) / 0.6;
}

// -----------------------------------------------------------------------------
// Terrain evaluation. Computes height, slope, ridge map, and the tree value
// (pre-baked into the heightfield as a small silhouette bump).
// -----------------------------------------------------------------------------

fn slope_to_world_factor() -> f32 {
    return demo.terrain0.y * demo.terrain0.z / demo.terrain0.x * demo.terrain1.z;
}

fn evaluate_terrain_with_octaves(uv: vec2<f32>, max_erosion_octaves: i32, add_tree_bump: bool) -> TerrainEval {
    let p = terrain_point(uv);

    let height_amp = 0.125;
    let raw = fbm(p, 3.0, 3, 2.0, 0.1) * height_amp;
    let fade_target = clamp(raw.x / (height_amp * 0.6), -1.0, 1.0);
    let base = raw * 0.5 + vec3<f32>(0.5, 0.0, 0.0);

    var delta = vec3<f32>(0.0);
    var magnitude = 0.0;
    var ridge_map = 1.0;
    var debug = fade_target;

    if demo.terrain1.x > 0.5 {
        let octaves = min(i32(round(demo.erosion2.x)), max_erosion_octaves);
        let filtered = erosion_filter(p, base, fade_target, erosion_params_for_octaves(octaves));
        delta = filtered.delta;
        magnitude = filtered.magnitude;
        ridge_map = filtered.ridge_map;
        debug = filtered.debug;
    }

    let height_offset = demo.erosion2.w * magnitude;
    var unit_height = base.x + delta.x + height_offset;
    let slope = base.yz + delta.yz;

    var erosion_delta = 0.0;
    if magnitude > 0.0 {
        erosion_delta = delta.x / magnitude;
    }
    let occlusion = saturate(erosion_delta + 0.5);

    // World normal used for the tree mask (and later for shading) — the
    // thresholds fight aspect-ratio either way, so just use the visible normal.
    let stw = slope_to_world_factor();
    let world_normal = normalize(vec3<f32>(-slope.x * stw, 1.0, -slope.y * stw));

    let water_unit = demo.terrain1.w / demo.terrain0.y + 0.43;

    let trees_amount = get_trees_amount(unit_height, world_normal.y, occlusion, ridge_map, water_unit);
    let high_freq_noise = noised((p + vec2<f32>(0.5)) * 200.0).x * 0.5 + 0.5;
    let tree_value = (trees_amount - high_freq_noise * high_freq_noise) * 1.5;

    if add_tree_bump && tree_value > 0.0 {
        unit_height += tree_value / 300.0;
    }

    var eval: TerrainEval;
    eval.unit_height = unit_height;
    eval.world_height = (unit_height - 0.43) * demo.terrain0.y;
    eval.slope = slope;
    eval.erosion_delta = erosion_delta;
    eval.ridge_map = ridge_map;
    eval.debug = debug;
    eval.tree_amount = tree_value;
    return eval;
}

fn evaluate_terrain(uv: vec2<f32>) -> TerrainEval {
    return evaluate_terrain_with_octaves(uv, i32(round(demo.erosion2.x)), true);
}

fn evaluate_terrain_geometry(uv: vec2<f32>) -> TerrainEval {
    return evaluate_terrain_with_octaves(uv, GEOMETRY_EROSION_OCTAVES, false);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let eval = evaluate_terrain_geometry(vertex.uv);
    let top_factor = clamp(vertex.position.y, 0.0, 1.0);
    let y = mix(demo.terrain0.w, eval.world_height, top_factor);
    let local_position = vec3<f32>(vertex.position.x, y, vertex.position.z);
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    let world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(local_position, 1.0),
    ).xyz;

    var out: VertexOutput;
    out.position = position_world_to_clip(world_position);
    out.world_position = world_position;
    out.uv = vertex.uv;
    out.top_factor = top_factor;
    out.side_normal = vertex.normal;
    return out;
}

fn smootherstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn terrain_normal(eval: TerrainEval) -> vec3<f32> {
    let stw = slope_to_world_factor();
    return normalize(vec3<f32>(
        -eval.slope.x * stw,
        1.0,
        -eval.slope.y * stw,
    ));
}

fn debug_color(eval: TerrainEval) -> vec3<f32> {
    let mode = i32(round(demo.terrain1.y));
    if mode == 1 {
        let t = saturate(eval.erosion_delta * 0.5 + 0.5);
        return mix(vec3<f32>(0.05, 0.10, 0.16), vec3<f32>(0.82, 0.76, 0.66), t);
    }
    if mode == 2 {
        let crease = smootherstep(0.0, 1.0, -eval.ridge_map);
        let ridge = smootherstep(0.0, 1.0, eval.ridge_map);
        return mix(
            mix(vec3<f32>(0.30, 0.34, 0.35), vec3<f32>(0.03, 0.18, 0.26), crease),
            vec3<f32>(0.95, 0.94, 0.88),
            ridge,
        );
    }
    if mode == 3 {
        let t = saturate(eval.debug * 0.5 + 0.5);
        return mix(vec3<f32>(0.08, 0.16, 0.34), vec3<f32>(0.95, 0.83, 0.58), t);
    }
    return vec3<f32>(-1.0);
}

// -----------------------------------------------------------------------------
// Albedo cascade. Direct port of the reference's M_GROUND branch with the
// detail-texture `breakup` term zeroed out (deferred).
// -----------------------------------------------------------------------------

fn terrain_albedo(eval: TerrainEval, normal: vec3<f32>) -> vec3<f32> {
    let h = eval.unit_height;
    let occlusion = saturate(eval.erosion_delta + 0.5);
    // Reference renderer reads `trees` and `ridgemap` after the buffer pack-
    // unpack roundtrip `clamp01(x * 0.5 + 0.5)`, so the cascade sees them in
    // [0, 1]. Match that here for the visual cascade.
    let trees = saturate(eval.tree_amount * 0.5 + 0.5);
    let ridgemap = saturate(eval.ridge_map * 0.5 + 0.5);
    let water_unit = demo.terrain1.w / demo.terrain0.y + 0.43;

    // Cliff base, then mix dirt into low-occlusion areas.
    var color = CLIFF_COLOR * smoothstep(0.4, 0.52, h);
    color = mix(color, DIRT_COLOR, smoothstep_down(0.0, 0.6, occlusion));

    // Snow on the high tops.
    color = mix(color, vec3<f32>(1.0), smoothstep(0.53, 0.6, h));

    // Sand at the waterline.
    color = mix(color, SAND_COLOR, smoothstep_down(water_unit, water_unit + 0.005, h));

    // Grass: two-tone height blend gated by both height and slope/tree masks.
    let grass_mix = mix(GRASS_COLOR1, GRASS_COLOR2, smoothstep(0.4, 0.6, h - eval.erosion_delta * 0.05));
    let grass_height_mask = smoothstep_down(GRASS_HEIGHT + 0.02, GRASS_HEIGHT + 0.05, h + 0.01 + (occlusion - 0.8) * 0.05);
    let grass_normal_mask = smoothstep(0.8, 1.0, 1.0 - (1.0 - normal.y) * (1.0 - trees));
    color = mix(color, grass_mix, grass_height_mask * grass_normal_mask);

    // Trees darken the canopy where the tree value is well above zero.
    color = mix(color, TREE_COLOR * pow(trees, 8.0), saturate(trees * 2.2 - 0.8) * 0.6);

    // Drainage paint along creases (low ridgemap values, post-remap).
    let drainage = saturate((1.0 - saturate(ridgemap / DRAINAGE_WIDTH)) * 1.5);
    color = mix(color, vec3<f32>(1.0), drainage);

    return color;
}

fn shade_top(in: VertexOutput, eval: TerrainEval) -> vec3<f32> {
    let dbg = debug_color(eval);
    if dbg.x >= 0.0 {
        return dbg;
    }

    let normal = terrain_normal(eval);
    let albedo = terrain_albedo(eval, normal);
    let sun = normalize(vec3<f32>(-1.0, 0.4, 0.05));
    let view_dir = normalize(view.world_position.xyz - in.world_position);
    let occlusion = saturate(eval.erosion_delta + 0.5);

    return lit(albedo, vec3<f32>(0.04), 0.0, normal, view_dir, sun, occlusion);
}

// -----------------------------------------------------------------------------
// Side walls — single albedo with a half-Lambert wraparound. The standard
// `lit()` BRDF would clamp `dot(N, L)` to zero on the three sides that don't
// face the sun, leaving them nearly black; wraparound + a floor keeps every
// face visibly lit while preserving a sun-direction gradient.
// -----------------------------------------------------------------------------

const SIDE_ALBEDO: vec3<f32> = vec3<f32>(0.45, 0.40, 0.35);

fn side_color(in: VertexOutput, eval: TerrainEval) -> vec3<f32> {
    let normal = normalize(in.side_normal);
    let sun = normalize(vec3<f32>(-1.0, 0.4, 0.05));
    let half_lambert = dot(normal, sun) * 0.5 + 0.5;
    let wrap = half_lambert * 0.7 + 0.3;
    return SIDE_ALBEDO * SUN_COLOR * wrap * fd_lambert();
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let eval = evaluate_terrain(in.uv);
    let top = smootherstep(0.985, 1.0, in.top_factor);
    let top_color = shade_top(in, eval);
    let side = side_color(in, eval);
    return vec4<f32>(mix(side, top_color, top), 1.0);
}
