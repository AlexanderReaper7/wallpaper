// Wireframe cubes drawn as one screen-space quad per edge. The fragment stage
// only runs on pixels within GLOW_RADIUS of a line, so the cost scales with
// line length, not with the 8.3 million pixels of a 4K output.

struct Cube {
    // (cos a, sin a, cos b, sin b): yaw a, then pitch b. The CPU wraps the
    // angles in f64, so the shader never sees a large time value.
    rot: vec4<f32>,
    // (offset x, offset y, scale, brightness)
    place: vec4<f32>,
}

struct Globals {
    resolution: vec2<f32>,
    _pad: vec2<f32>,
    cubes: array<Cube, 3>,
}

@group(0) @binding(0) var<uniform> g: Globals;

const EDGES_PER_CUBE: u32 = 12u;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) a: vec2<f32>,
    @location(1) @interpolate(flat) b: vec2<f32>,
    // (depth at a, depth at b, brightness, unused)
    @location(2) @interpolate(flat) info: vec4<f32>,
}

fn corner(c: u32) -> vec3<f32> {
    return vec3<f32>(f32(c & 1u), f32((c >> 1u) & 1u), f32((c >> 2u) & 1u)) * 2.0 - 1.0;
}

fn rotate(v: vec3<f32>, r: vec4<f32>) -> vec3<f32> {
    // Pitch about x, then yaw about y.
    let y = r.z * v.y - r.w * v.z;
    let z1 = r.w * v.y + r.z * v.z;
    let x = r.x * v.x + r.y * z1;
    let z = -r.y * v.x + r.x * z1;
    return vec3<f32>(x, y, z);
}

// Screen units: y spans -1..1 bottom to top, x spans -aspect..aspect.
fn to_pixels(p: vec2<f32>) -> vec2<f32> {
    let res = g.resolution;
    return vec2<f32>((p.x * res.y + res.x) * 0.5, (res.y - p.y * res.y) * 0.5);
}

fn glow_radius() -> f32 {
    return 0.03 * g.resolution.y;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let cube = g.cubes[ii / EDGES_PER_CUBE];
    let edge = ii % EDGES_PER_CUBE;

    // Edge k along axis n joins corner c to c with bit n set, where c is k with
    // a zero bit inserted at position n.
    let axis = edge / 4u;
    let k = edge % 4u;
    let low = k & ((1u << axis) - 1u);
    let ca = low | ((k >> axis) << (axis + 1u));
    let cb = ca | (1u << axis);

    let wa = rotate(corner(ca), cube.rot) * cube.place.z;
    let wb = rotate(corner(cb), cube.rot) * cube.place.z;
    let a = to_pixels(wa.xy / (4.0 - wa.z) * 2.0 + cube.place.xy);
    let b = to_pixels(wb.xy / (4.0 - wb.z) * 2.0 + cube.place.xy);

    var dir = vec2<f32>(1.0, 0.0);
    if (distance(a, b) > 1e-3) {
        dir = normalize(b - a);
    }
    let normal = vec2<f32>(-dir.y, dir.x);

    // Triangle strip: (start, -side), (start, +side), (end, -side), (end, +side).
    let along = f32(vi >> 1u);
    let side = f32(vi & 1u) * 2.0 - 1.0;
    let r = glow_radius();
    let p = mix(a, b, along) + dir * (along * 2.0 - 1.0) * r + normal * side * r;

    var out: VsOut;
    out.pos = vec4<f32>(p.x / g.resolution.x * 2.0 - 1.0, 1.0 - p.y / g.resolution.y * 2.0, 0.0, 1.0);
    out.a = a;
    out.b = b;
    out.info = vec4<f32>(wa.z, wb.z, cube.place.w, 0.0);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let pa = in.pos.xy - in.a;
    let ba = in.b - in.a;
    let t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    let d = length(pa - ba * t);

    let z = mix(in.info.x, in.info.y, t);
    let shade = 0.35 + 0.65 * smoothstep(-1.5, 1.5, z);
    let core = 1.0 - smoothstep(0.0, 1.5, d);
    let r = glow_radius();
    let knee = 0.002 * g.resolution.y;
    let glow = 0.15 * knee / (d + knee) * (1.0 - smoothstep(0.5 * r, r, d));
    let c = (core + glow) * shade * in.info.z;
    return vec4<f32>(c, c, c, 1.0);
}
