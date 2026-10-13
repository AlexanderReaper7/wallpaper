// A network of points joined by lines, drawn as one screen-space quad per line
// and per point. The fragment stage only runs on pixels within the glow radius
// of a line, so the cost scales with line length, not with the 8.3 million
// pixels of a 4K output.
//
// Brightness at a pixel is a falling function of its distance to the nearest
// line. The pipeline blends with max, and the max over lines of a falling
// function of distance is that function of the smallest distance, so where
// lines meet the result is as bright as the brightest of them, never their sum.

struct Globals {
    resolution: vec2<f32>,
    glow_radius: f32,
    glow_strength: f32,
    // Shimmer phase in turns, wrapped to 0..1 on the CPU in f64.
    shimmer_clock: f32,
    shimmer_wavelength: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> g: Globals;

struct Segment {
    // Endpoints in physical pixels, y down. A point is a segment with a == b.
    @location(0) a: vec2<f32>,
    @location(1) b: vec2<f32>,
    // (core half-width px, brightness, shimmer phase in turns, shimmer depth)
    @location(2) style: vec4<f32>,
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) a: vec2<f32>,
    @location(1) @interpolate(flat) b: vec2<f32>,
    @location(2) @interpolate(flat) style: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, seg: Segment) -> VsOut {
    var dir = vec2<f32>(1.0, 0.0);
    if (distance(seg.a, seg.b) > 1e-3) {
        dir = normalize(seg.b - seg.a);
    }
    let normal = vec2<f32>(-dir.y, dir.x);

    // Triangle strip: (start, -side), (start, +side), (end, -side), (end, +side).
    let along = f32(vi >> 1u);
    let side = f32(vi & 1u) * 2.0 - 1.0;
    let r = g.glow_radius + seg.style.x + 1.0;
    let p = mix(seg.a, seg.b, along) + dir * (along * 2.0 - 1.0) * r + normal * side * r;

    var out: VsOut;
    out.pos = vec4<f32>(p.x / g.resolution.x * 2.0 - 1.0, 1.0 - p.y / g.resolution.y * 2.0, 0.0, 1.0);
    out.a = seg.a;
    out.b = seg.b;
    out.style = seg.style;
    return out;
}

const TAU: f32 = 6.283185307;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let pa = in.pos.xy - in.a;
    let ba = in.b - in.a;
    let t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    // Distance from the edge of the core, so the glow starts where the core ends.
    let d = max(length(pa - ba * t) - in.style.x, 0.0);

    let core = 1.0 - smoothstep(0.0, 1.0, d);
    let x = clamp(d / g.glow_radius, 0.0, 1.0);
    let glow = g.glow_strength * pow(1.0 - x, 4.0);

    // A soft brightening that travels slowly along the line from a to b.
    let s = length(ba) * t;
    let wave = 0.5 + 0.5 * sin(TAU * (s / g.shimmer_wavelength - g.shimmer_clock + in.style.z));
    let shimmer = 1.0 - in.style.w * wave;

    let c = min(core + glow, 1.0) * in.style.y * shimmer;
    return vec4<f32>(c, c, c, 1.0);
}
