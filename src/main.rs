//! Animated wallpaper: one background layer surface per output, each drawn
//! once per frame callback from that output, so every screen runs at its own
//! refresh rate and a covered screen stops being asked for frames.
//!
//! The picture is a network of points drifting across a black background,
//! with a white shimmering line between every two points that are close enough.

use std::{
    f64::consts::TAU,
    hash::{DefaultHasher, Hash, Hasher},
    path::{Path, PathBuf},
    ptr::NonNull,
    time::Instant,
};

use raw_window_handle::{RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, FrameCallbackData},
    delegate_dispatch2, delegate_registry,
    dispatch2::Dispatch2,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
        WaylandSurface,
    },
};
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_output::WlOutput, wl_surface::WlSurface},
    Connection, Proxy, QueueHandle,
};
use wayland_protocols::wp::viewporter::client::{wp_viewport::WpViewport, wp_viewporter::WpViewporter};

// Tuning. Distances and speeds scale with the screen height, so the picture
// looks the same on any resolution; widths are in physical pixels.

/// Points per screen. They spread over the screen plus a margin just outside
/// it, where they wrap around unseen.
const POINTS: usize = 90;
/// Two points closer than this are joined, as a fraction of the screen height.
/// Lines fade out as they approach this length.
const CONNECT_DISTANCE: f64 = 0.2;
/// How fast the fastest point wants to drift, as a fraction of the screen
/// height per second. 0.004 is about 9 px/s on a 2160-line screen.
const SPEED: f64 = 0.004;
/// How fast the slowest point wants to drift, as a fraction of SPEED. Each
/// point gets its own speed, spread evenly between this and SPEED.
const SLOWEST: f64 = 0.15;
/// How strongly joined points pull each other's velocity together, so a group
/// drifts as one and groups move at different speeds. 0 lets every point keep
/// its own velocity. At 2, one full-strength line weighs twice a point's own
/// wish. Weaker, longer lines pull less.
const GROUP_PULL: f64 = 2.0;
/// Seconds a point takes to settle most of the way (63%) into its group's motion.
const GROUP_SETTLE: f64 = 4.0;
/// Two points closer than this push apart, so the points stay evenly spread
/// instead of clumping. A fraction of the even spacing, the side of the square
/// each point would get if the field were shared out equally. The push is at
/// most SPEED, when two points sit on top of each other.
const SPACING: f64 = 0.7;
/// Width of a line's solid core, in pixels.
const LINE_WIDTH: f32 = 1.5;
/// Diameter of a point's solid core, in pixels. It shows where lines meet and
/// is no brighter than a line.
const POINT_WIDTH: f32 = 4.0;
/// Brightness of the glow right at the edge of the core, 0..1.
const GLOW_STRENGTH: f32 = 0.12;
/// How far the glow reaches past the core, in pixels. Each line is drawn as a
/// quad this much wider on every side, so this sets most of the GPU cost.
const GLOW_RADIUS: f32 = 40.0;
/// Peak brightness of lines and points, 0..1.
const BRIGHTNESS: f32 = 0.85;
/// How much the shimmer dims a line at its darkest, 0..1.
const SHIMMER_DEPTH: f32 = 0.5;
/// Shimmer waves per second travelling along each line.
const SHIMMER_SPEED: f64 = 0.08;
/// Distance between shimmer waves along a line, in pixels.
const SHIMMER_WAVELENGTH: f32 = 600.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    resolution: [f32; 2],
    glow_radius: f32,
    glow_strength: f32,
    shimmer_clock: f32,
    shimmer_wavelength: f32,
    _pad: [f32; 2],
}

/// One line, or one point when `a == b`. Matches `Segment` in lines.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Segment {
    a: [f32; 2],
    b: [f32; 2],
    /// (core half-width px, brightness, shimmer phase in turns, shimmer depth)
    style: [f32; 4],
}

const MAX_SEGMENTS: usize = POINTS + POINTS * (POINTS - 1) / 2;

/// A point drifting across a field that wraps at its edges. The field is the
/// screen grown by a margin wide enough that a point wraps only once nothing it
/// is joined to can be seen. Position and velocity are in pixels.
struct Point {
    position: [f64; 2],
    velocity: [f64; 2],
    /// The velocity the point drifts at when nothing is joined to it.
    preferred: [f64; 2],
    phase: f32,
}

/// `d` folded onto the shortest way around a wrapping axis of length `size`.
fn wrap_delta(d: f64, size: f64) -> f64 {
    d - size * (d / size).round()
}

/// splitmix64, enough to scatter the points without a dependency.
fn random(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// One screen's points, and the lines between them.
struct Network {
    points: Vec<Point>,
    /// Wrapping field size and its margin past each screen edge, in pixels.
    field: [f64; 2],
    margin: f64,
    height: f64,
    /// Rebuilt every frame from the points.
    segments: Vec<Segment>,
}

impl Network {
    /// Seeded by the output's name, so each screen has its own network and
    /// starts from the same picture on every run.
    fn new(name: &str, physical: (u32, u32)) -> Network {
        // A point wrapping at the field's edge is at least `margin` off screen,
        // so its lines and their glow are too.
        let height = physical.1 as f64;
        let margin = CONNECT_DISTANCE * height + (GLOW_RADIUS + POINT_WIDTH) as f64;
        let field = [physical.0 as f64 + 2.0 * margin, height + 2.0 * margin];
        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        let mut seed = hasher.finish();
        // Mitchell's best candidate: of a few random spots, each point takes the
        // one farthest from the points already placed. That gives an even
        // spread without a grid's regularity.
        let mut placed: Vec<[f64; 2]> = Vec::with_capacity(POINTS);
        for _ in 0..POINTS {
            let best = (0..16)
                .map(|_| [random(&mut seed) * field[0], random(&mut seed) * field[1]])
                .map(|c| {
                    let nearest = placed
                        .iter()
                        .map(|p| wrap_delta(c[0] - p[0], field[0]).hypot(wrap_delta(c[1] - p[1], field[1])))
                        .fold(f64::INFINITY, f64::min);
                    (c, nearest)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0;
            placed.push(best);
        }
        let points = placed
            .into_iter()
            .map(|position| {
                let angle = random(&mut seed) * TAU;
                let speed = SPEED * height * (SLOWEST + (1.0 - SLOWEST) * random(&mut seed));
                let preferred = [angle.cos() * speed, angle.sin() * speed];
                let phase = random(&mut seed) as f32;
                Point { position, velocity: preferred, preferred, phase }
            })
            .collect();

        Network { points, field, margin, height, segments: Vec::with_capacity(MAX_SEGMENTS) }
    }

    /// Builds this frame's segments from the current positions, then moves the
    /// points on by `dt` seconds.
    fn frame(&mut self, dt: f64) -> &[Segment] {
        let positions: Vec<[f32; 2]> = self
            .points
            .iter()
            .map(|p| [(p.position[0] - self.margin) as f32, (p.position[1] - self.margin) as f32])
            .collect();
        // Per point, the sum of joined points' velocities weighted by line
        // strength, and the sum of those weights.
        let mut pull = vec![([0.0f64; 2], 0.0f64); positions.len()];
        let reach = (CONNECT_DISTANCE * self.height) as f32;
        self.segments.clear();
        for (i, &a) in positions.iter().enumerate() {
            for (j, &b) in positions.iter().enumerate().skip(i + 1) {
                let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
                if length >= reach {
                    continue;
                }
                // Long lines are dim, so a line appears and disappears gradually.
                let f = 1.0 - length / reach;
                let fade = f * f * (3.0 - 2.0 * f);
                let phase = (self.points[i].phase + self.points[j].phase).fract();
                let (vi, vj) = (self.points[i].velocity, self.points[j].velocity);
                let w = fade as f64;
                pull[i].0 = [pull[i].0[0] + w * vj[0], pull[i].0[1] + w * vj[1]];
                pull[i].1 += w;
                pull[j].0 = [pull[j].0[0] + w * vi[0], pull[j].0[1] + w * vi[1]];
                pull[j].1 += w;
                self.segments.push(Segment {
                    a,
                    b,
                    style: [LINE_WIDTH / 2.0, BRIGHTNESS * fade, phase, SHIMMER_DEPTH],
                });
            }
        }
        for &p in &positions {
            self.segments.push(Segment { a: p, b: p, style: [POINT_WIDTH / 2.0, BRIGHTNESS, 0.0, 0.0] });
        }
        // Each point eases toward a blend of its own preferred velocity and its
        // neighbours' velocities, so joined points come to move together.
        // On top of that, points too close together push apart, measured the
        // short way around the wrapping field.
        let field = self.field;
        let spacing = SPACING * (field[0] * field[1] / POINTS as f64).sqrt();
        let push_speed = SPEED * self.height;
        let mut push = vec![[0.0f64; 2]; self.points.len()];
        for i in 0..self.points.len() {
            for j in i + 1..self.points.len() {
                let (a, b) = (self.points[i].position, self.points[j].position);
                let d = [wrap_delta(a[0] - b[0], field[0]), wrap_delta(a[1] - b[1], field[1])];
                let length = d[0].hypot(d[1]);
                if length >= spacing || length < 1e-6 {
                    continue;
                }
                let strength = push_speed * (1.0 - length / spacing) / length;
                for axis in 0..2 {
                    push[i][axis] += d[axis] * strength;
                    push[j][axis] -= d[axis] * strength;
                }
            }
        }
        let settle = 1.0 - (-dt / GROUP_SETTLE).exp();
        for ((p, (sum, weight)), push) in self.points.iter_mut().zip(pull).zip(push) {
            let norm = 1.0 + GROUP_PULL * weight;
            for axis in 0..2 {
                let target = (p.preferred[axis] + GROUP_PULL * sum[axis]) / norm + push[axis];
                p.velocity[axis] += (target - p.velocity[axis]) * settle;
                p.position[axis] = (p.position[axis] + p.velocity[axis] * dt).rem_euclid(self.field[axis]);
            }
        }

        &self.segments
    }
}

/// The shader's uniforms for a frame at `t` seconds. The shimmer clock is
/// wrapped in f64, so it keeps full precision however long the process runs.
fn globals(physical: (u32, u32), t: f64) -> Globals {
    Globals {
        resolution: [physical.0 as f32, physical.1 as f32],
        glow_radius: GLOW_RADIUS,
        glow_strength: GLOW_STRENGTH,
        shimmer_clock: (SHIMMER_SPEED * t).fract() as f32,
        shimmer_wavelength: SHIMMER_WAVELENGTH,
        _pad: [0.0; 2],
    }
}

struct Gpu {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    /// Built on the first configure, when the surface format is known.
    pipeline: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
}

/// The buffers one picture is drawn from.
struct Target {
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instances: wgpu::Buffer,
}

struct Screen {
    // Declared first so it drops before the wl_surface it renders into.
    surface: wgpu::Surface<'static>,
    viewport: WpViewport,
    layer: LayerSurface,
    output: WlOutput,
    name: String,
    /// Buffer size in physical pixels, taken from the output's current mode.
    physical: (u32, u32),
    target: Target,
    network: Network,
    last_step: Option<Instant>,
    configured: bool,
    /// Refresh rate in Hz, to count frame callbacks that came late.
    refresh: f64,
    frames: u32,
    stats: FrameStats,
}

/// Where a screen's frames spent their time since the last report.
#[derive(Default)]
struct FrameStats {
    /// Frame callbacks that came more than one and a half refresh periods
    /// after the previous one.
    late: u32,
    /// Whole draw call, and the part of it inside get_current_texture.
    draw: Timing,
    texture_wait: Timing,
}

#[derive(Default)]
struct Timing {
    total: f64,
    max: f64,
}

impl Timing {
    fn add(&mut self, seconds: f64) {
        self.total += seconds;
        self.max = self.max.max(seconds);
    }

    fn show(&self, frames: u32) -> String {
        format!("mean {:.2} max {:.2} ms", self.total / frames.max(1) as f64 * 1e3, self.max * 1e3)
    }
}

struct App {
    registry: RegistryState,
    outputs: OutputState,
    compositor: CompositorState,
    layer_shell: LayerShell,
    viewporter: WpViewporter,
    conn: Connection,
    gpu: Gpu,
    screens: Vec<Screen>,
    start: Instant,
    last_report: Instant,
    /// Set by `--still <dir>`: write each screen's first frame there as
    /// `<output>.png` and exit, instead of drawing on the screens.
    still: Option<PathBuf>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let still = match (args.next().as_deref(), args.next(), args.next()) {
        (None, ..) => None,
        (Some("--still"), Some(dir), None) => Some(PathBuf::from(dir)),
        _ => {
            eprintln!("usage: wallpaper [--still <dir>]");
            std::process::exit(2);
        }
    };

    let conn = Connection::connect_to_env().expect("no Wayland display");
    let (globals, mut queue) = registry_queue_init(&conn).expect("registry");
    let qh = queue.handle();

    let compositor = CompositorState::bind(&globals, &qh).expect("wl_compositor");
    let layer_shell = LayerShell::bind(&globals, &qh).expect("zwlr_layer_shell_v1");
    let viewporter: WpViewporter = globals.bind(&qh, 1..=1, Ignore).expect("wp_viewporter");

    let mut app = App {
        registry: RegistryState::new(&globals),
        outputs: OutputState::new(&globals, &qh),
        compositor,
        layer_shell,
        viewporter,
        conn: conn.clone(),
        gpu: pollster::block_on(Gpu::new()),
        screens: Vec::new(),
        start: Instant::now(),
        last_report: Instant::now(),
        still,
    };

    if app.still.is_some() {
        // The first roundtrip announces the outputs, the second their modes
        // and names, which is when new_output writes the stills.
        queue.roundtrip(&mut app).expect("roundtrip");
        queue.roundtrip(&mut app).expect("roundtrip");
        return;
    }
    loop {
        queue.blocking_dispatch(&mut app).expect("dispatch");
    }
}

impl Gpu {
    async fn new() -> Gpu {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("no Vulkan adapter");
        eprintln!("adapter: {:?}", adapter.get_info().name);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("device");
        let shader = device.create_shader_module(wgpu::include_wgsl!("lines.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        Gpu { instance, adapter, device, queue, shader, layout, pipeline: None }
    }

    fn pipeline(&mut self, format: wgpu::TextureFormat) -> &wgpu::RenderPipeline {
        if self.pipeline.as_ref().is_some_and(|(f, _)| *f == format) {
            return &self.pipeline.as_ref().unwrap().1;
        }
        assert!(self.pipeline.is_none(), "outputs with different surface formats are not handled");
        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&self.layout)],
            ..Default::default()
        });
        // Max, not add: see the top of lines.wgsl.
        let brightest = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        };
        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Segment>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState { color: brightest, alpha: brightest }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        self.pipeline = Some((format, pipeline));
        &self.pipeline.as_ref().unwrap().1
    }

    fn target(&self) -> Target {
        let uniforms = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });
        let instances = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (MAX_SEGMENTS * size_of::<Segment>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Target { uniforms, bind_group, instances }
    }

    /// Records drawing `segments` onto a black `view` into `encoder`.
    fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &Target,
        view: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        globals: &Globals,
        segments: &[Segment],
    ) {
        self.queue.write_buffer(&target.uniforms, 0, bytemuck::bytes_of(globals));
        self.queue.write_buffer(&target.instances, 0, bytemuck::cast_slice(segments));
        let pipeline = self.pipeline(format);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &target.bind_group, &[]);
        pass.set_vertex_buffer(0, target.instances.slice(..));
        pass.draw(0..4, 0..segments.len() as u32);
    }

    /// Renders the first frame `name`'s screen shows and writes it as a PNG,
    /// in the sRGB format the live surfaces use.
    fn write_still(&mut self, name: &str, physical: (u32, u32), path: &Path) {
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let size = wgpu::Extent3d { width: physical.0, height: physical.1, depth_or_array_layers: 1 };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let row = (physical.0 * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: row as u64 * physical.1 as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let target = self.target();
        let mut network = Network::new(name, physical);
        let view = texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.render(&mut encoder, &target, &view, format, &globals(physical, 0.0), network.frame(0.0));
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
            },
            size,
        );
        self.queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |result| result.expect("map readback"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");

        let mapped = readback.slice(..).get_mapped_range().expect("mapped range");
        let mut pixels = Vec::with_capacity((physical.0 * 4 * physical.1) as usize);
        for line in mapped.chunks(row as usize) {
            pixels.extend_from_slice(&line[..(physical.0 * 4) as usize]);
        }
        let file = std::io::BufWriter::new(std::fs::File::create(path).expect("create still"));
        let mut png = png::Encoder::new(file, physical.0, physical.1);
        png.set_color(png::ColorType::Rgba);
        png.set_depth(png::BitDepth::Eight);
        png.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        png.write_header().and_then(|mut w| w.write_image_data(&pixels)).expect("write still");
        eprintln!("{name}: wrote {}", path.display());
    }
}

impl App {
    fn add_screen(&mut self, qh: &QueueHandle<Self>, output: WlOutput) {
        let Some(info) = self.outputs.info(&output) else { return };
        let Some(mode) = info.modes.iter().find(|m| m.current) else { return };
        let physical = (mode.dimensions.0 as u32, mode.dimensions.1 as u32);
        let name = info.name.clone().unwrap_or_default();
        eprintln!("{name}: {}x{} @ {:.3} Hz", physical.0, physical.1, mode.refresh_rate as f64 / 1000.0);

        let wl_surface = self.compositor.create_surface(qh);
        let viewport = self.viewporter.get_viewport(&wl_surface, qh, Ignore);
        let layer = self.layer_shell.create_layer_surface(
            qh,
            wl_surface,
            Layer::Background,
            Some("wallpaper"),
            Some(&output),
        );
        layer.set_anchor(Anchor::all());
        layer.set_exclusive_zone(-1);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer.set_size(0, 0);
        layer.commit();

        let display = NonNull::new(self.conn.backend().display_ptr().cast()).unwrap();
        let window = NonNull::new(layer.wl_surface().id().as_ptr().cast()).unwrap();
        // SAFETY: the wl_display outlives App, and Screen drops the wgpu
        // surface before the LayerSurface that owns the wl_surface.
        let surface = unsafe {
            self.gpu.instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(RawDisplayHandle::Wayland(WaylandDisplayHandle::new(display))),
                raw_window_handle: RawWindowHandle::Wayland(WaylandWindowHandle::new(window)),
            })
        }
        .expect("wgpu surface");

        let target = self.gpu.target();
        let network = Network::new(&name, physical);
        self.screens.push(Screen {
            surface,
            viewport,
            layer,
            output,
            name,
            physical,
            target,
            network,
            last_step: None,
            configured: false,
            refresh: mode.refresh_rate as f64 / 1000.0,
            frames: 0,
            stats: FrameStats::default(),
        });
    }

    fn draw(&mut self, qh: &QueueHandle<Self>, index: usize) {
        let now = Instant::now();
        let t = (now - self.start).as_secs_f64();
        let screen = &mut self.screens[index];
        // Capped, so a screen that was covered and drawn once a second does not
        // jump when it shows again.
        let dt = screen.last_step.map_or(0.0, |last| (now - last).as_secs_f64());
        if dt > 1.5 / screen.refresh && dt < 0.5 {
            screen.stats.late += 1;
        }
        let dt = dt.min(0.1);
        screen.last_step = Some(now);

        let waiting = Instant::now();
        let frame = match screen.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                frame
            }
            other => {
                eprintln!("{}: no surface texture: {other:?}", screen.name);
                return;
            }
        };
        screen.stats.texture_wait.add(waiting.elapsed().as_secs_f64());
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.gpu.render(
            &mut encoder,
            &screen.target,
            &view,
            frame.texture.format(),
            &globals(screen.physical, t),
            screen.network.frame(dt),
        );
        self.gpu.queue.submit([encoder.finish()]);

        // The callback has to be on the surface before present commits it.
        let wl_surface = screen.layer.wl_surface();
        wl_surface.frame(qh, FrameCallbackData(wl_surface.clone()));
        self.gpu.queue.present(frame);
        screen.frames += 1;
        screen.stats.draw.add(now.elapsed().as_secs_f64());

        let since = self.last_report.elapsed().as_secs_f64();
        if since >= 5.0 {
            let rates: Vec<String> = self
                .screens
                .iter_mut()
                .map(|s| {
                    let frames = std::mem::take(&mut s.frames);
                    let stats = std::mem::take(&mut s.stats);
                    format!(
                        "{} {:.1} fps ({} late, draw {}, texture wait {})",
                        s.name,
                        frames as f64 / since,
                        stats.late,
                        stats.draw.show(frames),
                        stats.texture_wait.show(frames)
                    )
                })
                .collect();
            eprintln!("{}", rates.join(", "));
            self.last_report = Instant::now();
        }
    }
}

impl CompositorHandler for App {
    fn frame(&mut self, _: &Connection, qh: &QueueHandle<Self>, surface: &WlSurface, _: u32) {
        if let Some(i) = self.screens.iter().position(|s| s.layer.wl_surface() == surface) {
            self.draw(qh, i);
        }
    }
    fn scale_factor_changed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlSurface, _: i32) {}
    fn transform_changed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlSurface, _: wayland_client::protocol::wl_output::Transform) {}
    fn surface_enter(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlSurface, _: &WlOutput) {}
    fn surface_leave(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlSurface, _: &WlOutput) {}
}

impl OutputHandler for App {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.outputs
    }
    fn new_output(&mut self, _: &Connection, qh: &QueueHandle<Self>, output: WlOutput) {
        if let Some(dir) = &self.still {
            let Some(info) = self.outputs.info(&output) else { return };
            let Some(mode) = info.modes.iter().find(|m| m.current) else { return };
            let name = info.name.clone().unwrap_or_default();
            let path = dir.join(format!("{name}.png"));
            self.gpu.write_still(&name, (mode.dimensions.0 as u32, mode.dimensions.1 as u32), &path);
            return;
        }
        self.add_screen(qh, output);
    }
    fn update_output(&mut self, _: &Connection, qh: &QueueHandle<Self>, output: WlOutput) {
        // A mode change alters the physical size; rebuild the screen from scratch.
        self.screens.retain(|s| s.output != output);
        self.add_screen(qh, output);
    }
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, output: WlOutput) {
        self.screens.retain(|s| s.output != output);
    }
}

impl LayerShellHandler for App {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, layer: &LayerSurface) {
        self.screens.retain(|s| &s.layer != layer);
    }

    fn configure(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _: u32,
    ) {
        let Some(i) = self.screens.iter().position(|s| &s.layer == layer) else { return };
        let screen = &mut self.screens[i];
        let (w, h) = configure.new_size;

        // The buffer is the output's full pixel size, and the viewport maps it
        // onto the logical size the compositor gave us. At 125% scale that is
        // 3840x2160 pixels onto 3072x1728 logical, drawn back 1:1.
        screen.viewport.set_destination(w as i32, h as i32);

        let caps = screen.surface.get_capabilities(&self.gpu.adapter);
        let present_mode = [wgpu::PresentMode::Mailbox, wgpu::PresentMode::Immediate, wgpu::PresentMode::Fifo]
            .into_iter()
            .find(|m| caps.present_modes.contains(m))
            .unwrap();
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes[0]
        };
        screen.surface.configure(
            &self.gpu.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: caps.formats[0],
                width: screen.physical.0,
                height: screen.physical.1,
                present_mode,
                desired_maximum_frame_latency: 1,
                alpha_mode,
                view_formats: vec![],
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );
        eprintln!(
            "{}: logical {w}x{h}, buffer {}x{}, {:?}, {:?}",
            screen.name, screen.physical.0, screen.physical.1, caps.formats[0], present_mode
        );

        if !screen.configured {
            screen.configured = true;
            self.draw(qh, i);
        }
    }
}

impl ProvidesRegistryState for App {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry
    }
    registry_handlers![OutputState];
}

delegate_registry!(App);
delegate_dispatch2!(App);

/// User data for the viewporter objects, which send no events.
struct Ignore;

impl<I: Proxy> Dispatch2<I, App> for Ignore {
    fn event(&self, _: &mut App, _: &I, _: I::Event, _: &Connection, _: &QueueHandle<App>) {}
}
