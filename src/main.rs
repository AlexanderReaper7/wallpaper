//! Animated wallpaper: one background layer surface per output, each drawn
//! once per frame callback from that output, so every screen runs at its own
//! refresh rate and a covered screen stops being asked for frames.

use std::{f64::consts::TAU, ptr::NonNull, time::Instant};

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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Cube {
    rot: [f32; 4],
    place: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    resolution: [f32; 2],
    _pad: [f32; 2],
    cubes: [Cube; 3],
}

/// Per cube: yaw speed, pitch speed, pitch phase (rad/s, rad/s, rad), offset,
/// scale, brightness.
const CUBES: [([f64; 3], [f32; 2], f32, f32); 3] = [
    ([0.31, 0.23, 0.0], [0.0, 0.0], 0.9, 1.0),
    ([-0.17, 0.29, 1.0], [-1.1, 0.35], 0.45, 0.6),
    ([0.21, -0.13, 2.0], [1.2, -0.45], 0.35, 0.5),
];
const EDGES: u32 = 12 * CUBES.len() as u32;

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

struct Screen {
    // Declared first so it drops before the wl_surface it renders into.
    surface: wgpu::Surface<'static>,
    viewport: WpViewport,
    layer: LayerSurface,
    output: WlOutput,
    name: String,
    /// Buffer size in physical pixels, taken from the output's current mode.
    physical: (u32, u32),
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    configured: bool,
    frames: u32,
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
}

fn main() {
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
    };

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
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState { color: additive, alpha: additive }),
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

        let uniforms = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.gpu.layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });

        self.screens.push(Screen {
            surface,
            viewport,
            layer,
            output,
            name,
            physical,
            uniforms,
            bind_group,
            configured: false,
            frames: 0,
        });
    }

    fn draw(&mut self, qh: &QueueHandle<Self>, index: usize) {
        let t = self.start.elapsed().as_secs_f64();
        let screen = &mut self.screens[index];

        let mut globals = Globals {
            resolution: [screen.physical.0 as f32, screen.physical.1 as f32],
            _pad: [0.0; 2],
            cubes: [Cube { rot: [0.0; 4], place: [0.0; 4] }; 3],
        };
        for (out, (speed, offset, scale, brightness)) in globals.cubes.iter_mut().zip(CUBES) {
            // Wrapped in f64 so the angles keep full precision however long
            // the process runs.
            let a = (speed[0] * t).rem_euclid(TAU);
            let b = (speed[1] * t + speed[2]).rem_euclid(TAU);
            out.rot = [a.cos() as f32, a.sin() as f32, b.cos() as f32, b.sin() as f32];
            out.place = [offset[0], offset[1], scale, brightness];
        }
        self.gpu.queue.write_buffer(&screen.uniforms, 0, bytemuck::bytes_of(&globals));

        let frame = match screen.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                frame
            }
            other => {
                eprintln!("{}: no surface texture: {other:?}", screen.name);
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let format = frame.texture.format();
        let pipeline = self.gpu.pipeline(format).clone();
        let screen = &mut self.screens[index];

        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &screen.bind_group, &[]);
            pass.draw(0..4, 0..EDGES);
        }
        self.gpu.queue.submit([encoder.finish()]);

        // The callback has to be on the surface before present commits it.
        let wl_surface = screen.layer.wl_surface();
        wl_surface.frame(qh, FrameCallbackData(wl_surface.clone()));
        self.gpu.queue.present(frame);
        screen.frames += 1;

        let since = self.last_report.elapsed().as_secs_f64();
        if since >= 5.0 {
            let rates: Vec<String> = self
                .screens
                .iter_mut()
                .map(|s| format!("{} {:.1} fps", s.name, std::mem::take(&mut s.frames) as f64 / since))
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
