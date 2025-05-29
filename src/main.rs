#![feature(build_hasher_simple_hash_one)]

use winit::{
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState, MouseButton},
    event_loop::EventLoop,
    window::WindowBuilder,
};
use bytemuck;
use wgpu::{TextureUsages, util::make_spirv};
use raytracer_shared::{Camera, Sphere, PushConstants};


struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    
    // Compute pipeline for raytracing
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    raytraced_texture: wgpu::Texture,
    spheres_buffer: wgpu::Buffer,
    
    // Render pipeline for displaying texture
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    
    // Scene data
    camera: Camera,
    spheres: Vec<Sphere>,
    
    // Input state
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    
    // Performance metrics
    start_time: std::time::Instant,
    last_compute_time: std::time::Duration,
    frame_count: u64,
    
    // Progressive tile rendering
    needs_recompute: bool,
    tile_size: u32,
    tiles_x: u32,
    tiles_y: u32,
    current_tile: u32,
    is_progressive_rendering: bool,
    progressive_start_time: std::time::Instant,
}

impl State {
    async fn new(window: &winit::window::Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::PUSH_CONSTANTS | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    limits: wgpu::Limits {
                        max_push_constant_size: 128,
                        ..Default::default()
                    },
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_DST,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // Load shader binary
        let shader_binary = include_bytes!(env!("shader.spv"));
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader Module"),
            source: make_spirv(shader_binary),
        });


        // Create texture for raytraced output
        let raytraced_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Raytraced Texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create sampler for texture
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Initialize default camera
        let camera = Camera::new();

        // Initialize default scene
        let spheres = vec![
            Sphere::new([0.0, 0.0, -1.0], 0.5, [0.8, 0.3, 0.3], 0),
            Sphere::new([-1.0, 0.0, -1.0], 0.5, [0.8, 0.8, 0.2], 1),
            Sphere::new([1.0, 0.0, -1.0], 0.5, [0.2, 0.3, 0.8], 2),
        ];

        // Create spheres buffer
        let spheres_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spheres Buffer"),
            size: (std::mem::size_of::<Sphere>() * 64) as u64, // Support up to 64 spheres
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Compute bind group layout and bind group
        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &raytraced_texture.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spheres_buffer.as_entire_binding(),
                },
            ],
        });

        // Render bind group layout and bind group
        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &raytraced_texture.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Compute pipeline
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: "main_cs",
        });

        // Render pipeline
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "main_vs",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "main_fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Calculate tile layout
        let tile_size = 64; // 64x64 pixel tiles
        let tiles_x = (size.width + tile_size - 1) / tile_size;
        let tiles_y = (size.height + tile_size - 1) / tile_size;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            compute_pipeline,
            compute_bind_group,
            raytraced_texture,
            spheres_buffer,
            render_pipeline,
            render_bind_group,
            sampler,
            camera,
            spheres,
            mouse_pressed: false,
            last_mouse_pos: None,
            start_time: std::time::Instant::now(),
            last_compute_time: std::time::Duration::ZERO,
            frame_count: 0,
            needs_recompute: true,
            tile_size,
            tiles_x,
            tiles_y,
            current_tile: 0,
            is_progressive_rendering: false,
            progressive_start_time: std::time::Instant::now(),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Recalculate tile layout
            self.tiles_x = (new_size.width + self.tile_size - 1) / self.tile_size;
            self.tiles_y = (new_size.height + self.tile_size - 1) / self.tile_size;

            // Recreate raytraced texture
            self.raytraced_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Raytraced Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            // Recreate bind groups with new texture
            self.recreate_bind_groups();

            self.needs_recompute = true;
            self.current_tile = 0;
            self.is_progressive_rendering = false;
        }
    }
    
    fn recreate_bind_groups(&mut self) {
        // Recreate compute bind group
        let compute_bind_group_layout = self.compute_pipeline.get_bind_group_layout(0);
        self.compute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &self.raytraced_texture.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.spheres_buffer.as_entire_binding(),
                },
            ],
        });

        // Recreate render bind group
        let render_bind_group_layout = self.render_pipeline.get_bind_group_layout(0);
        self.render_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &self.raytraced_texture.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
    }
    
    fn run_compute(&mut self) {
        // Start progressive rendering if we need to recompute
        if self.needs_recompute && !self.is_progressive_rendering {
            self.is_progressive_rendering = true;
            self.current_tile = 0;
            self.progressive_start_time = std::time::Instant::now();
            self.needs_recompute = false;
            
            println!("Starting progressive render: {}x{} tiles", self.tiles_x, self.tiles_y);
        }
        
        // Continue progressive rendering if in progress
        if !self.is_progressive_rendering {
            return;
        }

        let total_tiles = self.tiles_x * self.tiles_y;
        if self.current_tile >= total_tiles {
            // Progressive rendering completed
            self.is_progressive_rendering = false;
            let total_time = self.progressive_start_time.elapsed();
            println!("Progressive render completed in {:.2}ms", total_time.as_secs_f32() * 1000.0);
            return;
        }

        let compute_start = std::time::Instant::now();
        
        // Calculate current tile position
        let tile_x = self.current_tile % self.tiles_x;
        let tile_y = self.current_tile / self.tiles_x;
        
        let tile_offset_x = tile_x * self.tile_size;
        let tile_offset_y = tile_y * self.tile_size;
        
        // Calculate actual tile size (handle edge tiles)
        let actual_tile_width = std::cmp::min(self.tile_size, self.size.width - tile_offset_x);
        let actual_tile_height = std::cmp::min(self.tile_size, self.size.height - tile_offset_y);
        
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);

            // Update spheres buffer if needed
            self.queue.write_buffer(
                &self.spheres_buffer,
                0,
                bytemuck::cast_slice(&self.spheres),
            );

            let push_constants = PushConstants {
                resolution: [self.size.width as f32, self.size.height as f32],
                time: self.start_time.elapsed().as_secs_f32(),
                camera: self.camera,
                sphere_count: self.spheres.len() as u32,
                tile_offset: [tile_offset_x, tile_offset_y],
                tile_size: [actual_tile_width, actual_tile_height],
                total_tiles: [self.tiles_x, self.tiles_y],
                current_tile_index: self.current_tile,
            };
            compute_pass.set_push_constants(0, bytemuck::cast_slice(&[push_constants]));

            // Dispatch workgroups for current tile
            let workgroup_x = (actual_tile_width + 15) / 16;
            let workgroup_y = (actual_tile_height + 15) / 16;
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Don't wait for GPU - allow frame to be displayed immediately
        self.last_compute_time = compute_start.elapsed();
        
        // Move to next tile
        self.current_tile += 1;
        
        // Progress feedback
        if self.current_tile % 4 == 0 || self.current_tile == total_tiles {
            let progress = (self.current_tile as f32 / total_tiles as f32) * 100.0;
            println!("Progress: {:.1}% ({}/{})", progress, self.current_tile, total_tiles);
        }
    }

    pub fn trigger_recompute(&mut self) {
        self.needs_recompute = true;
    }

    fn rotate_camera(&mut self, delta_x: f64, delta_y: f64) {
        let sensitivity = 0.005;
        let delta_x = delta_x as f32 * sensitivity;
        let delta_y = delta_y as f32 * sensitivity;
        
        // Create rotation around Y axis (yaw)
        let cos_yaw = delta_x.cos();
        let sin_yaw = delta_x.sin();
        
        // Rotate direction vector around Y axis
        let old_dir_x = self.camera.direction[0];
        let old_dir_z = self.camera.direction[2];
        self.camera.direction[0] = old_dir_x * cos_yaw - old_dir_z * sin_yaw;
        self.camera.direction[2] = old_dir_x * sin_yaw + old_dir_z * cos_yaw;
        
        // Simple pitch rotation (just modify Y component)
        self.camera.direction[1] = (self.camera.direction[1] - delta_y).clamp(-0.99, 0.99);
        
        // Normalize direction
        let len = (self.camera.direction[0].powi(2) + 
                  self.camera.direction[1].powi(2) + 
                  self.camera.direction[2].powi(2)).sqrt();
        if len > 0.0 {
            self.camera.direction[0] /= len;
            self.camera.direction[1] /= len;
            self.camera.direction[2] /= len;
        }
        
        self.needs_recompute = true;
        self.current_tile = 0;
        self.is_progressive_rendering = false;
    }
    
    fn move_camera(&mut self, forward: f32, right: f32) {
        let speed = 0.1;
        
        // Forward/backward movement
        self.camera.position[0] += self.camera.direction[0] * forward * speed;
        self.camera.position[1] += self.camera.direction[1] * forward * speed;
        self.camera.position[2] += self.camera.direction[2] * forward * speed;
        
        // Right/left movement (cross product of direction and up)
        let right_vec = [
            self.camera.direction[1] * self.camera.up[2] - self.camera.direction[2] * self.camera.up[1],
            self.camera.direction[2] * self.camera.up[0] - self.camera.direction[0] * self.camera.up[2],
            self.camera.direction[0] * self.camera.up[1] - self.camera.direction[1] * self.camera.up[0],
        ];
        
        self.camera.position[0] += right_vec[0] * right * speed;
        self.camera.position[1] += right_vec[1] * right * speed;
        self.camera.position[2] += right_vec[2] * right * speed;
        
        self.needs_recompute = true;
        self.current_tile = 0;
        self.is_progressive_rendering = false;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.frame_count += 1;
        
        // Print performance stats every 60 frames
        if self.frame_count % 60 == 0 {
            let elapsed = self.start_time.elapsed().as_secs_f32();
            let fps = self.frame_count as f32 / elapsed;
            println!("FPS: {:.1}, Last compute: {:.2}ms", 
                     fps, 
                     self.last_compute_time.as_secs_f32() * 1000.0);
        }
        
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);
            render_pass.draw(0..3, 0..1); // Draw fullscreen triangle
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("GPU Raytracer")
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await;

    //window.set_resize_increments()

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => control_flow.set_exit(),
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    state.resize(**new_inner_size);
                }
                WindowEvent::KeyboardInput {
                    input: KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(key),
                        ..
                    },
                    ..
                } => {
                    match key {
                        VirtualKeyCode::Space => {
                            state.needs_recompute = true;
                        }
                        VirtualKeyCode::W => {
                            state.move_camera(1.0, 0.0);
                        }
                        VirtualKeyCode::S => {
                            state.move_camera(-1.0, 0.0);
                        }
                        VirtualKeyCode::A => {
                            state.move_camera(0.0, -1.0);
                        }
                        VirtualKeyCode::D => {
                            state.move_camera(0.0, 1.0);
                        }
                        VirtualKeyCode::Escape => {
                            control_flow.set_exit();
                        }
                        _ => {}
                    }
                }
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state: button_state,
                    ..
                } => {
                    state.mouse_pressed = *button_state == ElementState::Pressed;
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if state.mouse_pressed {
                        if let Some(last_pos) = state.last_mouse_pos {
                            let delta_x = position.x - last_pos.0;
                            let delta_y = position.y - last_pos.1;
                            state.rotate_camera(delta_x, delta_y);
                        }
                    }
                    state.last_mouse_pos = Some((position.x, position.y));
                }
                _ => {}
            },
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                state.run_compute();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => control_flow.set_exit(),
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}
