use wgpu::{TextureUsages, util::make_spirv};

use raytracer_shared::{PushConstants, RaytracerConfig, TileHelper};
use crate::buffers::BufferManager;


/// GPU resources and rendering pipelines
pub struct RenderState {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    
    // Pipelines
    pub compute_pipeline: wgpu::ComputePipeline,
    pub render_pipeline: wgpu::RenderPipeline,
    
    // Textures and samplers for chromatic aberration
    pub raytraced_texture_red: wgpu::Texture,
    pub raytraced_texture_green: wgpu::Texture,
    pub raytraced_texture_blue: wgpu::Texture,
    pub sampler: wgpu::Sampler,
    
    // Bind groups
    pub compute_bind_group_red: wgpu::BindGroup,
    pub compute_bind_group_green: wgpu::BindGroup,
    pub compute_bind_group_blue: wgpu::BindGroup,
    pub render_bind_group: wgpu::BindGroup,
    
    // Buffer reference tracking for optimized bind group recreation
    last_scene_metadata_buffer_ptr: *const wgpu::Buffer,
    last_triangle_buffers_count: usize,
    last_materials_buffer_ptr: *const wgpu::Buffer,
    last_textures_buffer_ptr: *const wgpu::Buffer,
    last_texture_data_buffer_ptr: *const wgpu::Buffer,
}

/// Progressive tile rendering state
pub struct ProgressiveState {
    pub needs_recompute: bool,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub current_tile: u32,
    pub is_progressive_rendering: bool,
    pub progressive_start_time: std::time::Instant,
    pub tiles_per_frame: u32,
}

/// Performance tracking
pub struct PerformanceState {
    pub start_time: std::time::Instant,
    pub last_compute_time: std::time::Duration,
    pub frame_count: u64,
}

impl RenderState {
    pub async fn new(window: &winit::window::Window) -> Self {
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
                        max_push_constant_size: RaytracerConfig::MAX_PUSH_CONSTANT_SIZE,
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

        // Create 3 separate textures for chromatic aberration (R, G, B channels)
        let (raytraced_texture_red, raytraced_texture_green, raytraced_texture_blue) = 
            Self::create_raytraced_textures(&device, size.width, size.height);

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

        // Create bind group layouts
        let (compute_pipeline, render_pipeline) = Self::create_pipelines(&device, &shader_module);

        // Create dummy bind groups (will be properly set up later)
        let dummy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy Buffer"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let compute_bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        
        // Create 3 separate compute bind groups (one for each color channel)
        let compute_bind_group_red = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group Red"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &raytraced_texture_red.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry { binding: 1, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: dummy_buffer.as_entire_binding() },
            ],
        });

        let compute_bind_group_green = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group Green"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &raytraced_texture_green.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry { binding: 1, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: dummy_buffer.as_entire_binding() },
            ],
        });

        let compute_bind_group_blue = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group Blue"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &raytraced_texture_blue.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry { binding: 1, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dummy_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: dummy_buffer.as_entire_binding() },
            ],
        });

        let render_bind_group_layout = render_pipeline.get_bind_group_layout(0);
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                // Binding 0: Red channel texture
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &raytraced_texture_red.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                // Binding 1: Green channel texture
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &raytraced_texture_green.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                // Binding 2: Blue channel texture
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &raytraced_texture_blue.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                // Binding 3: Sampler for all textures
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            compute_pipeline,
            render_pipeline,
            raytraced_texture_red,
            raytraced_texture_green,
            raytraced_texture_blue,
            sampler,
            compute_bind_group_red,
            compute_bind_group_green,
            compute_bind_group_blue,
            render_bind_group,
            // Initialize buffer tracking to null pointers
            last_scene_metadata_buffer_ptr: std::ptr::null(),
            last_triangle_buffers_count: 0,
            last_materials_buffer_ptr: std::ptr::null(),
            last_textures_buffer_ptr: std::ptr::null(),
            last_texture_data_buffer_ptr: std::ptr::null(),
        }
    }

    fn create_pipelines(device: &wgpu::Device, shader_module: &wgpu::ShaderModule) -> (wgpu::ComputePipeline, wgpu::RenderPipeline) {
        // Create bind group layouts (8 total bindings: 1 texture + 7 storage buffers)
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
                // Binding 1: Scene metadata buffer (spheres, lights, BVH nodes, triangle indices)
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
                // Binding 2: Triangle buffer 0
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Triangle buffer 1
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Triangle buffer 2
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Materials buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: Textures buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Texture data buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
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

        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Bind Group Layout"),
            entries: &[
                // Binding 0: Red channel texture
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
                // Binding 1: Green channel texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Binding 2: Blue channel texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Binding 3: Sampler for all textures
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
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
                    format: wgpu::TextureFormat::Bgra8UnormSrgb, // Will be updated during configure
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

        (compute_pipeline, render_pipeline)
    }

    /// Helper function to create raytraced textures for chromatic aberration
    fn create_raytraced_textures(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::Texture, wgpu::Texture) {
        let create_texture = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        (
            create_texture("Raytraced Texture Red"),
            create_texture("Raytraced Texture Green"),
            create_texture("Raytraced Texture Blue"),
        )
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Recreate 3 separate raytraced textures for chromatic aberration
            let (red, green, blue) = Self::create_raytraced_textures(&self.device, new_size.width, new_size.height);
            self.raytraced_texture_red = red;
            self.raytraced_texture_green = green;
            self.raytraced_texture_blue = blue;
        }
    }

    /// Recreate bind groups with updated buffer references
    pub fn recreate_bind_groups(&mut self, buffers: &BufferManager) {
        // Check if any buffer references have changed (resizing)
        let scene_metadata_ptr = &buffers.scene_metadata_buffer as *const wgpu::Buffer;
        let triangle_buffers_count = buffers.triangle_buffers.len();
        let materials_ptr = &buffers.materials_buffer as *const wgpu::Buffer;
        let textures_ptr = &buffers.textures_buffer as *const wgpu::Buffer;
        let texture_data_ptr = &buffers.texture_data_buffer as *const wgpu::Buffer;
        
        let any_buffer_resized = 
            scene_metadata_ptr != self.last_scene_metadata_buffer_ptr ||
            triangle_buffers_count != self.last_triangle_buffers_count ||
            materials_ptr != self.last_materials_buffer_ptr ||
            textures_ptr != self.last_textures_buffer_ptr ||
            texture_data_ptr != self.last_texture_data_buffer_ptr;
        
        // Recreate bind groups for both resize and content changes
        let _recreation_reason = if any_buffer_resized {
            "Buffer references changed"
        } else {
            "Buffer content changed"
        };
        
        // Update tracking state
        self.last_scene_metadata_buffer_ptr = scene_metadata_ptr;
        self.last_triangle_buffers_count = triangle_buffers_count;
        self.last_materials_buffer_ptr = materials_ptr;
        self.last_textures_buffer_ptr = textures_ptr;
        self.last_texture_data_buffer_ptr = texture_data_ptr;
        
        // Recreate 3 separate compute bind groups (one for each color channel)
        let compute_bind_group_layout = self.compute_pipeline.get_bind_group_layout(0);
        
        self.compute_bind_group_red = self.create_compute_bind_group(
            &compute_bind_group_layout, 
            &self.raytraced_texture_red, 
            buffers, 
            "Compute Bind Group Red"
        );

        self.compute_bind_group_green = self.create_compute_bind_group(
            &compute_bind_group_layout, 
            &self.raytraced_texture_green, 
            buffers, 
            "Compute Bind Group Green"
        );

        self.compute_bind_group_blue = self.create_compute_bind_group(
            &compute_bind_group_layout, 
            &self.raytraced_texture_blue, 
            buffers, 
            "Compute Bind Group Blue"
        );

        // Recreate render bind group with all 3 color channel textures
        let render_bind_group_layout = self.render_pipeline.get_bind_group_layout(0);
        self.render_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                // Binding 0: Red channel texture
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &self.raytraced_texture_red.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                // Binding 1: Green channel texture
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &self.raytraced_texture_green.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                // Binding 2: Blue channel texture
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &self.raytraced_texture_blue.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                // Binding 3: Sampler for all textures
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
    }

    /// Helper function to create compute bind groups with different output textures
    fn create_compute_bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        output_texture: &wgpu::Texture,
        buffers: &BufferManager,
        label: &str,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &output_texture.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.scene_metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.get_triangle_buffer(0)
                        .unwrap_or(&buffers.triangle_buffers[0])
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.get_triangle_buffer(1)
                        .unwrap_or(&buffers.triangle_buffers[0])
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.get_triangle_buffer(2)
                        .unwrap_or(&buffers.triangle_buffers[0])
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.materials_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.textures_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.texture_data_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Get the appropriate compute bind group for the given color channel
    pub fn get_compute_bind_group(&self, color_channel: u32) -> Result<&wgpu::BindGroup, String> {
        match color_channel {
            0 => Ok(&self.compute_bind_group_red),
            1 => Ok(&self.compute_bind_group_green),
            2 => Ok(&self.compute_bind_group_blue),
            _ => Err(format!("Invalid color channel: {}. Valid channels are 0 (red), 1 (green), 2 (blue)", color_channel)),
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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

impl ProgressiveState {
    pub fn new(width: u32, height: u32) -> Self {
        let (tiles_x, tiles_y) = TileHelper::calculate_tile_count(width, height, RaytracerConfig::TILE_SIZE);
        let total_tiles = tiles_x * tiles_y;
        let tiles_per_frame = TileHelper::calculate_tiles_per_frame(total_tiles);

        Self {
            needs_recompute: true,
            tiles_x,
            tiles_y,
            current_tile: 0,
            is_progressive_rendering: false,
            progressive_start_time: std::time::Instant::now(),
            tiles_per_frame,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let (tiles_x, tiles_y) = TileHelper::calculate_tile_count(width, height, RaytracerConfig::TILE_SIZE);
        self.tiles_x = tiles_x;
        self.tiles_y = tiles_y;
        let total_tiles = tiles_x * tiles_y;
        
        self.tiles_per_frame = TileHelper::calculate_tiles_per_frame(total_tiles);
        self.needs_recompute = true;
        self.current_tile = 0;
        self.is_progressive_rendering = false;
    }

    pub fn trigger_recompute(&mut self) {
        self.needs_recompute = true;
        self.current_tile = 0;
        self.is_progressive_rendering = false;
    }
}

impl PerformanceState {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            last_compute_time: std::time::Duration::ZERO,
            frame_count: 0,
        }
    }

    pub fn update_frame_count(&mut self) {
        self.frame_count += 1;
        
        // Print performance stats every 60 frames
        if self.frame_count % RaytracerConfig::PERFORMANCE_STATS_INTERVAL == 0 {
            let elapsed = self.start_time.elapsed().as_secs_f32();
            let fps = self.frame_count as f32 / elapsed;
            println!("FPS: {:.1}, Last compute: {:.2}ms", 
                     fps, 
                     self.last_compute_time.as_secs_f32() * RaytracerConfig::MILLISECONDS_PER_SECOND);
        }
    }
}