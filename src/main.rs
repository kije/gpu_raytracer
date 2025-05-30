#![feature(build_hasher_simple_hash_one)]

use winit::{
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState, MouseButton},
    event_loop::EventLoop,
    window::WindowBuilder,
};
use bytemuck;
use wgpu::{TextureUsages, util::make_spirv};
use raytracer_shared::{Camera, Sphere, Triangle, Material, Light, TextureInfo, PushConstants, RaytracerConfig, TileHelper, SceneBuilder};

mod gltf_loader;
use gltf_loader::{GltfLoader, GltfError, LoadedScene};

/// GPU resources and rendering pipelines
struct RenderState {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    
    // Pipelines
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    
    // Textures and samplers
    raytraced_texture: wgpu::Texture,
    sampler: wgpu::Sampler,
    
    // Bind groups
    compute_bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,
}

/// Smart buffer management for geometry data
struct BufferManager {
    spheres_buffer: wgpu::Buffer,
    triangles_buffer: wgpu::Buffer,
    materials_buffer: wgpu::Buffer,
    lights_buffer: wgpu::Buffer,
    textures_buffer: wgpu::Buffer,
    texture_data_buffer: wgpu::Buffer,
    spheres_capacity: usize,
    triangles_capacity: usize,
    materials_capacity: usize,
    lights_capacity: usize,
    textures_capacity: usize,
    texture_data_capacity: usize,
    spheres_dirty: bool,
    triangles_dirty: bool,
    materials_dirty: bool,
    lights_dirty: bool,
    textures_dirty: bool,
    texture_data_dirty: bool,
}

/// Scene geometry and camera
struct SceneState {
    camera: Camera,
    spheres: Vec<Sphere>,
    triangles: Vec<Triangle>,
    materials: Vec<Material>,
    lights: Vec<Light>,
    textures: Vec<TextureInfo>,
    texture_data: Vec<u8>,
}

/// Progressive tile rendering state
struct ProgressiveState {
    needs_recompute: bool,
    tiles_x: u32,
    tiles_y: u32,
    current_tile: u32,
    is_progressive_rendering: bool,
    progressive_start_time: std::time::Instant,
    tiles_per_frame: u32,
}

/// Input handling state
struct InputState {
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
}

/// Performance tracking
struct PerformanceState {
    start_time: std::time::Instant,
    last_compute_time: std::time::Duration,
    frame_count: u64,
}

impl RenderState {
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

        // Create placeholder bind groups (will be properly set up later)
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
            ],
        });

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

        // Create dummy bind groups (will be updated with proper buffers)
        let dummy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy Buffer"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
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
                    resource: dummy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dummy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dummy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dummy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dummy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dummy_buffer.as_entire_binding(),
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

        Self {
            surface,
            device,
            queue,
            config,
            size,
            compute_pipeline,
            render_pipeline,
            raytraced_texture,
            sampler,
            compute_bind_group,
            render_bind_group,
        }
    }
}

impl BufferManager {
    fn new(device: &wgpu::Device) -> Self {
        let spheres_capacity = RaytracerConfig::DEFAULT_MAX_SPHERES;
        let triangles_capacity = RaytracerConfig::DEFAULT_MAX_TRIANGLES;
        let materials_capacity = 64; // Default materials capacity
        let lights_capacity = 32; // Default lights capacity
        let textures_capacity = 64; // Default textures capacity
        let texture_data_capacity = 256 * 1024; // 1MB default texture data capacity (in u32 units)

        let spheres_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spheres Buffer"),
            size: (std::mem::size_of::<Sphere>() * spheres_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let triangles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Triangles Buffer"),
            size: (std::mem::size_of::<Triangle>() * triangles_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let materials_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Materials Buffer"),
            size: (std::mem::size_of::<Material>() * materials_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lights Buffer"),
            size: (std::mem::size_of::<Light>() * lights_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let textures_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Textures Buffer"),
            size: (std::mem::size_of::<TextureInfo>() * textures_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let texture_data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Texture Data Buffer"),
            size: (texture_data_capacity * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            spheres_buffer,
            triangles_buffer,
            materials_buffer,
            lights_buffer,
            textures_buffer,
            texture_data_buffer,
            spheres_capacity,
            triangles_capacity,
            materials_capacity,
            lights_capacity,
            textures_capacity,
            texture_data_capacity,
            spheres_dirty: true,
            triangles_dirty: true,
            materials_dirty: true,
            lights_dirty: true,
            textures_dirty: true,
            texture_data_dirty: true,
        }
    }

    /// Update spheres buffer with new data, resizing if necessary
    fn update_spheres(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, spheres: &[Sphere]) -> bool {
        let needs_resize = spheres.len() > self.spheres_capacity;
        
        if needs_resize {
            // Double the capacity to accommodate growth
            self.spheres_capacity = (spheres.len() * 2).max(RaytracerConfig::DEFAULT_MAX_SPHERES);
            
            self.spheres_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Spheres Buffer"),
                size: (std::mem::size_of::<Sphere>() * self.spheres_capacity) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            println!("Resized spheres buffer to capacity: {}", self.spheres_capacity);
        }
        
        if self.spheres_dirty || needs_resize {
            queue.write_buffer(
                &self.spheres_buffer,
                0,
                bytemuck::cast_slice(spheres),
            );
            self.spheres_dirty = false;
        }
        
        needs_resize
    }

    /// Update triangles buffer with new data, resizing if necessary
    fn update_triangles(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, triangles: &[Triangle]) -> bool {
        let needs_resize = triangles.len() > self.triangles_capacity;
        
        if needs_resize {
            // Double the capacity to accommodate growth
            self.triangles_capacity = (triangles.len() * 2).max(RaytracerConfig::DEFAULT_MAX_TRIANGLES);
            
            self.triangles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Triangles Buffer"),
                size: (std::mem::size_of::<Triangle>() * self.triangles_capacity) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            println!("Resized triangles buffer to capacity: {}", self.triangles_capacity);
        }
        
        if self.triangles_dirty || needs_resize {
            queue.write_buffer(
                &self.triangles_buffer,
                0,
                bytemuck::cast_slice(triangles),
            );
            self.triangles_dirty = false;
        }
        
        needs_resize
    }

    /// Mark spheres buffer as dirty (needs update)
    fn mark_spheres_dirty(&mut self) {
        self.spheres_dirty = true;
    }

    /// Mark triangles buffer as dirty (needs update)
    fn mark_triangles_dirty(&mut self) {
        self.triangles_dirty = true;
    }

    /// Update materials buffer with new data, resizing if necessary
    fn update_materials(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, materials: &[Material]) -> bool {
        let needs_resize = materials.len() > self.materials_capacity;
        
        if needs_resize {
            // Double the capacity to accommodate growth
            self.materials_capacity = (materials.len() * 2).max(64);
            
            self.materials_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Materials Buffer"),
                size: (std::mem::size_of::<Material>() * self.materials_capacity) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            println!("Resized materials buffer to capacity: {}", self.materials_capacity);
        }
        
        if self.materials_dirty || needs_resize {
            queue.write_buffer(
                &self.materials_buffer,
                0,
                bytemuck::cast_slice(materials),
            );
            self.materials_dirty = false;
        }
        
        needs_resize
    }

    /// Mark materials buffer as dirty (needs update)
    fn mark_materials_dirty(&mut self) {
        self.materials_dirty = true;
    }

    /// Update lights buffer with new data, resizing if necessary
    fn update_lights(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, lights: &[Light]) -> bool {
        let needs_resize = lights.len() > self.lights_capacity;
        
        if needs_resize {
            // Double the capacity to accommodate growth
            self.lights_capacity = (lights.len() * 2).max(32);
            
            self.lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Lights Buffer"),
                size: (std::mem::size_of::<Light>() * self.lights_capacity) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            println!("Resized lights buffer to capacity: {}", self.lights_capacity);
        }
        
        if self.lights_dirty || needs_resize {
            queue.write_buffer(
                &self.lights_buffer,
                0,
                bytemuck::cast_slice(lights),
            );
            self.lights_dirty = false;
        }
        
        needs_resize
    }

    /// Update textures buffer with new data, resizing if necessary
    fn update_textures(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, textures: &[TextureInfo]) -> bool {
        let needs_resize = textures.len() > self.textures_capacity;
        
        if needs_resize {
            // Double the capacity to accommodate growth
            self.textures_capacity = (textures.len() * 2).max(64);
            
            self.textures_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Textures Buffer"),
                size: (std::mem::size_of::<TextureInfo>() * self.textures_capacity) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            println!("Resized textures buffer to capacity: {}", self.textures_capacity);
        }
        
        if self.textures_dirty || needs_resize {
            queue.write_buffer(
                &self.textures_buffer,
                0,
                bytemuck::cast_slice(textures),
            );
            self.textures_dirty = false;
        }
        
        needs_resize
    }

    /// Update texture data buffer with new data, resizing if necessary
    fn update_texture_data(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, texture_data: &[u8]) -> bool {
        // Pack u8 data into u32 for GPU compatibility (4 bytes per u32)
        let mut packed_data = Vec::new();
        for chunk in texture_data.chunks(4) {
            let mut packed: u32 = 0;
            for (i, &byte) in chunk.iter().enumerate() {
                packed |= (byte as u32) << (i * 8);
            }
            packed_data.push(packed);
        }
        
        let needs_resize = packed_data.len() > self.texture_data_capacity;
        
        if needs_resize {
            // Double the capacity to accommodate growth (capacity is in u32 units)
            self.texture_data_capacity = (packed_data.len() * 2).max(256 * 1024); // 1MB in u32 units
            
            self.texture_data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Texture Data Buffer"),
                size: (self.texture_data_capacity * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            println!("Resized texture data buffer to capacity: {} u32s", self.texture_data_capacity);
        }
        
        if self.texture_data_dirty || needs_resize {
            queue.write_buffer(
                &self.texture_data_buffer,
                0,
                bytemuck::cast_slice(&packed_data),
            );
            self.texture_data_dirty = false;
        }
        
        needs_resize
    }

    /// Mark lights buffer as dirty (needs update)
    fn mark_lights_dirty(&mut self) {
        self.lights_dirty = true;
    }

    /// Mark textures buffer as dirty (needs update)
    fn mark_textures_dirty(&mut self) {
        self.textures_dirty = true;
    }

    /// Mark texture data buffer as dirty (needs update)
    fn mark_texture_data_dirty(&mut self) {
        self.texture_data_dirty = true;
    }

    /// Check if any buffers need updating
    fn needs_update(&self) -> bool {
        self.spheres_dirty || self.triangles_dirty || self.materials_dirty || 
        self.lights_dirty || self.textures_dirty || self.texture_data_dirty
    }
}

impl SceneState {
    fn new() -> Self {
        let (spheres, triangles, materials) = SceneBuilder::build_default_scene();
        
        Self {
            camera: Camera::new(),
            spheres,
            triangles,
            materials,
            lights: Vec::new(),
            textures: Vec::new(),
            texture_data: Vec::new(),
        }
    }

    /// Load scene from glTF file
    fn load_from_gltf<P: AsRef<std::path::Path>>(path: P) -> Result<Self, GltfError> {
        let loader = GltfLoader::load_from_path(&path)?;
        let loaded_scene = loader.extract_scene(None)?;
        
        // Use the first camera from glTF if available, otherwise default camera
        let camera = loaded_scene.cameras.first().copied().unwrap_or_else(Camera::new);
        
        Ok(Self {
            camera,
            spheres: loaded_scene.spheres,
            triangles: loaded_scene.triangles,
            materials: loaded_scene.materials,
            lights: loaded_scene.lights,
            textures: loaded_scene.textures,
            texture_data: loaded_scene.texture_data,
        })
    }

    /// Load scene from glTF file with fallback to default scene
    fn load_from_gltf_or_default<P: AsRef<std::path::Path>>(path: P) -> Self {
        match Self::load_from_gltf(path.as_ref()) {
            Ok(scene) => {
                println!("Successfully loaded glTF scene from: {:?}", path.as_ref());
                scene
            }
            Err(e) => {
                println!("Failed to load glTF scene from {:?}, using default scene. Error: {:?}", 
                         path.as_ref(), e);
                Self::new()
            }
        }
    }

    /// Replace current scene with glTF data
    fn replace_with_gltf<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<(), GltfError> {
        let loader = GltfLoader::load_from_path(&path)?;
        let loaded_scene = loader.extract_scene(None)?;
        
        // Use the first camera from glTF if available, otherwise keep current camera
        if let Some(new_camera) = loaded_scene.cameras.first() {
            self.camera = *new_camera;
            println!("Loaded camera from glTF: pos={:?}, dir={:?}, fov={}", 
                     self.camera.position, self.camera.direction, self.camera.fov);
        }
        
        self.spheres = loaded_scene.spheres;
        self.triangles = loaded_scene.triangles;
        self.materials = loaded_scene.materials;
        self.lights = loaded_scene.lights;
        self.textures = loaded_scene.textures;
        self.texture_data = loaded_scene.texture_data;
        
        println!("Loaded glTF scene: {} spheres, {} triangles, {} materials, {} lights, {} textures",
                 self.spheres.len(), self.triangles.len(), self.materials.len(),
                 self.lights.len(), self.textures.len());
        
        Ok(())
    }
}

impl ProgressiveState {
    fn new(width: u32, height: u32) -> Self {
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
}

impl InputState {
    fn new() -> Self {
        Self {
            mouse_pressed: false,
            last_mouse_pos: None,
        }
    }
}

impl PerformanceState {
    fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            last_compute_time: std::time::Duration::ZERO,
            frame_count: 0,
        }
    }
}

/// Main application state
struct State {
    render: RenderState,
    buffers: BufferManager,
    scene: SceneState,
    progressive: ProgressiveState,
    input: InputState,
    performance: PerformanceState,
}

impl State {
    async fn new(window: &winit::window::Window) -> Self {
        let size = window.inner_size();

        // Initialize render state
        let mut render = RenderState::new(window).await;
        
        // Initialize buffer manager
        let buffers = BufferManager::new(&render.device);
        
        // Initialize scene state
        let scene = SceneState::new();
        
        // Initialize progressive rendering state
        let progressive = ProgressiveState::new(size.width, size.height);
        
        // Initialize input state
        let input = InputState::new();
        
        // Initialize performance state
        let performance = PerformanceState::new();

        // Now update the render state's bind groups with the proper buffers
        let compute_bind_group_layout = render.compute_pipeline.get_bind_group_layout(0);
        render.compute_bind_group = render.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &render.raytraced_texture.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.spheres_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.triangles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.materials_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.textures_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.texture_data_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            render,
            buffers,
            scene,
            progressive,
            input,
            performance,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.render.size = new_size;
            self.render.config.width = new_size.width;
            self.render.config.height = new_size.height;
            self.render.surface.configure(&self.render.device, &self.render.config);

            // Recalculate tile layout and adaptive tiles per frame
            let (tiles_x, tiles_y) = TileHelper::calculate_tile_count(new_size.width, new_size.height, RaytracerConfig::TILE_SIZE);
            self.progressive.tiles_x = tiles_x;
            self.progressive.tiles_y = tiles_y;
            let total_tiles = tiles_x * tiles_y;
            
            self.progressive.tiles_per_frame = TileHelper::calculate_tiles_per_frame(total_tiles);

            // Recreate raytraced texture
            self.render.raytraced_texture = self.render.device.create_texture(&wgpu::TextureDescriptor {
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

            self.progressive.needs_recompute = true;
            self.progressive.current_tile = 0;
            self.progressive.is_progressive_rendering = false;
        }
    }
    
    fn recreate_bind_groups(&mut self) {
        // Recreate compute bind group
        let compute_bind_group_layout = self.render.compute_pipeline.get_bind_group_layout(0);
        self.render.compute_bind_group = self.render.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &self.render.raytraced_texture.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffers.spheres_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers.triangles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.materials_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buffers.lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buffers.textures_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buffers.texture_data_buffer.as_entire_binding(),
                },
            ],
        });

        // Recreate render bind group
        let render_bind_group_layout = self.render.render_pipeline.get_bind_group_layout(0);
        self.render.render_bind_group = self.render.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &self.render.raytraced_texture.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.render.sampler),
                },
            ],
        });
    }
    
    fn run_compute(&mut self) {
        // Start progressive rendering if we need to recompute
        if self.progressive.needs_recompute && !self.progressive.is_progressive_rendering {
            self.progressive.is_progressive_rendering = true;
            self.progressive.current_tile = 0;
            self.progressive.progressive_start_time = std::time::Instant::now();
            self.progressive.needs_recompute = false;
            
            println!("Starting parallel progressive render: {}x{} tiles, {} tiles per frame", 
                     self.progressive.tiles_x, self.progressive.tiles_y, self.progressive.tiles_per_frame);
        }
        
        // Continue progressive rendering if in progress
        if !self.progressive.is_progressive_rendering {
            return;
        }

        let total_tiles = self.progressive.tiles_x * self.progressive.tiles_y;
        if self.progressive.current_tile >= total_tiles {
            // Progressive rendering completed
            self.progressive.is_progressive_rendering = false;
            let total_time = self.progressive.progressive_start_time.elapsed();
            println!("Progressive render completed in {:.2}ms", total_time.as_secs_f32() * RaytracerConfig::MILLISECONDS_PER_SECOND);
            return;
        }

        let compute_start = std::time::Instant::now();
        
        // Calculate how many tiles to process this frame
        let tiles_remaining = total_tiles - self.progressive.current_tile;
        let tiles_this_frame = std::cmp::min(self.progressive.tiles_per_frame, tiles_remaining);
        
        let mut encoder = self
            .render.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Parallel Compute Encoder"),
            });
        
        // Update geometry buffers using smart buffer management
        let spheres_resized = self.buffers.update_spheres(&self.render.device, &self.render.queue, &self.scene.spheres);
        let triangles_resized = self.buffers.update_triangles(&self.render.device, &self.render.queue, &self.scene.triangles);
        let materials_resized = self.buffers.update_materials(&self.render.device, &self.render.queue, &self.scene.materials);
        let lights_resized = self.buffers.update_lights(&self.render.device, &self.render.queue, &self.scene.lights);
        let textures_resized = self.buffers.update_textures(&self.render.device, &self.render.queue, &self.scene.textures);
        let texture_data_resized = self.buffers.update_texture_data(&self.render.device, &self.render.queue, &self.scene.texture_data);
        
        // If buffers were resized, recreate bind groups with new buffers
        if spheres_resized || triangles_resized || materials_resized || lights_resized || textures_resized || texture_data_resized {
            self.recreate_bind_groups();
        }
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Parallel Compute Pass"),
            });

            compute_pass.set_pipeline(&self.render.compute_pipeline);
            compute_pass.set_bind_group(0, &self.render.compute_bind_group, &[]);

            // Process multiple tiles in parallel
            for i in 0..tiles_this_frame {
                let tile_index = self.progressive.current_tile + i;
                let tile_x = tile_index % self.progressive.tiles_x;
                let tile_y = tile_index / self.progressive.tiles_x;
                
                let tile_offset_x = tile_x * RaytracerConfig::TILE_SIZE;
                let tile_offset_y = tile_y * RaytracerConfig::TILE_SIZE;
                
                // Calculate actual tile size (handle edge tiles)
                let actual_tile_width = std::cmp::min(RaytracerConfig::TILE_SIZE, self.render.size.width - tile_offset_x);
                let actual_tile_height = std::cmp::min(RaytracerConfig::TILE_SIZE, self.render.size.height - tile_offset_y);
                
                let push_constants = PushConstants::new(
                    [self.render.size.width as f32, self.render.size.height as f32],
                    self.performance.start_time.elapsed().as_secs_f32(),
                    self.scene.camera,
                    self.scene.spheres.len() as u32,
                    self.scene.triangles.len() as u32,
                    self.scene.materials.len() as u32,
                    self.scene.lights.len() as u32,
                    [tile_offset_x, tile_offset_y],
                    [actual_tile_width, actual_tile_height],
                    [self.progressive.tiles_x, self.progressive.tiles_y],
                );
                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[push_constants]));

                // Dispatch workgroups for current tile
                let workgroup_x = (actual_tile_width + RaytracerConfig::THREAD_GROUP_SIZE.0 - 1) / RaytracerConfig::THREAD_GROUP_SIZE.0;
                let workgroup_y = (actual_tile_height + RaytracerConfig::THREAD_GROUP_SIZE.1 - 1) / RaytracerConfig::THREAD_GROUP_SIZE.1;
                compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
            }
        }

        self.render.queue.submit(std::iter::once(encoder.finish()));
        
        // Don't wait for GPU - allow frame to be displayed immediately
        self.performance.last_compute_time = compute_start.elapsed();
        
        // Move to next set of tiles
        self.progressive.current_tile += tiles_this_frame;
        
        // Progress feedback
        if self.progressive.current_tile % self.progressive.tiles_per_frame == 0 || self.progressive.current_tile == total_tiles {
            let progress = (self.progressive.current_tile as f32 / total_tiles as f32) * RaytracerConfig::PROGRESS_PERCENTAGE_SCALE;
            println!("Progress: {:.1}% ({}/{}) - {} tiles/frame", 
                     progress, self.progressive.current_tile, total_tiles, tiles_this_frame);
        }
    }

    pub fn trigger_recompute(&mut self) {
        self.progressive.needs_recompute = true;
    }

    fn rotate_camera(&mut self, delta_x: f64, delta_y: f64) {
        let sensitivity = RaytracerConfig::CAMERA_ROTATE_SENSITIVITY;
        let delta_x = delta_x as f32 * sensitivity;
        let delta_y = delta_y as f32 * sensitivity;
        
        // Create rotation around Y axis (yaw)
        let cos_yaw = delta_x.cos();
        let sin_yaw = delta_x.sin();
        
        // Rotate direction vector around Y axis
        let old_dir_x = self.scene.camera.direction[0];
        let old_dir_z = self.scene.camera.direction[2];
        self.scene.camera.direction[0] = old_dir_x * cos_yaw - old_dir_z * sin_yaw;
        self.scene.camera.direction[2] = old_dir_x * sin_yaw + old_dir_z * cos_yaw;
        
        // Simple pitch rotation (just modify Y component)
        self.scene.camera.direction[1] = (self.scene.camera.direction[1] - delta_y).clamp(-RaytracerConfig::CAMERA_PITCH_CLAMP, RaytracerConfig::CAMERA_PITCH_CLAMP);
        
        // Normalize direction
        let len = (self.scene.camera.direction[0].powi(2) + 
                  self.scene.camera.direction[1].powi(2) + 
                  self.scene.camera.direction[2].powi(2)).sqrt();
        if len > 0.0 {
            self.scene.camera.direction[0] /= len;
            self.scene.camera.direction[1] /= len;
            self.scene.camera.direction[2] /= len;
        }
        
        self.progressive.needs_recompute = true;
        self.progressive.current_tile = 0;
        self.progressive.is_progressive_rendering = false;
    }
    
    fn move_camera(&mut self, forward: f32, right: f32) {
        let speed = RaytracerConfig::CAMERA_MOVE_SPEED;
        
        // Forward/backward movement
        self.scene.camera.position[0] += self.scene.camera.direction[0] * forward * speed;
        self.scene.camera.position[1] += self.scene.camera.direction[1] * forward * speed;
        self.scene.camera.position[2] += self.scene.camera.direction[2] * forward * speed;
        
        // Right/left movement (cross product of direction and up)
        let right_vec = [
            self.scene.camera.direction[1] * self.scene.camera.up[2] - self.scene.camera.direction[2] * self.scene.camera.up[1],
            self.scene.camera.direction[2] * self.scene.camera.up[0] - self.scene.camera.direction[0] * self.scene.camera.up[2],
            self.scene.camera.direction[0] * self.scene.camera.up[1] - self.scene.camera.direction[1] * self.scene.camera.up[0],
        ];
        
        self.scene.camera.position[0] += right_vec[0] * right * speed;
        self.scene.camera.position[1] += right_vec[1] * right * speed;
        self.scene.camera.position[2] += right_vec[2] * right * speed;
        
        self.progressive.needs_recompute = true;
        self.progressive.current_tile = 0;
        self.progressive.is_progressive_rendering = false;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.performance.frame_count += 1;
        
        // Print performance stats every 60 frames
        if self.performance.frame_count % RaytracerConfig::PERFORMANCE_STATS_INTERVAL == 0 {
            let elapsed = self.performance.start_time.elapsed().as_secs_f32();
            let fps = self.performance.frame_count as f32 / elapsed;
            println!("FPS: {:.1}, Last compute: {:.2}ms", 
                     fps, 
                     self.performance.last_compute_time.as_secs_f32() * RaytracerConfig::MILLISECONDS_PER_SECOND);
        }
        
        let output = self.render.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .render.device
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

            render_pass.set_pipeline(&self.render.render_pipeline);
            render_pass.set_bind_group(0, &self.render.render_bind_group, &[]);
            render_pass.draw(0..3, 0..1); // Draw fullscreen triangle
        }

        self.render.queue.submit(std::iter::once(encoder.finish()));
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
                            state.progressive.needs_recompute = true;
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
                        VirtualKeyCode::L => {
                            // Try to load a glTF file (example usage)
                            if let Err(e) = state.scene.replace_with_gltf("model.gltf") {
                                println!("Failed to load glTF file: {:?}", e);
                            } else {
                                state.progressive.needs_recompute = true;
                                state.buffers.mark_spheres_dirty();
                                state.buffers.mark_triangles_dirty();
                                state.buffers.mark_materials_dirty();
                                state.buffers.mark_lights_dirty();
                                state.buffers.mark_textures_dirty();
                                state.buffers.mark_texture_data_dirty();
                            }
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
                    state.input.mouse_pressed = *button_state == ElementState::Pressed;
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if state.input.mouse_pressed {
                        if let Some(last_pos) = state.input.last_mouse_pos {
                            let delta_x = position.x - last_pos.0;
                            let delta_y = position.y - last_pos.1;
                            state.rotate_camera(delta_x, delta_y);
                        }
                    }
                    state.input.last_mouse_pos = Some((position.x, position.y));
                }
                _ => {}
            },
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                state.run_compute();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.render.size),
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
