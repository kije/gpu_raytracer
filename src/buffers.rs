use bytemuck;
use raytracer_shared::{Sphere, Triangle, Vertex, Material, Light, TextureInfo, BvhNode, RaytracerConfig, SceneMetadataOffsets};

/// Smart buffer management for geometry data with combined scene metadata buffer
pub struct BufferManager {
    // Combined scene metadata buffer (spheres, lights, BVH nodes, triangle indices)
    pub scene_metadata_buffer: wgpu::Buffer,
    pub triangle_buffers: Vec<wgpu::Buffer>,
    pub materials_buffer: wgpu::Buffer,
    pub textures_buffer: wgpu::Buffer,
    pub texture_data_buffer: wgpu::Buffer,
    
    // Buffer capacities and management
    pub scene_metadata_capacity: usize, // Combined buffer capacity in bytes
    pub triangles_per_buffer: usize,
    pub total_triangles_capacity: usize,
    pub materials_capacity: usize,
    pub textures_capacity: usize,
    pub texture_data_capacity: usize,
    
    // Individual component capacities within scene metadata buffer
    pub spheres_capacity: usize,
    pub lights_capacity: usize,
    pub bvh_nodes_capacity: usize,
    pub triangle_indices_capacity: usize,
    pub vertices_capacity: usize,
    
    // Dirty tracking
    pub scene_metadata_dirty: bool,
    pub triangles_dirty: bool,
    pub materials_dirty: bool,
    pub textures_dirty: bool,
    pub texture_data_dirty: bool,
    
    // Change tracking for incremental updates
    last_spheres_count: usize,
    last_lights_count: usize,
    last_bvh_nodes_count: usize,
    last_triangle_indices_count: usize,
    last_vertices_count: usize,
    last_triangles_count: usize,
    last_materials_count: usize,
    last_textures_count: usize,
    last_texture_data_len: usize,
}

impl BufferManager {
    /// Calculate maximum triangles per buffer based on WGPU buffer size limit
    fn max_triangles_per_buffer() -> usize {
        const MAX_BUFFER_SIZE: usize = 128 * 1024 * 1024; // 128MB limit
        const TRIANGLE_SIZE: usize = std::mem::size_of::<Triangle>();
        MAX_BUFFER_SIZE / TRIANGLE_SIZE
    }
    
    /// Get triangle buffer by index with proper bounds checking
    pub fn get_triangle_buffer(&self, index: usize) -> Option<&wgpu::Buffer> {
        self.triangle_buffers.get(index)
    }
    
    /// Get triangle buffer by index, or return an error if invalid
    pub fn get_triangle_buffer_checked(&self, index: usize) -> Result<&wgpu::Buffer, String> {
        self.triangle_buffers.get(index)
            .ok_or_else(|| format!("Triangle buffer index {} out of bounds (have {})", index, self.triangle_buffers.len()))
    }

    pub fn new(device: &wgpu::Device) -> Self {
        let spheres_capacity = RaytracerConfig::DEFAULT_MAX_SPHERES;
        // fixme add also to RaytracerConfig
        let lights_capacity = 32; // Default lights capacity
        let bvh_nodes_capacity = 256; // Default BVH nodes capacity
        let triangle_indices_capacity = 1024; // Default triangle indices capacity
        let vertices_capacity = 1024; // Default vertices capacity
        let triangles_per_buffer = Self::max_triangles_per_buffer();
        let materials_capacity = 64; // Default materials capacity
        let textures_capacity = 64; // Default textures capacity
        let texture_data_capacity = 256 * 1024; // 1MB default texture data capacity (in u32 units)

        // Calculate combined scene metadata buffer size
        let scene_metadata_capacity = spheres_capacity * std::mem::size_of::<Sphere>() +
                                     lights_capacity * std::mem::size_of::<Light>() +
                                     bvh_nodes_capacity * std::mem::size_of::<BvhNode>() +
                                     triangle_indices_capacity * std::mem::size_of::<u32>() +
                                     vertices_capacity * std::mem::size_of::<Vertex>();

        let scene_metadata_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene Metadata Buffer"),
            size: scene_metadata_capacity as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create initial triangle buffer (one buffer for small scenes)
        let triangle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Triangle Buffer 0"),
            size: (std::mem::size_of::<Triangle>() * triangles_per_buffer) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let triangle_buffers = vec![triangle_buffer];
        
        let materials_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Materials Buffer"),
            size: (std::mem::size_of::<Material>() * materials_capacity) as u64,
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
            scene_metadata_buffer,
            triangle_buffers,
            materials_buffer,
            textures_buffer,
            texture_data_buffer,
            scene_metadata_capacity,
            triangles_per_buffer,
            total_triangles_capacity: triangles_per_buffer,
            materials_capacity,
            textures_capacity,
            texture_data_capacity,
            spheres_capacity,
            lights_capacity,
            bvh_nodes_capacity,
            triangle_indices_capacity,
            vertices_capacity,
            scene_metadata_dirty: true,
            triangles_dirty: true,
            materials_dirty: true,
            textures_dirty: true,
            texture_data_dirty: true,
            last_spheres_count: 0,
            last_lights_count: 0,
            last_bvh_nodes_count: 0,
            last_triangle_indices_count: 0,
            last_vertices_count: 0,
            last_triangles_count: 0,
            last_materials_count: 0,
            last_textures_count: 0,
            last_texture_data_len: 0,
        }
    }
    
    /// Update scene metadata buffer with incremental updates for spheres, lights, BVH nodes, triangle indices, and vertices
    pub fn update_scene_metadata(
        &mut self, 
        device: &wgpu::Device, 
        queue: &wgpu::Queue, 
        spheres: &[Sphere], 
        lights: &[Light],
        bvh_nodes: &[BvhNode], 
        triangle_indices: &[u32],
        vertices: &[Vertex]
    ) -> (bool, SceneMetadataOffsets) {
        let spheres_size = spheres.len() * std::mem::size_of::<Sphere>();
        let lights_size = lights.len() * std::mem::size_of::<Light>();
        let bvh_nodes_size = bvh_nodes.len() * std::mem::size_of::<BvhNode>();
        let triangle_indices_size = triangle_indices.len() * std::mem::size_of::<u32>();
        let vertices_size = vertices.len() * std::mem::size_of::<Vertex>();
        
        let total_size = spheres_size + lights_size + bvh_nodes_size + triangle_indices_size + vertices_size;
        let needs_resize = total_size > self.scene_metadata_capacity;
        
        // Check if data actually changed for incremental updates
        let spheres_changed = spheres.len() != self.last_spheres_count;
        let lights_changed = lights.len() != self.last_lights_count;
        let bvh_nodes_changed = bvh_nodes.len() != self.last_bvh_nodes_count;
        let triangle_indices_changed = triangle_indices.len() != self.last_triangle_indices_count;
        let vertices_changed = vertices.len() != self.last_vertices_count;
        
        let any_changed = spheres_changed || lights_changed || bvh_nodes_changed || triangle_indices_changed || vertices_changed;
        
        if needs_resize {
            // Double the capacity to accommodate growth  
            self.spheres_capacity = (spheres.len() * 2).max(RaytracerConfig::DEFAULT_MAX_SPHERES);
            self.lights_capacity = (lights.len() * 2).max(32);
            self.bvh_nodes_capacity = (bvh_nodes.len() * 2).max(256);
            self.triangle_indices_capacity = (triangle_indices.len() * 2).max(1024);
            self.vertices_capacity = (vertices.len() * 2).max(1024);
            
            self.scene_metadata_capacity = self.spheres_capacity * std::mem::size_of::<Sphere>() +
                                          self.lights_capacity * std::mem::size_of::<Light>() +
                                          self.bvh_nodes_capacity * std::mem::size_of::<BvhNode>() +
                                          self.triangle_indices_capacity * std::mem::size_of::<u32>() +
                                          self.vertices_capacity * std::mem::size_of::<Vertex>();
            
            self.scene_metadata_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Scene Metadata Combined Buffer"),
                size: self.scene_metadata_capacity as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            println!("Resized scene metadata buffer: {} spheres, {} lights, {} BVH nodes, {} triangle indices, {} vertices ({:.2} MB)",
                     self.spheres_capacity, self.lights_capacity, self.bvh_nodes_capacity, self.triangle_indices_capacity, self.vertices_capacity,
                     self.scene_metadata_capacity as f64 / (1024.0 * 1024.0));
        }
        
        // Only update if dirty flag is set OR data actually changed OR buffer was resized
        if self.scene_metadata_dirty || needs_resize || any_changed {
            let mut combined_data = Vec::new();
            let mut offset_in_bytes = 0;
            
            // Track offsets in u32 units for shader access
            let _spheres_offset_u32 = offset_in_bytes / 4;
            combined_data.extend_from_slice(bytemuck::cast_slice(spheres));
            offset_in_bytes += spheres_size;
            
            let _lights_offset_u32 = offset_in_bytes / 4;
            combined_data.extend_from_slice(bytemuck::cast_slice(lights));
            offset_in_bytes += lights_size;
            
            let _bvh_nodes_offset_u32 = offset_in_bytes / 4;
            combined_data.extend_from_slice(bytemuck::cast_slice(bvh_nodes));
            offset_in_bytes += bvh_nodes_size;
            
            let _triangle_indices_offset_u32 = offset_in_bytes / 4;
            combined_data.extend_from_slice(bytemuck::cast_slice(triangle_indices));
            offset_in_bytes += triangle_indices_size;
            
            let _vertices_offset_u32 = offset_in_bytes / 4;
            combined_data.extend_from_slice(bytemuck::cast_slice(vertices));
            
            queue.write_buffer(
                &self.scene_metadata_buffer,
                0,
                &combined_data,
            );
            
            self.scene_metadata_dirty = false;
            
            // Update change tracking
            self.last_spheres_count = spheres.len();
            self.last_lights_count = lights.len();
            self.last_bvh_nodes_count = bvh_nodes.len();
            self.last_triangle_indices_count = triangle_indices.len();
            self.last_vertices_count = vertices.len();
            
            if any_changed && !needs_resize {
                println!("Incremental scene metadata update: {} spheres, {} lights, {} BVH nodes, {} triangle indices, {} vertices",
                         spheres.len(), lights.len(), bvh_nodes.len(), triangle_indices.len(), vertices.len());
            }
        }
        
        let offsets = SceneMetadataOffsets::new(
            0, // spheres always start at offset 0
            spheres.len() as u32,
            (spheres_size / 4) as u32, // lights offset in u32 units
            lights.len() as u32,
            ((spheres_size + lights_size) / 4) as u32, // BVH nodes offset in u32 units
            bvh_nodes.len() as u32,
            ((spheres_size + lights_size + bvh_nodes_size) / 4) as u32, // triangle indices offset in u32 units
            triangle_indices.len() as u32,
            ((spheres_size + lights_size + bvh_nodes_size + triangle_indices_size) / 4) as u32, // vertices offset in u32 units
            vertices.len() as u32, // vertices count
        );
        
        (needs_resize, offsets)
    }

    /// Update triangles buffers with new data, creating additional buffers if necessary
    pub fn update_triangles(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, triangles: &[Triangle]) -> bool {
        let triangles_needed = triangles.len();
        let buffers_needed = (triangles_needed + self.triangles_per_buffer - 1) / self.triangles_per_buffer;
        let buffers_needed = buffers_needed.max(1); // Always need at least one buffer
        
        let mut needs_resize = false;
        
        // Create additional triangle buffers if needed
        while self.triangle_buffers.len() < buffers_needed {
            let buffer_index = self.triangle_buffers.len();
            let triangle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Triangle Buffer {}", buffer_index)),
                size: (std::mem::size_of::<Triangle>() * self.triangles_per_buffer) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.triangle_buffers.push(triangle_buffer);
            self.total_triangles_capacity += self.triangles_per_buffer;
            needs_resize = true;
            
            println!("Created triangle buffer {}: {} triangles ({:.2} MB)", 
                     buffer_index,
                     self.triangles_per_buffer,
                     (std::mem::size_of::<Triangle>() * self.triangles_per_buffer) as f64 / (1024.0 * 1024.0));
        }
        
        if needs_resize {
            println!("Triangle buffer system: {} buffers, total capacity: {} triangles ({:.2} MB)",
                     self.triangle_buffers.len(),
                     self.total_triangles_capacity,
                     (std::mem::size_of::<Triangle>() * self.total_triangles_capacity) as f64 / (1024.0 * 1024.0));
        }
        
        // Check if data actually changed for incremental updates
        let triangles_changed = triangles.len() != self.last_triangles_count;
        
        // Distribute triangles across buffers if dirty flag is set OR data changed OR buffer was resized
        if self.triangles_dirty || needs_resize || triangles_changed {
            for (buffer_index, buffer) in self.triangle_buffers.iter().enumerate() {
                let start_triangle = buffer_index * self.triangles_per_buffer;
                let end_triangle = std::cmp::min(start_triangle + self.triangles_per_buffer, triangles.len());
                
                if start_triangle < triangles.len() {
                    let buffer_triangles = &triangles[start_triangle..end_triangle];
                    queue.write_buffer(
                        buffer,
                        0,
                        bytemuck::cast_slice(buffer_triangles),
                    );
                }
            }
            self.triangles_dirty = false;
            
            // Update change tracking
            self.last_triangles_count = triangles.len();
            
            if triangles_changed && !needs_resize {
                println!("Incremental triangles update: {} triangles", triangles.len());
            }
        }
        
        needs_resize
    }

    /// Update materials buffer with new data, resizing if necessary
    pub fn update_materials(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, materials: &[Material]) -> bool {
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
        
        // Check if data actually changed for incremental updates
        let materials_changed = materials.len() != self.last_materials_count;
        
        // Only update if dirty flag is set OR data actually changed OR buffer was resized
        if self.materials_dirty || needs_resize || materials_changed {
            queue.write_buffer(
                &self.materials_buffer,
                0,
                bytemuck::cast_slice(materials),
            );
            self.materials_dirty = false;
            
            // Update change tracking
            self.last_materials_count = materials.len();
            
            if materials_changed && !needs_resize {
                println!("Incremental materials update: {} materials", materials.len());
            }
        }
        
        needs_resize
    }


    /// Update textures buffer with new data, resizing if necessary
    pub fn update_textures(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, textures: &[TextureInfo]) -> bool {
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
        
        // Check if data actually changed for incremental updates
        let textures_changed = textures.len() != self.last_textures_count;
        
        // Only update if dirty flag is set OR data actually changed OR buffer was resized
        if self.textures_dirty || needs_resize || textures_changed {
            queue.write_buffer(
                &self.textures_buffer,
                0,
                bytemuck::cast_slice(textures),
            );
            self.textures_dirty = false;
            
            // Update change tracking
            self.last_textures_count = textures.len();
            
            if textures_changed && !needs_resize {
                println!("Incremental textures update: {} textures", textures.len());
            }
        }
        
        needs_resize
    }

    /// Update texture data buffer with new data, resizing if necessary
    pub fn update_texture_data(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, texture_data: &[u8]) -> bool {
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
        
        // Check if data actually changed for incremental updates
        let texture_data_changed = texture_data.len() != self.last_texture_data_len;
        
        // Only update if dirty flag is set OR data actually changed OR buffer was resized
        if self.texture_data_dirty || needs_resize || texture_data_changed {
            queue.write_buffer(
                &self.texture_data_buffer,
                0,
                bytemuck::cast_slice(&packed_data),
            );
            self.texture_data_dirty = false;
            
            // Update change tracking
            self.last_texture_data_len = texture_data.len();
            
            if texture_data_changed && !needs_resize {
                println!("Incremental texture data update: {} bytes", texture_data.len());
            }
        }
        
        needs_resize
    }

    /// Mark scene metadata buffer as dirty (needs update)
    pub fn mark_scene_metadata_dirty(&mut self) {
        self.scene_metadata_dirty = true;
    }

    /// Mark triangles buffer as dirty (needs update)
    pub fn mark_triangles_dirty(&mut self) {
        self.triangles_dirty = true;
    }

    /// Mark materials buffer as dirty (needs update)
    pub fn mark_materials_dirty(&mut self) {
        self.materials_dirty = true;
    }

    /// Mark textures buffer as dirty (needs update)
    pub fn mark_textures_dirty(&mut self) {
        self.textures_dirty = true;
    }

    /// Mark texture data buffer as dirty (needs update)
    pub fn mark_texture_data_dirty(&mut self) {
        self.texture_data_dirty = true;
    }

    /// Check if any buffers need updating
    pub fn needs_update(&self) -> bool {
        self.scene_metadata_dirty || self.triangles_dirty || self.materials_dirty || 
        self.textures_dirty || self.texture_data_dirty
    }
}