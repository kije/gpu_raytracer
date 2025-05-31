use bytemuck;
use raytracer_shared::{Sphere, Triangle, Material, Light, TextureInfo, RaytracerConfig};

/// Smart buffer management for geometry data
pub struct BufferManager {
    pub spheres_buffer: wgpu::Buffer,
    pub triangle_buffers: Vec<wgpu::Buffer>,
    pub materials_buffer: wgpu::Buffer,
    pub lights_buffer: wgpu::Buffer,
    pub textures_buffer: wgpu::Buffer,
    pub texture_data_buffer: wgpu::Buffer,
    pub spheres_capacity: usize,
    pub triangles_per_buffer: usize,
    pub total_triangles_capacity: usize,
    pub materials_capacity: usize,
    pub lights_capacity: usize,
    pub textures_capacity: usize,
    pub texture_data_capacity: usize,
    spheres_dirty: bool,
    triangles_dirty: bool,
    materials_dirty: bool,
    lights_dirty: bool,
    textures_dirty: bool,
    texture_data_dirty: bool,
}

impl BufferManager {
    /// Calculate maximum triangles per buffer based on WGPU buffer size limit
    fn max_triangles_per_buffer() -> usize {
        const MAX_BUFFER_SIZE: usize = 128 * 1024 * 1024; // 128MB limit
        const TRIANGLE_SIZE: usize = std::mem::size_of::<Triangle>();
        MAX_BUFFER_SIZE / TRIANGLE_SIZE
    }
    
    /// Get triangle buffer by index, or the first buffer if index is out of bounds
    pub fn get_triangle_buffer(&self, index: usize) -> &wgpu::Buffer {
        self.triangle_buffers.get(index).unwrap_or(&self.triangle_buffers[0])
    }

    pub fn new(device: &wgpu::Device) -> Self {
        let spheres_capacity = RaytracerConfig::DEFAULT_MAX_SPHERES;
        let triangles_per_buffer = Self::max_triangles_per_buffer();
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
            triangle_buffers,
            materials_buffer,
            lights_buffer,
            textures_buffer,
            texture_data_buffer,
            spheres_capacity,
            triangles_per_buffer,
            total_triangles_capacity: triangles_per_buffer,
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
    pub fn update_spheres(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, spheres: &[Sphere]) -> bool {
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
        
        // Distribute triangles across buffers if dirty or resized
        if self.triangles_dirty || needs_resize {
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

    /// Update lights buffer with new data, resizing if necessary
    pub fn update_lights(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, lights: &[Light]) -> bool {
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

    /// Mark spheres buffer as dirty (needs update)
    pub fn mark_spheres_dirty(&mut self) {
        self.spheres_dirty = true;
    }

    /// Mark triangles buffer as dirty (needs update)
    pub fn mark_triangles_dirty(&mut self) {
        self.triangles_dirty = true;
    }

    /// Mark materials buffer as dirty (needs update)
    pub fn mark_materials_dirty(&mut self) {
        self.materials_dirty = true;
    }

    /// Mark lights buffer as dirty (needs update)
    pub fn mark_lights_dirty(&mut self) {
        self.lights_dirty = true;
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
        self.spheres_dirty || self.triangles_dirty || self.materials_dirty || 
        self.lights_dirty || self.textures_dirty || self.texture_data_dirty
    }
}