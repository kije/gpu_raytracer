#![no_std]

#[cfg(not(target_arch = "spirv"))]
extern crate alloc;
#[cfg(not(target_arch = "spirv"))]
use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};

/// Configuration constants for the raytracer
pub struct RaytracerConfig;

impl RaytracerConfig {
    pub const TILE_SIZE: u32 = 64;
    pub const THREAD_GROUP_SIZE: (u32, u32) = (16, 16);
    pub const DEFAULT_MAX_SPHERES: usize = 64;
    pub const DEFAULT_MAX_TRIANGLES: usize = 64;
    pub const CAMERA_MOVE_SPEED: f32 = 0.1;
    pub const CAMERA_ROTATE_SENSITIVITY: f32 = 0.005;
    pub const MIN_RAY_DISTANCE: f32 = 0.00001;
    
    // GPU configuration constants
    pub const MAX_PUSH_CONSTANT_SIZE: u32 = 128;
    pub const PERFORMANCE_STATS_INTERVAL: u64 = 60; // frames
    pub const CAMERA_PITCH_CLAMP: f32 = 0.99;
    pub const MILLISECONDS_PER_SECOND: f32 = 1000.0;
    pub const PROGRESS_PERCENTAGE_SCALE: f32 = 100.0;
}

/// Camera configuration for raytracing
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Camera {
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
}

/// PBR Material properties for raytracing
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Material {
    pub albedo: [f32; 3],          // Base color/albedo
    pub metallic: f32,             // Metallic factor (0.0 = dielectric, 1.0 = metallic)
    pub roughness: f32,            // Surface roughness (0.0 = mirror, 1.0 = rough)
    pub emission: [f32; 3],        // Emissive color
    pub ior: f32,                  // Index of refraction for dielectrics
    pub transmission: f32,         // Transmission factor (0.0 = opaque, 1.0 = transparent)
    pub _padding: [f32; 2],        // Padding for 16-byte alignment
}

/// Sphere primitive for raytracing
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub material_id: u32,          // Index into material buffer
    pub _padding: [f32; 3],        // Padding for alignment
}

/// Triangle primitive for raytracing
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Triangle {
    pub v0: [f32; 3],       // First vertex
    pub _padding0: f32,     // Padding for alignment
    pub v1: [f32; 3],       // Second vertex  
    pub _padding1: f32,     // Padding for alignment
    pub v2: [f32; 3],       // Third vertex
    pub _padding2: f32,     // Padding for alignment
    pub material_id: u32,   // Index into material buffer
    pub _padding3: [f32; 3], // Padding for alignment
}

/// Push constants for compute shader
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct PushConstants {
    pub resolution: [f32; 2],
    pub time: f32,
    pub camera: Camera,
    pub sphere_count: u32,
    pub triangle_count: u32,
    pub material_count: u32,
    pub tile_offset: [u32; 2],
    pub tile_size: [u32; 2],
    pub total_tiles: [u32; 2],
    pub current_tile_index: u32,
    pub _padding: u32, // Keep 16-byte alignment
}

impl Camera {
    /// Create a new camera with default parameters
    pub fn new() -> Self {
        Self {
            position: [0.0, 0.0, 5.0],
            direction: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
            fov: 45.0,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

impl Material {
    /// Create a new PBR material
    pub fn new(
        albedo: [f32; 3],
        metallic: f32,
        roughness: f32,
        emission: [f32; 3],
        ior: f32,
        transmission: f32,
    ) -> Self {
        Self {
            albedo,
            metallic,
            roughness,
            emission,
            ior,
            transmission,
            _padding: [0.0; 2],
        }
    }
    
    /// Create a diffuse material
    pub fn diffuse(albedo: [f32; 3]) -> Self {
        Self::new(albedo, 0.0, 1.0, [0.0; 3], 1.5, 0.0)
    }
    
    /// Create a metallic material
    pub fn metallic(albedo: [f32; 3], roughness: f32) -> Self {
        Self::new(albedo, 1.0, roughness, [0.0; 3], 1.5, 0.0)
    }
    
    /// Create a glass/dielectric material
    pub fn glass(albedo: [f32; 3], ior: f32, transmission: f32) -> Self {
        Self::new(albedo, 0.0, 0.0, [0.0; 3], ior, transmission)
    }
    
    /// Create an emissive material
    pub fn emissive(albedo: [f32; 3], emission: [f32; 3]) -> Self {
        Self::new(albedo, 0.0, 1.0, emission, 1.5, 0.0)
    }
}

impl Sphere {
    /// Create a new sphere
    pub fn new(center: [f32; 3], radius: f32, material_id: u32) -> Self {
        Self {
            center,
            radius,
            material_id,
            _padding: [0.0; 3],
        }
    }
}

impl Triangle {
    /// Create a new triangle
    pub fn new(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], material_id: u32) -> Self {
        Self {
            v0,
            _padding0: 0.0,
            v1,
            _padding1: 0.0,
            v2,
            _padding2: 0.0,
            material_id,
            _padding3: [0.0; 3],
        }
    }
}

impl PushConstants {
    /// Create new push constants
    pub fn new(
        resolution: [f32; 2],
        time: f32,
        camera: Camera,
        sphere_count: u32,
        triangle_count: u32,
        material_count: u32,
        tile_offset: [u32; 2],
        tile_size: [u32; 2],
        total_tiles: [u32; 2],
        current_tile_index: u32,
    ) -> Self {
        Self {
            resolution,
            time,
            camera,
            sphere_count,
            triangle_count,
            material_count,
            tile_offset,
            tile_size,
            total_tiles,
            current_tile_index,
            _padding: 0,
        }
    }
}

/// Helper functions for tile calculations
pub struct TileHelper;

impl TileHelper {
    /// Calculate number of tiles needed for given dimensions
    pub fn calculate_tile_count(width: u32, height: u32, tile_size: u32) -> (u32, u32) {
        let tiles_x = (width + tile_size - 1) / tile_size;
        let tiles_y = (height + tile_size - 1) / tile_size;
        (tiles_x, tiles_y)
    }
    
    /// Calculate adaptive tiles per frame based on total tile count
    pub fn calculate_tiles_per_frame(total_tiles: u32) -> u32 {
        match total_tiles {
            0..=16 => total_tiles,        // Render all at once for small images
            17..=64 => total_tiles / 2,   // 2 batches for medium images  
            65..=256 => 64,               // More aggressive for larger images
            257..=1024 => 128,            // Even more aggressive for very large images
            _ => 256,                     // Maximum parallelism for huge images
        }.max(1) // Ensure at least 1 tile per frame
    }
}

/// Scene management utilities
#[cfg(not(target_arch = "spirv"))]
pub struct SceneBuilder {
    spheres: Vec<Sphere>,
    triangles: Vec<Triangle>,
    materials: Vec<Material>,
}

#[cfg(not(target_arch = "spirv"))]
impl SceneBuilder {
    pub fn new() -> Self {
        Self {
            spheres: Vec::new(),
            triangles: Vec::new(),
            materials: Vec::new(),
        }
    }
    
    pub fn add_material(mut self, material: Material) -> Self {
        self.materials.push(material);
        self
    }
    
    pub fn add_sphere(mut self, center: [f32; 3], radius: f32, material_id: u32) -> Self {
        self.spheres.push(Sphere::new(center, radius, material_id));
        self
    }
    
    pub fn add_triangle(mut self, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], material_id: u32) -> Self {
        self.triangles.push(Triangle::new(v0, v1, v2, material_id));
        self
    }
    
    pub fn build_default_scene() -> (Vec<Sphere>, Vec<Triangle>, Vec<Material>) {
        use alloc::vec;
        
        // Create materials
        let materials = vec![
            Material::diffuse([0.8, 0.3, 0.3]),        // 0: Red diffuse
            Material::metallic([0.8, 0.8, 0.2], 0.1),  // 1: Yellow metal, low roughness
            Material::glass([0.2, 0.3, 0.8], 1.5, 0.9), // 2: Blue glass
            Material::emissive([1.0, 1.0, 1.0], [0.5, 0.5, 1.0]), // 3: Blue light
        ];
        
        let spheres = vec![
            Sphere::new([0.0, 0.0, -1.0], 0.5, 0),   // Red diffuse
            Sphere::new([-1.0, 0.0, -1.0], 0.5, 1),  // Yellow metal
            Sphere::new([1.0, 0.0, -1.0], 0.5, 2),   // Blue glass
            Sphere::new([2.0, 0.0, -3.0], 0.5, 2),   // Blue glass
            Sphere::new([-2.0, 0.0, -4.0], 0.5, 1),  // Yellow metal
            Sphere::new([-1.0, 2.0, -5.0], 0.5, 3),  // Blue light
        ];
        
        let triangles = vec![
            Triangle::new(
                [0.0, 1.0, -2.0],  // Top vertex
                [-0.5, 0.0, -2.0], // Bottom left
                [0.5, 0.0, -2.0],  // Bottom right
                0                   // Red diffuse material
            ),
            Triangle::new(
                [1.5, 0.5, -3.0],  // Right triangle
                [1.0, -0.5, -3.0],
                [2.0, -0.5, -3.0],
                1                    // Yellow metal material
            ),
        ];
        
        (spheres, triangles, materials)
    }
    
    pub fn build(self) -> (Vec<Sphere>, Vec<Triangle>, Vec<Material>) {
        (self.spheres, self.triangles, self.materials)
    }
}