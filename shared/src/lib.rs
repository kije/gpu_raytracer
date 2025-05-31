#![no_std]

#[cfg(not(target_arch = "spirv"))]
extern crate alloc;
#[cfg(not(target_arch = "spirv"))]
use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};

/// Configuration constants for the raytracer
pub struct RaytracerConfig;

impl RaytracerConfig {
    pub const TILE_SIZE: u32 = 128;
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

/// Enhanced PBR Material properties for raytracing
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Material {
    pub albedo: [f32; 3],          // Base color/albedo
    pub metallic: f32,             // Metallic factor (0.0 = dielectric, 1.0 = metallic)
    pub roughness: f32,            // Surface roughness (0.0 = mirror, 1.0 = rough)
    pub emission: [f32; 3],        // Emissive color
    pub ior: f32,                  // Index of refraction for dielectrics
    pub transmission: f32,         // Transmission factor (0.0 = opaque, 1.0 = transparent)
    pub specular_factor: f32,      // KHR_materials_specular
    pub specular_color: [f32; 3],  // KHR_materials_specular
    pub attenuation_distance: f32, // KHR_materials_volume
    pub attenuation_color: [f32; 3], // KHR_materials_volume
    pub thickness_factor: f32,     // KHR_materials_volume
    pub diffuse_factor: [f32; 3],  // KHR_materials_pbrSpecularGlossiness
    pub glossiness_factor: f32,    // KHR_materials_pbrSpecularGlossiness
    pub material_type: u32,        // 0=metallic-roughness, 1=specular-glossiness
    pub texture_indices: [u32; 8], // Texture indices for various maps
    pub _padding: [f32; 2],        // Padding for alignment
}

/// Light source for raytracing
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Light {
    pub position: [f32; 3],        // Light position (or direction for directional)
    pub light_type: u32,           // 0=directional, 1=point, 2=spot
    pub color: [f32; 3],           // Light color
    pub intensity: f32,            // Light intensity
    pub direction: [f32; 3],       // Light direction (for directional/spot lights)
    pub range: f32,                // Light range (for point/spot lights)
    pub inner_cone_angle: f32,     // Inner cone angle (for spot lights)
    pub outer_cone_angle: f32,     // Outer cone angle (for spot lights)
    pub _padding: [f32; 2],        // Padding for alignment
}

/// Texture information
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct TextureInfo {
    pub width: u32,
    pub height: u32,
    pub format: u32,               // 0=R8, 1=RG8, 2=RGB8, 3=RGBA8, 4=R32F, etc.
    pub mip_levels: u32,
    pub offset: u32,               // Offset in texture buffer
    pub size: u32,                 // Size in bytes
    pub _padding: [u32; 2],        // Padding for alignment
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
    pub lights_count: u32,
    pub tile_offset: [u32; 2],
    pub tile_size: [u32; 2],
    pub total_tiles: [u32; 2],
    pub triangles_per_buffer: u32, // Triangles per buffer for multi-buffer access
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
    /// Create a new enhanced PBR material
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
            specular_factor: 1.0,
            specular_color: [1.0, 1.0, 1.0],
            attenuation_distance: f32::INFINITY,
            attenuation_color: [1.0, 1.0, 1.0],
            thickness_factor: 0.0,
            diffuse_factor: albedo,
            glossiness_factor: 1.0 - roughness,
            material_type: 0, // metallic-roughness workflow
            texture_indices: [u32::MAX; 8], // No textures by default
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
    
    /// Create a specular-glossiness material (KHR_materials_pbrSpecularGlossiness)
    pub fn specular_glossiness(
        diffuse: [f32; 3],
        specular: [f32; 3], 
        glossiness: f32
    ) -> Self {
        let mut material = Self::new(diffuse, 0.0, 1.0 - glossiness, [0.0; 3], 1.5, 0.0);
        material.material_type = 1; // specular-glossiness workflow
        material.diffuse_factor = diffuse;
        material.specular_color = specular;
        material.glossiness_factor = glossiness;
        material
    }
    
    /// Set volume properties (KHR_materials_volume)
    pub fn with_volume(mut self, thickness: f32, attenuation_distance: f32, attenuation_color: [f32; 3]) -> Self {
        self.thickness_factor = thickness;
        self.attenuation_distance = attenuation_distance;
        self.attenuation_color = attenuation_color;
        self
    }
    
    /// Set specular properties (KHR_materials_specular)
    pub fn with_specular(mut self, factor: f32, color: [f32; 3]) -> Self {
        self.specular_factor = factor;
        self.specular_color = color;
        self
    }
    
    /// Set texture indices
    pub fn with_textures(mut self, textures: [u32; 8]) -> Self {
        self.texture_indices = textures;
        self
    }
}

impl Light {
    /// Create a directional light
    pub fn directional(direction: [f32; 3], color: [f32; 3], intensity: f32) -> Self {
        Self {
            position: [0.0; 3],
            light_type: 0,
            color,
            intensity,
            direction,
            range: f32::INFINITY,
            inner_cone_angle: 0.0,
            outer_cone_angle: 0.0,
            _padding: [0.0; 2],
        }
    }
    
    /// Create a point light
    pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> Self {
        Self {
            position,
            light_type: 1,
            color,
            intensity,
            direction: [0.0; 3],
            range,
            inner_cone_angle: 0.0,
            outer_cone_angle: 0.0,
            _padding: [0.0; 2],
        }
    }
    
    /// Create a spot light
    pub fn spot(
        position: [f32; 3], 
        direction: [f32; 3], 
        color: [f32; 3], 
        intensity: f32, 
        range: f32,
        inner_cone_angle: f32,
        outer_cone_angle: f32
    ) -> Self {
        Self {
            position,
            light_type: 2,
            color,
            intensity,
            direction,
            range,
            inner_cone_angle,
            outer_cone_angle,
            _padding: [0.0; 2],
        }
    }
}

impl TextureInfo {
    /// Create new texture info
    pub fn new(width: u32, height: u32, format: u32, offset: u32, size: u32) -> Self {
        Self {
            width,
            height,
            format,
            mip_levels: 1,
            offset,
            size,
            _padding: [0; 2],
        }
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
        lights_count: u32,
        tile_offset: [u32; 2],
        tile_size: [u32; 2],
        total_tiles: [u32; 2],
        triangles_per_buffer: u32,
    ) -> Self {
        Self {
            resolution,
            time,
            camera,
            sphere_count,
            triangle_count,
            material_count,
            lights_count,
            tile_offset,
            tile_size,
            total_tiles,
            triangles_per_buffer,
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
    /// Made much more conservative to prevent system hangs with large triangle counts
    pub fn calculate_tiles_per_frame(total_tiles: u32) -> u32 {
        match total_tiles {
            0..=16 => total_tiles,        // Render all at once for small images
            17..=64 => 8,                 // Very conservative for medium images  
            65..=256 => 2,                // Ultra conservative for larger images
            257..=1024 => 1,              // Single tile per frame for very large images
            _ => 1,                       // Absolute minimum for huge images
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

#[macro_export]
macro_rules! branchless_float_if {
    ($condition:expr, $if_true:expr, $if_false:expr) => {
        branchless_float_if!($condition, $if_true, $if_false, (f32::MAX), &(f32::MAX - 1.0), lt)
    };
    ($condition:expr, $if_true:expr, $if_false:expr, $max_val: expr, $max_minus_one_val: expr, $cmp_lt_method: ident) => {
        {{
            // if $if_true is nan => min will return $if_true+($if_false * 1.0_f32.copysign($if_false)
            let actual_if_true = ($if_true).min($max_val); // min returns the smaller of the two, but if eitehr is nan, the other is returned -> result here will be either valid if_true or max
            let actual_if_false = ($if_false).min($max_val);

            let true_contrib = branchless_float_if!(@nonnan; actual_if_true.$cmp_lt_method($max_minus_one_val), actual_if_true, actual_if_false);
            let false_contrib = branchless_float_if!(@nonnan; actual_if_false.$cmp_lt_method($max_minus_one_val), actual_if_false, actual_if_true);

           let res = branchless_float_if!(@nonnan; $condition, true_contrib, false_contrib);

            // (res, res_is_valid)
            (res, res.$cmp_lt_method($max_minus_one_val))
        }}
    };
    (@nonnan; $condition:expr, $if_true:expr, $if_false:expr) => {
        ((($condition) as u32) as f32) * ($if_true) + ((($condition) as u32) ^ 1) as f32 * ($if_false)
    }
}

#[macro_export]
macro_rules! branchless_u32_if {
    ($condition:expr, $if_true:expr, $if_false:expr) => {
        {{
            use ::core::ops::{BitXor, BitAnd};
            ($if_true).bitxor(($if_true).bitxor($if_false).bitand((0u32 + (1u32 * ($condition as u32))).wrapping_sub(1)))
        }}
    };
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn branchless_float_if_trivial_non_nan() {
        assert_eq!(branchless_float_if!(@nonnan; true, 0.5f32, -1.0), 0.5);
        assert_eq!(branchless_float_if!(@nonnan; false, 0.5f32, -1.0), -1.0);

        assert_eq!(branchless_float_if!(@nonnan; true, 2.5f32, -3000.0), 2.5);
        assert_eq!(branchless_float_if!(@nonnan; false, 2.5f32, -3000.0), -3000.0);
    }

    #[test]
    fn branchless_float_if_trivial() {
        assert_eq!(branchless_float_if!(true, 0.5f32, -1.0f32), (0.5, true));
        assert_eq!(branchless_float_if!(false, 0.5f32, -1.0f32), (-1.0, true));

        assert_eq!(branchless_float_if!(true, -0.5f32, 1.0f32), (-0.5, true));
        assert_eq!(branchless_float_if!(false, -0.5f32, 1.0f32), (1.0,  true));
    }

    #[test]
    fn branchless_float_if_nan_values() {
        assert_eq!(branchless_float_if!(true, 0.5f32, f32::NAN), (0.5, true));
        assert_eq!(branchless_float_if!(true, -0.5f32, f32::NAN), (-0.5, true));

        assert_eq!(branchless_float_if!(false, 0.5f32, f32::NAN), (0.5, true));
        assert_eq!(branchless_float_if!(false, -0.5f32, f32::NAN), (-0.5, true));

        assert_eq!(branchless_float_if!(true, f32::NAN, 1.0f32), (1.0, true));
        assert_eq!(branchless_float_if!(true, f32::NAN, -1.0f32), (-1.0, true));

        assert_eq!(branchless_float_if!(false, f32::NAN, 1.0f32), (1.0, true));
        assert_eq!(branchless_float_if!(false, f32::NAN, -1.0f32), (-1.0, true));

        assert_eq!(branchless_float_if!(false, f32::NAN, f32::NAN), (f32::MAX, false));
    }
}