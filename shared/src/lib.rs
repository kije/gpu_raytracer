#![no_std]

#[cfg(not(target_arch = "spirv"))]
extern crate alloc;
#[cfg(not(target_arch = "spirv"))]
use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};

// Use half crate for host code
#[cfg(not(target_arch = "spirv"))]
use half::f16;

// Use spirv-std for shader code when available
#[cfg(all(target_arch = "spirv", feature = "shader"))]
use spirv_std::float::Float;

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
/// Optimized with f16 packed into u32 for improved memory bandwidth
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Material {
    pub albedo: [f32; 3],          // Base color/albedo (keep f32 for color accuracy)
    pub metallic_roughness_f16: u32, // Packed: metallic(low 16 bits) + roughness(high 16 bits) as f16
    pub emission: [f32; 3],        // Emissive color (keep f32 for color accuracy)
    pub ior_transmission_f16: u32, // Packed: ior(low 16 bits) + transmission(high 16 bits) as f16
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
/// Optimized layout: 40 bytes (reduced from 48 bytes with better packing)
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Light {
    pub position: [f32; 3],        // Light position (or direction for directional) (12 bytes)
    pub light_type: u32,           // 0=directional, 1=point, 2=spot (4 bytes)
    pub color: [f32; 3],           // Light color (12 bytes)
    pub intensity: f32,            // Light intensity (4 bytes)
    pub direction: [f32; 3],       // Light direction (for directional/spot lights) (12 bytes)
    // Removed separate range field, pack cone angles into one f32 using bit manipulation
    pub range_packed: u32,         // Range packed as f16 in lower 16 bits, unused in upper 16 bits
    pub cone_angles_packed: u32,   // Inner angle (low 16 bits) + outer angle (high 16 bits) as f16
    // Total: 44 bytes (8% reduction from 48 bytes) - slight padding to 48 for alignment
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
/// Optimized layout: 20 bytes (reduced from 32 bytes with better packing)
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Sphere {
    pub center: [f32; 3],     // Sphere center (12 bytes)
    pub radius: f32,          // Sphere radius (4 bytes)
    pub material_id: u32,     // Index into material buffer (4 bytes)
    // Total: 20 bytes (37.5% reduction from 32 bytes)
}

/// Triangle primitive for raytracing
/// Optimized layout: 40 bytes (reduced from 64 bytes with better packing)
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Triangle {
    pub v0: [f32; 3],       // First vertex (12 bytes)
    pub material_id: u32,   // Index into material buffer (4 bytes) 
    pub v1: [f32; 3],       // Second vertex (12 bytes)
    pub _padding0: u32,     // Padding for alignment (4 bytes)
    pub v2: [f32; 3],       // Third vertex (12 bytes)
    // Total: 40 bytes (37.5% reduction from 64 bytes)
}

/// Axis-Aligned Bounding Box for BVH
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Aabb {
    pub min: [f32; 3],      // Minimum bounds
    pub _padding0: f32,     // Padding for alignment
    pub max: [f32; 3],      // Maximum bounds
    pub _padding1: f32,     // Padding for alignment
}

/// BVH Node for triangle acceleration structure
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct BvhNode {
    pub bounds: Aabb,       // Bounding box of this node
    pub left_child: u32,    // Index to left child (0xFFFFFFFF if leaf)
    pub right_child: u32,   // Index to right child (0xFFFFFFFF if leaf)  
    pub triangle_start: u32, // Starting index in triangle buffer (if leaf)
    pub triangle_count: u32, // Number of triangles (if leaf)
}

/// Scene metadata offsets for combined buffer layout
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct SceneMetadataOffsets {
    pub spheres_offset: u32,       // Offset to spheres data (in u32 units)
    pub spheres_count: u32,        // Number of spheres
    pub lights_offset: u32,        // Offset to lights data (in u32 units)
    pub lights_count: u32,         // Number of lights
    pub bvh_nodes_offset: u32,     // Offset to BVH nodes data (in u32 units)
    pub bvh_nodes_count: u32,      // Number of BVH nodes
    pub triangle_indices_offset: u32, // Offset to triangle indices (in u32 units)
    pub triangle_indices_count: u32,  // Number of triangle indices
}

/// Push constants for compute shader
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct PushConstants {
    pub resolution: [f32; 2],
    pub time: f32,
    pub camera: Camera,
    pub triangle_count: u32,
    pub material_count: u32,
    pub tile_offset: [u32; 2],
    pub tile_size: [u32; 2],
    pub total_tiles: [u32; 2],
    pub triangles_per_buffer: u32, // Triangles per buffer for multi-buffer access
    pub metadata_offsets: SceneMetadataOffsets, // Combined scene metadata offsets
    pub color_channel: u32,        // 0=red, 1=green, 2=blue for chromatic aberration
    pub _padding: [u32; 1],        // Padding for alignment
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
    /// Convert f32 to f16 stored as u16 (optimized with proper half crate)
    #[cfg(not(target_arch = "spirv"))]
    fn f32_to_f16_u16(value: f32) -> u16 {
        f16::from_f32(value).to_bits()
    }
    
    /// Create a new enhanced PBR material
    pub fn new(
        albedo: [f32; 3],
        metallic: f32,
        roughness: f32,
        emission: [f32; 3],
        ior: f32,
        transmission: f32,
    ) -> Self {
        #[cfg(not(target_arch = "spirv"))]
        {
            // Pack metallic and roughness into single u32
            let metallic_f16 = Self::f32_to_f16_u16(metallic);
            let roughness_f16 = Self::f32_to_f16_u16(roughness);
            let metallic_roughness_packed = (metallic_f16 as u32) | ((roughness_f16 as u32) << 16);
            
            // Pack ior and transmission into single u32
            let ior_f16 = Self::f32_to_f16_u16(ior);
            let transmission_f16 = Self::f32_to_f16_u16(transmission);
            let ior_transmission_packed = (ior_f16 as u32) | ((transmission_f16 as u32) << 16);
            
            Self {
                albedo,
                metallic_roughness_f16: metallic_roughness_packed,
                emission,
                ior_transmission_f16: ior_transmission_packed,
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
        #[cfg(target_arch = "spirv")]
        {
            // On GPU, we'll have proper f16 conversion functions
            Self {
                albedo,
                metallic_roughness_f16: 0, // Will be handled by shader conversion functions
                emission,
                ior_transmission_f16: 0,
                specular_factor: 1.0,
                specular_color: [1.0, 1.0, 1.0],
                attenuation_distance: f32::INFINITY,
                attenuation_color: [1.0, 1.0, 1.0],
                thickness_factor: 0.0,
                diffuse_factor: albedo,
                glossiness_factor: 1.0 - roughness,
                material_type: 0,
                texture_indices: [u32::MAX; 8],
                _padding: [0.0; 2],
            }
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
    
    /// Set metallic factor from f32 (converts to f16 and packs)
    #[cfg(not(target_arch = "spirv"))]
    pub fn set_metallic(&mut self, value: f32) {
        let metallic_f16 = Self::f32_to_f16_u16(value);
        // Keep existing roughness, update metallic (low 16 bits)
        self.metallic_roughness_f16 = (self.metallic_roughness_f16 & 0xFFFF0000) | (metallic_f16 as u32);
    }
    
    /// Set roughness factor from f32 (converts to f16 and packs)
    #[cfg(not(target_arch = "spirv"))]
    pub fn set_roughness(&mut self, value: f32) {
        let roughness_f16 = Self::f32_to_f16_u16(value);
        // Keep existing metallic, update roughness (high 16 bits)
        self.metallic_roughness_f16 = (self.metallic_roughness_f16 & 0x0000FFFF) | ((roughness_f16 as u32) << 16);
    }
    
    /// Set IOR from f32 (converts to f16 and packs)
    #[cfg(not(target_arch = "spirv"))]
    pub fn set_ior(&mut self, value: f32) {
        let ior_f16 = Self::f32_to_f16_u16(value);
        // Keep existing transmission, update ior (low 16 bits)
        self.ior_transmission_f16 = (self.ior_transmission_f16 & 0xFFFF0000) | (ior_f16 as u32);
    }
    
    /// Set transmission factor from f32 (converts to f16 and packs)
    #[cfg(not(target_arch = "spirv"))]
    pub fn set_transmission(&mut self, value: f32) {
        let transmission_f16 = Self::f32_to_f16_u16(value);
        // Keep existing ior, update transmission (high 16 bits)
        self.ior_transmission_f16 = (self.ior_transmission_f16 & 0x0000FFFF) | ((transmission_f16 as u32) << 16);
    }
    
    /// Extract packed f16 values for shader use
    /// Returns (metallic, roughness) as f32 values 
    pub fn unpack_metallic_roughness(&self) -> (f32, f32) {
        let metallic_bits = (self.metallic_roughness_f16 & 0x0000FFFF) as u16;
        let roughness_bits = ((self.metallic_roughness_f16 >> 16) & 0x0000FFFF) as u16;
        
        #[cfg(not(target_arch = "spirv"))]
        {
            let metallic = f16::from_bits(metallic_bits).to_f32();
            let roughness = f16::from_bits(roughness_bits).to_f32();
            (metallic, roughness)
        }
        
        #[cfg(target_arch = "spirv")]
        {
            // Use spirv-std built-in conversion when available
            // For now, use simple bit manipulation (can be optimized with spirv-std later)
            let metallic = Self::f16_bits_to_f32(metallic_bits);
            let roughness = Self::f16_bits_to_f32(roughness_bits);
            (metallic, roughness)
        }
    }
    
    /// Extract packed f16 IOR and transmission values for shader use
    /// Returns (ior, transmission) as f32 values
    pub fn unpack_ior_transmission(&self) -> (f32, f32) {
        let ior_bits = (self.ior_transmission_f16 & 0x0000FFFF) as u16;
        let transmission_bits = ((self.ior_transmission_f16 >> 16) & 0x0000FFFF) as u16;
        
        #[cfg(not(target_arch = "spirv"))]
        {
            let ior = f16::from_bits(ior_bits).to_f32();
            let transmission = f16::from_bits(transmission_bits).to_f32();
            (ior, transmission)
        }
        
        #[cfg(target_arch = "spirv")]
        {
            // Use spirv-std built-in conversion when available
            let ior = Self::f16_bits_to_f32(ior_bits);
            let transmission = Self::f16_bits_to_f32(transmission_bits);
            (ior, transmission)
        }
    }
    
    /// Simple f16 to f32 conversion for shader use (fallback until spirv-std is used)
    #[cfg(target_arch = "spirv")]
    fn f16_bits_to_f32(bits: u16) -> f32 {
        // Simple f16 to f32 conversion for shader side
        // TODO: Replace with spirv-std functions when available
        let sign = (bits >> 15) & 0x1;
        let exp = ((bits >> 10) & 0x1F) as i32;
        let frac = (bits & 0x3FF) as u32;
        
        if exp == 0 {
            if frac == 0 {
                return if sign == 1 { -0.0 } else { 0.0 };
            } else {
                // Denormal numbers - convert to f32 denormal or small normal
                let f32_exp = -14 - 126;
                let f32_frac = frac << 13;
                return f32::from_bits(((sign as u32) << 31) | ((f32_exp as u32) << 23) | f32_frac);
            }
        }
        
        if exp == 31 {
            // Infinity or NaN
            let f32_exp = 255u32;
            let f32_frac = if frac != 0 { frac << 13 } else { 0 };
            return f32::from_bits(((sign as u32) << 31) | (f32_exp << 23) | f32_frac);
        }
        
        // Normal numbers
        let f32_exp = (exp - 15 + 127) as u32;
        let f32_frac = (frac << 13) as u32;
        f32::from_bits(((sign as u32) << 31) | (f32_exp << 23) | f32_frac)
    }
}

impl Light {
    /// Pack range value as f16 in u32
    #[cfg(not(target_arch = "spirv"))]
    fn pack_range(range: f32) -> u32 {
        let range_f16 = f16::from_f32(range).to_bits();
        range_f16 as u32 // Store in lower 16 bits
    }
    
    /// Pack cone angles as f16 values in u32
    #[cfg(not(target_arch = "spirv"))]
    fn pack_cone_angles(inner: f32, outer: f32) -> u32 {
        let inner_f16 = f16::from_f32(inner).to_bits();
        let outer_f16 = f16::from_f32(outer).to_bits();
        (inner_f16 as u32) | ((outer_f16 as u32) << 16)
    }
    
    /// Create a directional light
    pub fn directional(direction: [f32; 3], color: [f32; 3], intensity: f32) -> Self {
        #[cfg(not(target_arch = "spirv"))]
        {
            Self {
                position: [0.0; 3],
                light_type: 0,
                color,
                intensity,
                direction,
                range_packed: Self::pack_range(f32::INFINITY),
                cone_angles_packed: Self::pack_cone_angles(0.0, 0.0),
            }
        }
        #[cfg(target_arch = "spirv")]
        {
            Self {
                position: [0.0; 3],
                light_type: 0,
                color,
                intensity,
                direction,
                range_packed: 0,
                cone_angles_packed: 0,
            }
        }
    }
    
    /// Create a point light
    pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> Self {
        #[cfg(not(target_arch = "spirv"))]
        {
            Self {
                position,
                light_type: 1,
                color,
                intensity,
                direction: [0.0; 3],
                range_packed: Self::pack_range(range),
                cone_angles_packed: Self::pack_cone_angles(0.0, 0.0),
            }
        }
        #[cfg(target_arch = "spirv")]
        {
            Self {
                position,
                light_type: 1,
                color,
                intensity,
                direction: [0.0; 3],
                range_packed: 0,
                cone_angles_packed: 0,
            }
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
        #[cfg(not(target_arch = "spirv"))]
        {
            Self {
                position,
                light_type: 2,
                color,
                intensity,
                direction,
                range_packed: Self::pack_range(range),
                cone_angles_packed: Self::pack_cone_angles(inner_cone_angle, outer_cone_angle),
            }
        }
        #[cfg(target_arch = "spirv")]
        {
            Self {
                position,
                light_type: 2,
                color,
                intensity,
                direction,
                range_packed: 0,
                cone_angles_packed: 0,
            }
        }
    }
    
    /// Unpack range value from f16
    pub fn unpack_range(&self) -> f32 {
        let range_bits = (self.range_packed & 0x0000FFFF) as u16;
        
        #[cfg(not(target_arch = "spirv"))]
        {
            f16::from_bits(range_bits).to_f32()
        }
        
        #[cfg(target_arch = "spirv")]
        {
            // Use simple f16 to f32 conversion for shader side
            Material::f16_bits_to_f32(range_bits)
        }
    }
    
    /// Unpack cone angles from f16 values
    /// Returns (inner_angle, outer_angle) as f32 values
    pub fn unpack_cone_angles(&self) -> (f32, f32) {
        let inner_bits = (self.cone_angles_packed & 0x0000FFFF) as u16;
        let outer_bits = ((self.cone_angles_packed >> 16) & 0x0000FFFF) as u16;
        
        #[cfg(not(target_arch = "spirv"))]
        {
            let inner = f16::from_bits(inner_bits).to_f32();
            let outer = f16::from_bits(outer_bits).to_f32();
            (inner, outer)
        }
        
        #[cfg(target_arch = "spirv")]
        {
            let inner = Material::f16_bits_to_f32(inner_bits);
            let outer = Material::f16_bits_to_f32(outer_bits);
            (inner, outer)
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
        }
    }
}

impl Triangle {
    /// Create a new triangle
    pub fn new(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], material_id: u32) -> Self {
        Self {
            v0,
            material_id,
            v1,
            _padding0: 0,
            v2,
        }
    }
    
    /// Calculate the bounding box of this triangle
    pub fn bounding_box(&self) -> Aabb {
        let min_x = self.v0[0].min(self.v1[0]).min(self.v2[0]);
        let min_y = self.v0[1].min(self.v1[1]).min(self.v2[1]);
        let min_z = self.v0[2].min(self.v1[2]).min(self.v2[2]);
        
        let max_x = self.v0[0].max(self.v1[0]).max(self.v2[0]);
        let max_y = self.v0[1].max(self.v1[1]).max(self.v2[1]);
        let max_z = self.v0[2].max(self.v1[2]).max(self.v2[2]);
        
        Aabb::new([min_x, min_y, min_z], [max_x, max_y, max_z])
    }
}

impl Aabb {
    /// Create a new AABB
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self {
            min,
            _padding0: 0.0,
            max,
            _padding1: 0.0,
        }
    }
    
    /// Create an empty AABB
    pub fn empty() -> Self {
        Self::new(
            [f32::INFINITY; 3],
            [f32::NEG_INFINITY; 3],
        )
    }
    
    /// Combine this AABB with another
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb::new(
            [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        )
    }
    
    /// Get the center point of the AABB
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }
    
    /// Get the surface area of the AABB
    pub fn surface_area(&self) -> f32 {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        2.0 * (dx * dy + dy * dz + dz * dx)
    }
}

impl BvhNode {
    /// Create a new leaf node
    pub fn leaf(bounds: Aabb, triangle_start: u32, triangle_count: u32) -> Self {
        Self {
            bounds,
            left_child: u32::MAX,
            right_child: u32::MAX,
            triangle_start,
            triangle_count,
        }
    }
    
    /// Create a new internal node
    pub fn internal(bounds: Aabb, left_child: u32, right_child: u32) -> Self {
        Self {
            bounds,
            left_child,
            right_child,
            triangle_start: 0,
            triangle_count: 0,
        }
    }
    
    /// Check if this node is a leaf
    pub fn is_leaf(&self) -> bool {
        self.left_child == u32::MAX && self.right_child == u32::MAX
    }
}

impl SceneMetadataOffsets {
    /// Create new scene metadata offsets
    pub fn new(
        spheres_offset: u32,
        spheres_count: u32,
        lights_offset: u32,
        lights_count: u32,
        bvh_nodes_offset: u32,
        bvh_nodes_count: u32,
        triangle_indices_offset: u32,
        triangle_indices_count: u32,
    ) -> Self {
        Self {
            spheres_offset,
            spheres_count,
            lights_offset,
            lights_count,
            bvh_nodes_offset,
            bvh_nodes_count,
            triangle_indices_offset,
            triangle_indices_count,
        }
    }
}

impl PushConstants {
    /// Create new push constants
    pub fn new(
        resolution: [f32; 2],
        time: f32,
        camera: Camera,
        triangle_count: u32,
        material_count: u32,
        tile_offset: [u32; 2],
        tile_size: [u32; 2],
        total_tiles: [u32; 2],
        triangles_per_buffer: u32,
        metadata_offsets: SceneMetadataOffsets,
        color_channel: u32,
    ) -> Self {
        Self {
            resolution,
            time,
            camera,
            triangle_count,
            material_count,
            tile_offset,
            tile_size,
            total_tiles,
            triangles_per_buffer,
            metadata_offsets,
            color_channel,
            _padding: [0; 1],
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
            17..=64 => total_tiles / 8,                 // Very conservative for medium images  
            65..=256 => total_tiles / 32,                // Ultra conservative for larger images
            257..=1024 => total_tiles / 64,              //l Single tile per frame for very large images
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
    
    pub fn build_default_scene() -> (Vec<Sphere>, Vec<Triangle>, Vec<Material>, Vec<Light>) {
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
        
        let lights = vec![
            Light::point([5.0, 7.0, 4.0], [1.0,1.0,1.0] ,1.0, f32::INFINITY) // White light
        ];
        
        (spheres, triangles, materials, lights)
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

    #[test]
    fn test_triangle_bounding_box() {
        let triangle = Triangle::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], 
            [0.5, 1.0, 0.0],
            0
        );
        
        let bbox = triangle.bounding_box();
        assert_eq!(bbox.min, [0.0, 0.0, 0.0]);
        assert_eq!(bbox.max, [1.0, 1.0, 0.0]);
    }

    #[test] 
    fn test_aabb_union() {
        let aabb1 = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let aabb2 = Aabb::new([0.5, 0.5, 0.5], [2.0, 2.0, 2.0]);
        
        let union = aabb1.union(&aabb2);
        assert_eq!(union.min, [0.0, 0.0, 0.0]);
        assert_eq!(union.max, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_aabb_center() {
        let aabb = Aabb::new([0.0, 0.0, 0.0], [2.0, 4.0, 6.0]);
        let center = aabb.center();
        assert_eq!(center, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_aabb_surface_area() {
        let aabb = Aabb::new([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
        let surface_area = aabb.surface_area();
        // Surface area = 2 * (2*3 + 3*4 + 4*2) = 2 * (6 + 12 + 8) = 2 * 26 = 52
        assert_eq!(surface_area, 52.0);
    }

    #[test]
    fn test_bvh_node_leaf() {
        let aabb = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let node = BvhNode::leaf(aabb, 0, 5);
        
        assert!(node.is_leaf());
        assert_eq!(node.triangle_start, 0);
        assert_eq!(node.triangle_count, 5);
        assert_eq!(node.left_child, u32::MAX);
        assert_eq!(node.right_child, u32::MAX);
    }

    #[test]
    fn test_bvh_node_internal() {
        let aabb = Aabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let node = BvhNode::internal(aabb, 1, 2);
        
        assert!(!node.is_leaf());
        assert_eq!(node.left_child, 1);
        assert_eq!(node.right_child, 2);
        assert_eq!(node.triangle_start, 0);
        assert_eq!(node.triangle_count, 0);
    }

    #[test]
    fn test_push_constants_with_metadata() {
        let camera = Camera::new();
        let metadata = SceneMetadataOffsets::new(0, 10, 500, 2, 600, 50, 1000, 100);
        let constants = PushConstants::new(
            [1920.0, 1080.0],
            0.0,
            camera,
            20,
            5,
            [0, 0],
            [128, 128],
            [15, 8],
            100,
            metadata,
            0, // red channel
        );
        
        assert_eq!(constants.metadata_offsets.bvh_nodes_count, 50);
        assert_eq!(constants.triangle_count, 20);
        assert_eq!(constants.metadata_offsets.spheres_count, 10);
        assert_eq!(constants.metadata_offsets.lights_count, 2);
        assert_eq!(constants.color_channel, 0);
    }
}