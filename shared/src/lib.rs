#![no_std]


use bytemuck::{Pod, Zeroable};

/// Camera configuration for raytracing
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Camera {
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
}

/// Sphere primitive for raytracing
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub material: u32, // 0=diffuse, 1=metal, 2=glass
}

/// Push constants for compute shader
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct PushConstants {
    pub resolution: [f32; 2],
    pub time: f32,
    pub camera: Camera,
    pub sphere_count: u32,
    pub tile_offset: [u32; 2],
    pub tile_size: [u32; 2],
    pub total_tiles: [u32; 2],
    pub current_tile_index: u32,
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

impl Sphere {
    /// Create a new sphere
    pub fn new(center: [f32; 3], radius: f32, color: [f32; 3], material: u32) -> Self {
        Self {
            center,
            radius,
            color,
            material,
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
            tile_offset,
            tile_size,
            total_tiles,
            current_tile_index,
        }
    }
}