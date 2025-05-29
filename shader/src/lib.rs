#![no_std]

use spirv_std::spirv;
use spirv_std::glam::{vec2, vec4, Vec2, Vec3, Vec4, UVec2};
use spirv_std::num_traits::Float;

#[spirv(compute(threads(16, 16)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: spirv_std::glam::UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] output_image: &spirv_std::image::Image!(2D, format=rgba8, write),
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] _spheres: &[Sphere],
    #[spirv(push_constant)] push_constants: &PushConstants,
) {
    let width = push_constants.resolution.x as u32;
    let height = push_constants.resolution.y as u32;

    if id.x >= width || id.y >= height {
        return;
    }
    
    // Convert screen coordinates to camera ray
    let uv = vec2(
        (id.x as f32 + 0.5) / width as f32,
        (id.y as f32 + 0.5) / height as f32
    );
    
    // For now, still output a gradient but incorporate sphere count
    let sphere_factor = if push_constants.sphere_count > 0 {
        push_constants.sphere_count as f32 / 10.0
    } else {
        1.0
    };
    
    let color = vec4(
        uv.x * sphere_factor,
        uv.y, 
        0.5 + 0.3 * Float::sin(push_constants.time * 0.5),
        1.0
    );

    unsafe {
        output_image.write(UVec2::new(id.x, id.y), color);
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Camera {
    pub position: Vec3,
    pub direction: Vec3,
    pub up: Vec3,
    pub fov: f32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub color: Vec3,
    pub material: u32, // 0=diffuse, 1=metal, 2=glass
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct PushConstants {
    pub resolution: Vec2,
    pub time: f32,
    pub camera: Camera,
    pub sphere_count: u32,
}

// Vertex shader for fullscreen quad
#[spirv(vertex)]
pub fn main_vs(
    #[spirv(vertex_index)] vertex_index: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    uv: &mut Vec2,
) {
    // Generate fullscreen triangle that covers entire screen
    let x = if vertex_index == 0 { -1.0 } else if vertex_index == 1 { 3.0 } else { -1.0 };
    let y = if vertex_index == 0 { -1.0 } else if vertex_index == 1 { -1.0 } else { 3.0 };
    
    *out_pos = vec4(x, y, 0.0, 1.0);
    *uv = vec2((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
}

// Fragment shader to display raytraced texture
#[spirv(fragment)]
pub fn main_fs(
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] texture: &spirv_std::image::Image!(2D, type=f32, sampled),
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &spirv_std::Sampler,
    output: &mut Vec4,
) {
    *output = texture.sample(*sampler, uv);
}