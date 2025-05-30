#![no_std]

use spirv_std::spirv;
use spirv_std::glam::{vec2, vec3, vec4, Vec2, Vec4, UVec2};
use spirv_std::num_traits::Float;
use raytracer_shared::{Sphere, Triangle, PushConstants, RaytracerConfig};

#[spirv(compute(threads(16, 16)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: spirv_std::glam::UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] output_image: &spirv_std::image::Image!(2D, format=rgba8, write),
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] spheres: &[Sphere],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] triangles: &[Triangle],
    #[spirv(push_constant)] push_constants: &PushConstants,
) {
    // Calculate global pixel coordinates from tile offset and local thread id
    let pixel_x = push_constants.tile_offset[0] + id.x;
    let pixel_y = push_constants.tile_offset[1] + id.y;
    
    let width = push_constants.resolution[0] as u32;
    let height = push_constants.resolution[1] as u32;
    
    // Check if we're outside the tile bounds or image bounds
    if id.x >= push_constants.tile_size[0] || 
       id.y >= push_constants.tile_size[1] ||
       pixel_x >= width || 
       pixel_y >= height {
        return;
    }
    
    // Convert screen coordinates to camera ray
    let uv = vec2(
        (pixel_x as f32 + 0.5) / width as f32,
        (pixel_y as f32 + 0.5) / height as f32
    );
    
    // Convert UV coordinates to camera space
    let aspect_ratio = width as f32 / height as f32;
    let fov_scale = (push_constants.camera.fov * 0.5 * core::f32::consts::PI / 180.0).tan();
    
    let camera_x = (uv.x * 2.0 - 1.0) * aspect_ratio * fov_scale;
    let camera_y = (1.0 - uv.y * 2.0) * fov_scale;
    
    // Calculate camera right and up vectors
    let forward = vec3(push_constants.camera.direction[0], push_constants.camera.direction[1], push_constants.camera.direction[2]);
    let up = vec3(push_constants.camera.up[0], push_constants.camera.up[1], push_constants.camera.up[2]);
    let right = forward.cross(up);
    let true_up = right.cross(forward);
    
    // Calculate ray direction
    let ray_direction = forward + right * camera_x + true_up * camera_y;
    let ray_direction_normalized = ray_direction.normalize();
    
    // Raytracing - find closest intersection (spheres and triangles)
    let mut color = vec3(0.2, 0.2, 0.3); // Sky color
    let mut closest_t = f32::INFINITY;
    let camera_pos = vec3(push_constants.camera.position[0], push_constants.camera.position[1], push_constants.camera.position[2]);
    
    // Test sphere intersections
    for i in 0..push_constants.sphere_count {
        if i >= spheres.len() as u32 {
            break;
        }
        
        let sphere = spheres[i as usize];
        let sphere_center = vec3(sphere.center[0], sphere.center[1], sphere.center[2]);
        let oc = camera_pos - sphere_center;
        
        let a = ray_direction_normalized.dot(ray_direction_normalized);
        let b = 2.0 * oc.dot(ray_direction_normalized);
        let c = oc.dot(oc) - sphere.radius * sphere.radius;
        
        let discriminant = b * b - 4.0 * a * c;
        
        if discriminant >= 0.0 {
            let sqrt_discriminant = discriminant.sqrt();
            let t1 = (-b - sqrt_discriminant) / (2.0 * a);
            let t2 = (-b + sqrt_discriminant) / (2.0 * a);
            
            let t = if t1 > RaytracerConfig::MIN_RAY_DISTANCE { t1 } else { t2 };
            
            if t > RaytracerConfig::MIN_RAY_DISTANCE && t < closest_t {
                closest_t = t;
                
                // Calculate simple lighting
                let hit_point = camera_pos + ray_direction_normalized * t;
                let normal = (hit_point - sphere_center).normalize();
                let light_dir = vec3(1.0, 1.0, 1.0).normalize();
                let light_intensity = normal.dot(light_dir).max(0.0);
                
                let sphere_color = vec3(sphere.color[0], sphere.color[1], sphere.color[2]);
                color = sphere_color * (0.2 + 0.8 * light_intensity);
            }
        }
    }
    
    // Test triangle intersections using Möller-Trumbore algorithm
    for i in 0..push_constants.triangle_count {
        if i >= triangles.len() as u32 {
            break;
        }
        
        let triangle = triangles[i as usize];
        let v0 = vec3(triangle.v0[0], triangle.v0[1], triangle.v0[2]);
        let v1 = vec3(triangle.v1[0], triangle.v1[1], triangle.v1[2]);
        let v2 = vec3(triangle.v2[0], triangle.v2[1], triangle.v2[2]);
        
        // Möller-Trumbore ray-triangle intersection
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let h = ray_direction_normalized.cross(edge2);
        let a = edge1.dot(h);
        
        // If a is too small, ray is parallel to triangle
        if a.abs() < RaytracerConfig::MIN_RAY_DISTANCE {
            continue;
        }
        
        let f = 1.0 / a;
        let s = camera_pos - v0;
        let u = f * s.dot(h);
        
        if u < 0.0 || u > 1.0 {
            continue;
        }
        
        let q = s.cross(edge1);
        let v = f * ray_direction_normalized.dot(q);
        
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        
        let t = f * edge2.dot(q);
        
        if t > RaytracerConfig::MIN_RAY_DISTANCE && t < closest_t {
            closest_t = t;
            
            // Calculate triangle normal and lighting
            let normal = edge1.cross(edge2).normalize();
            let light_dir = vec3(1.0, 1.0, 1.0).normalize();
            let light_intensity = normal.dot(light_dir).max(0.0);
            
            let triangle_color = vec3(triangle.color[0], triangle.color[1], triangle.color[2]);
            color = triangle_color * (0.2 + 0.8 * light_intensity);
        }
    }
    
    let final_color = vec4(color.x, color.y, color.z, 1.0);
    
    unsafe {
        output_image.write(UVec2::new(pixel_x, pixel_y), final_color);
    }
}

// Custom functions removed - using glam built-ins for better performance


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