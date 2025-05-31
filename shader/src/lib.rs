#![no_std]

use spirv_std::spirv;
use spirv_std::glam::{vec2, vec3, vec4, Vec2, Vec4, UVec2, Vec3};
use spirv_std::num_traits::Float;
use raytracer_shared::{Sphere, Triangle, Material, Light, TextureInfo, PushConstants, RaytracerConfig, branchless_float_if, branchless_u32_if};

macro_rules! branchless_vec3_if {
    ($condition:expr, $if_true:expr, $if_false:expr) => {
        branchless_vec3_if!($condition, $if_true, $if_false, Vec3::MAX, (Vec3::MAX - Vec3::ONE), cmplt)
    };
    ($condition:expr, $if_true:expr, $if_false:expr, $max_val: expr, $max_minus_one_val: expr, $cmp_lt_method: ident) => {
        {{
            // if $if_true is nan => min will return $if_true+($if_false * 1.0_f32.copysign($if_false)
            let actual_if_true = ($if_true).min($max_val); // min returns the smaller of the two, but if eitehr is nan, the other is returned -> result here will be either valid if_true or max
            let actual_if_false = ($if_false).min($max_val);

            let true_contrib = branchless_float_if!(@nonnan; actual_if_true.$cmp_lt_method($max_minus_one_val).any(), actual_if_true, actual_if_false);
            let false_contrib = branchless_float_if!(@nonnan; actual_if_false.$cmp_lt_method($max_minus_one_val).any(), actual_if_false, actual_if_true);

            let res = branchless_float_if!(@nonnan; $condition, true_contrib, false_contrib);

            // (res, res_is_valid)
            (res, res.$cmp_lt_method($max_minus_one_val).any())
        }}
    };
}

#[spirv(compute(threads(16, 16)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: spirv_std::glam::UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] output_image: &spirv_std::image::Image!(2D, format=rgba8, write),
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] spheres: &[Sphere],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] triangles: &[Triangle],
    #[spirv(descriptor_set = 0, binding = 3, storage_buffer)] materials: &[Material],
    #[spirv(descriptor_set = 0, binding = 4, storage_buffer)] lights: &[Light],
    #[spirv(descriptor_set = 0, binding = 5, storage_buffer)] _textures: &[TextureInfo],
    #[spirv(descriptor_set = 0, binding = 6, storage_buffer)] _texture_data: &[u32],
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
    
    
    // BRANCHLESS
    // Raytracing - find closest intersection (spheres and triangles)
    let mut color = vec3(0.0, 0.0, 0.0); // Sky color
    let mut closest_t = f32::MAX - 2.0;
    let mut hit_material_id: u32 = u32::MAX;
    let mut hit_normal = vec3(0.0, 0.0, 0.0);
    let mut hit_point = vec3(0.0, 0.0, 0.0);
    let camera_pos = vec3(push_constants.camera.position[0], push_constants.camera.position[1], push_constants.camera.position[2]);

    // Test sphere intersections - branchless version
    for i in 0..push_constants.sphere_count {
        let sphere_index = i as usize;
        let sphere = spheres[sphere_index];
        let sphere_center = vec3(sphere.center[0], sphere.center[1], sphere.center[2]);
        let oc = camera_pos - sphere_center;

        let a = ray_direction_normalized.dot(ray_direction_normalized);
        let b = 2.0 * oc.dot(ray_direction_normalized);
        let c = oc.dot(oc) - sphere.radius * sphere.radius;

        let discriminant = b * b - 4.0 * a * c;

        // Branchless intersection calculation
        let sqrt_discriminant = discriminant.sqrt();
        let t1 = (-b - sqrt_discriminant) / (2.0 * a);
        let t2 = (-b + sqrt_discriminant) / (2.0 * a);

        // Select closest valid t without branching
        let t1_valid = t1 > RaytracerConfig::MIN_RAY_DISTANCE;
        let (t, t_valid) = branchless_float_if!(t1_valid, t1, t2);

        // Check if intersection is valid and closer
        let hit_valid = (discriminant >= 0.0) && (t > RaytracerConfig::MIN_RAY_DISTANCE) && (t < closest_t) && (i < spheres.len() as u32) && t_valid;

        // Branchless update of intersection data
        let (new_t, new_t_valid) = branchless_float_if!(hit_valid, t, closest_t);
        let new_material_id = branchless_u32_if!(hit_valid, sphere.material_id, hit_material_id);
        let (new_hit_point, new_hit_point_valid) = branchless_vec3_if!(hit_valid, camera_pos + ray_direction_normalized * t, hit_point);
        let (new_hit_normal, new_hit_normal_valid) = branchless_vec3_if!(hit_valid, ((new_hit_point - sphere_center).normalize()), hit_normal);

        let new_is_valid = hit_valid && new_t_valid && new_hit_point_valid && new_hit_normal_valid;
        closest_t = branchless_float_if!(new_is_valid, new_t, closest_t).0;
        hit_material_id = branchless_u32_if!(new_is_valid,new_material_id, hit_material_id);
        hit_point = branchless_vec3_if!(new_is_valid, new_hit_point, hit_point).0;
        hit_normal = branchless_vec3_if!(new_is_valid, new_hit_normal, hit_normal).0;
    }




    // // Test triangle intersections using Möller-Trumbore algorithm
    // for i in 0..push_constants.triangle_count {
    //     let triangle = triangles[i as usize];
    //     let v0 = vec3(triangle.v0[0], triangle.v0[1], triangle.v0[2]);
    //     let v1 = vec3(triangle.v1[0], triangle.v1[1], triangle.v1[2]);
    //     let v2 = vec3(triangle.v2[0], triangle.v2[1], triangle.v2[2]);
    // 
    //     // Möller-Trumbore ray-triangle intersection
    //     let edge1 = v1 - v0;
    //     let edge2 = v2 - v0;
    //     let h = ray_direction_normalized.cross(edge2);
    //     let a = edge1.dot(h);
    // 
    //     // If a is too small, ray is parallel to triangle
    //     let a_abs = a.abs();
    //     let epsilon = RaytracerConfig::MIN_RAY_DISTANCE;
    //     let a_invalid = a_abs < epsilon;
    // 
    // 
    //     let f = 1.0 / a;
    //     let s = camera_pos - v0;
    //     let u = f * s.dot(h);
    // 
    //     let u_invalid = (u < 0.0 || u > 1.0);
    // 
    //     let q = s.cross(edge1);
    //     let v = f * ray_direction_normalized.dot(q);
    // 
    //     let uv_invalid =  (v < 0.0) || ((u + v) > 1.0);
    // 
    // 
    // 
    //     let t = f * edge2.dot(q);
    // 
    //     let valid = !uv_invalid && !u_invalid && !a_invalid && (t > RaytracerConfig::MIN_RAY_DISTANCE) && (t < closest_t);
    // 
    //     let (new_t, new_t_valid) = branchless_float_if!(valid, t, closest_t);
    //     let new_material_id = branchless_u32_if!(valid, triangle.material_id, hit_material_id);
    //     let (new_hit_point, new_hit_point_valid) = branchless_vec3_if!(valid, camera_pos + ray_direction_normalized * t, hit_point);
    //     let (new_hit_normal, new_hit_normal_valid) = branchless_vec3_if!(valid, (edge1.cross(edge2).normalize()), hit_normal);
    // 
    // 
    //     let new_is_valid = valid && new_t_valid && new_hit_point_valid && new_hit_normal_valid;
    //     closest_t = branchless_float_if!(new_is_valid, new_t, closest_t).0;
    //     hit_material_id = branchless_u32_if!(new_is_valid, new_material_id, hit_material_id);
    //     hit_point = branchless_vec3_if!(new_is_valid, new_hit_point, hit_point).0;
    //     hit_normal = branchless_vec3_if!(new_is_valid, new_hit_normal, hit_normal).0;
    // }
    // fixme: try to implement this brancheless as well
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
            hit_material_id = triangle.material_id;
            hit_point = camera_pos + ray_direction_normalized * t;
            hit_normal = edge1.cross(edge2).normalize();
        }
    }

    

    // Apply PBR material if we hit something
    if closest_t < (f32::MAX - 2.0) && hit_material_id < push_constants.material_count {
        if hit_material_id < materials.len() as u32 {
            let material = materials[hit_material_id as usize];
            let albedo = vec3(material.albedo[0], material.albedo[1], material.albedo[2]);
            let emission = vec3(material.emission[0], material.emission[1], material.emission[2]);

            // Calculate lighting from all scene lights
            let _view_dir = -ray_direction_normalized;
            let mut total_lighting = vec3(0.0, 0.0, 0.0);

            // Ambient lighting
            let ambient = albedo * 0.1;
            total_lighting += ambient;

            // Process all lights in the scene - branchless version
            // for light_idx in 0..push_constants.lights_count {
            //     let index_valid = (light_idx < lights.len() as u32) as u32 as f32;
            //     let light_index = if lights.len() > 0 {
            //         if (light_idx as usize) < lights.len() { light_idx as usize } else { lights.len() - 1 }
            //     } else { 0 };
            //     let light = lights[light_index];
            // 
            //     let light_pos = vec3(light.position[0], light.position[1], light.position[2]);
            //     let light_dir_vec = vec3(light.direction[0], light.direction[1], light.direction[2]);
            //     let light_color = vec3(light.color[0], light.color[1], light.color[2]);
            // 
            //     // Calculate directional light properties
            //     let dir_light_dir = -light_dir_vec.normalize();
            //     let dir_light_intensity = hit_normal.dot(dir_light_dir).max(0.0) * light.intensity;
            // 
            //     // Calculate point/spot light properties
            //     let to_light = light_pos - hit_point;
            //     let distance = to_light.length();
            //     let point_light_dir = to_light.normalize();
            //     let attenuation = 1.0 / (1.0 + distance * distance * 0.01);
            //     let point_light_intensity = hit_normal.dot(point_light_dir).max(0.0) * light.intensity * attenuation;
            // 
            //     // Calculate spot light factor
            //     let spot_factor = (-light_dir_vec.normalize()).dot(point_light_dir).max(0.0);
            //     let spot_light_intensity = point_light_intensity * spot_factor;
            // 
            //     // Branchless light type selection using floats
            //     let is_directional = (light.light_type == 0) as u32 as f32;
            //     let is_point = (light.light_type == 1) as u32 as f32;
            //     let is_spot = (light.light_type == 2) as u32 as f32;
            // 
            //     // Combine light calculations branchlessly
            //     let _light_dir = dir_light_dir * is_directional + point_light_dir * (is_point + is_spot);
            //     let light_intensity = dir_light_intensity * is_directional +
            //                         point_light_intensity * is_point +
            //                         spot_light_intensity * is_spot;
            // 
            //     // Branchless BRDF evaluation
            //     let diffuse = albedo / core::f32::consts::PI;
            //     let is_metallic = (material.metallic > 0.5) as u32 as f32;
            //     let metallic_contrib = albedo * light_intensity * 0.5;
            //     let dielectric_contrib = diffuse * light_intensity;
            //     let light_contribution = metallic_contrib * is_metallic + dielectric_contrib * (1.0 - is_metallic);
            // 
            //     // Only add contribution if light intensity is positive and index is valid
            //     let contribution_valid = ((light_intensity > 0.0) as u32 as f32) * index_valid;
            //     total_lighting += light_contribution * light_color * contribution_valid;
            // }

            // Add emission
            color = total_lighting + emission;

            // Handle transmission (simplified) - branchless
            let transmission_factor = material.transmission.max(0.0).min(1.0);
            let transmitted_color = vec3(0.2, 0.2, 0.3);
            color = color * (1.0 - transmission_factor) + transmitted_color * transmission_factor;
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