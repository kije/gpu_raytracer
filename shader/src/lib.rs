#![no_std]

use spirv_std::float::{f32_to_f16, f16_to_f32, vec2_to_f16x2};
use spirv_std::spirv;
use spirv_std::glam::{vec2, vec3, vec4, Vec2, Vec4, UVec2, Vec3};
use spirv_std::num_traits::Float;
use raytracer_shared::{Sphere, Triangle, Material, Light, TextureInfo, PushConstants, RaytracerConfig, BvhNode};


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
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] scene_metadata: &[u32],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] triangles_buffer_0: &[Triangle],
    #[spirv(descriptor_set = 0, binding = 3, storage_buffer)] triangles_buffer_1: &[Triangle],
    #[spirv(descriptor_set = 0, binding = 4, storage_buffer)] triangles_buffer_2: &[Triangle],
    #[spirv(descriptor_set = 0, binding = 5, storage_buffer)] materials: &[Material],
    #[spirv(descriptor_set = 0, binding = 6, storage_buffer)] _textures: &[TextureInfo],
    #[spirv(descriptor_set = 0, binding = 7, storage_buffer)] _texture_data: &[u32],
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

    // Convert screen coordinates to camera ray using f16 for better memory efficiency
    let uv_f32 = vec2(
        (pixel_x as f32 + 0.5) / width as f32,
        (pixel_y as f32 + 0.5) / height as f32
    );
    
    // Pack UV coordinates into f16 for potential performance benefits
    let _uv_f16_packed = vec2_to_f16x2(uv_f32);
    let uv = uv_f32; // Use f32 for camera calculations to maintain precision

    // Convert UV coordinates to camera space
    let aspect_ratio = width as f32 / height as f32;
    let fov_scale = (push_constants.camera.fov * 0.5 * core::f32::consts::PI / 180.0).tan();

    // Use f16 for intermediate screen space calculations where precision allows
    let camera_x_f16 = f32_to_f16((uv.x * 2.0 - 1.0) * aspect_ratio * fov_scale);
    let camera_y_f16 = f32_to_f16((1.0 - uv.y * 2.0) * fov_scale);
    
    // Convert back to f32 for final ray direction calculation
    let camera_x = f16_to_f32(camera_x_f16);
    let camera_y = f16_to_f32(camera_y_f16);

    // Calculate camera right and up vectors
    let forward = vec3(push_constants.camera.direction[0], push_constants.camera.direction[1], push_constants.camera.direction[2]);
    let up = vec3(push_constants.camera.up[0], push_constants.camera.up[1], push_constants.camera.up[2]);
    let right = forward.cross(up);
    let true_up = right.cross(forward);

    // Calculate ray direction
    let ray_direction = forward + right * camera_x + true_up * camera_y;
    let ray_direction_normalized = ray_direction.normalize();


    // Helper functions to access combined scene metadata buffer - GPU friendly (no Option types)
    let get_sphere_center = |sphere_index: u32| -> Vec3 {
        let base_offset = push_constants.metadata_offsets.spheres_offset as usize;
        let sphere_offset_u32 = base_offset + (sphere_index as usize * (core::mem::size_of::<Sphere>() / 4));
        
        vec3(
            f32::from_bits(scene_metadata[sphere_offset_u32]),
            f32::from_bits(scene_metadata[sphere_offset_u32 + 1]),
            f32::from_bits(scene_metadata[sphere_offset_u32 + 2]),
        )
    };
    
    let get_sphere_radius = |sphere_index: u32| -> f32 {
        let base_offset = push_constants.metadata_offsets.spheres_offset as usize;
        let sphere_offset_u32 = base_offset + (sphere_index as usize * (core::mem::size_of::<Sphere>() / 4));
        f32::from_bits(scene_metadata[sphere_offset_u32 + 3])
    };
    
    let get_sphere_material_id = |sphere_index: u32| -> u32 {
        let base_offset = push_constants.metadata_offsets.spheres_offset as usize;
        let sphere_offset_u32 = base_offset + (sphere_index as usize * (core::mem::size_of::<Sphere>() / 4));
        scene_metadata[sphere_offset_u32 + 4]
    };
    
    let get_light_position = |light_index: u32| -> Vec3 {
        let base_offset = push_constants.metadata_offsets.lights_offset as usize;
        let light_offset_u32 = base_offset + (light_index as usize * (core::mem::size_of::<Light>() / 4));
        
        vec3(
            f32::from_bits(scene_metadata[light_offset_u32]),
            f32::from_bits(scene_metadata[light_offset_u32 + 1]),
            f32::from_bits(scene_metadata[light_offset_u32 + 2]),
        )
    };
    
    let get_light_type = |light_index: u32| -> u32 {
        let base_offset = push_constants.metadata_offsets.lights_offset as usize;
        let light_offset_u32 = base_offset + (light_index as usize * (core::mem::size_of::<Light>() / 4));
        scene_metadata[light_offset_u32 + 3]
    };
    
    let get_light_color = |light_index: u32| -> Vec3 {
        let base_offset = push_constants.metadata_offsets.lights_offset as usize;
        let light_offset_u32 = base_offset + (light_index as usize * (core::mem::size_of::<Light>() / 4));
        
        vec3(
            f32::from_bits(scene_metadata[light_offset_u32 + 4]),
            f32::from_bits(scene_metadata[light_offset_u32 + 5]),
            f32::from_bits(scene_metadata[light_offset_u32 + 6]),
        )
    };
    
    let get_light_intensity = |light_index: u32| -> f32 {
        let base_offset = push_constants.metadata_offsets.lights_offset as usize;
        let light_offset_u32 = base_offset + (light_index as usize * (core::mem::size_of::<Light>() / 4));
        f32::from_bits(scene_metadata[light_offset_u32 + 7])
    };
    
    let get_light_direction = |light_index: u32| -> Vec3 {
        let base_offset = push_constants.metadata_offsets.lights_offset as usize;
        let light_offset_u32 = base_offset + (light_index as usize * (core::mem::size_of::<Light>() / 4));
        
        vec3(
            f32::from_bits(scene_metadata[light_offset_u32 + 8]),
            f32::from_bits(scene_metadata[light_offset_u32 + 9]),
            f32::from_bits(scene_metadata[light_offset_u32 + 10]),
        )
    };
    
    // BVH helper functions to access BVH nodes from scene metadata buffer
    let get_bvh_node_bounds_min = |node_index: u32| -> Vec3 {
        let base_offset = push_constants.metadata_offsets.bvh_nodes_offset as usize;
        let node_offset_u32 = base_offset + (node_index as usize * (core::mem::size_of::<BvhNode>() / 4));
        
        vec3(
            f32::from_bits(scene_metadata[node_offset_u32]),
            f32::from_bits(scene_metadata[node_offset_u32 + 1]),
            f32::from_bits(scene_metadata[node_offset_u32 + 2]),
        )
    };
    
    let get_bvh_node_bounds_max = |node_index: u32| -> Vec3 {
        let base_offset = push_constants.metadata_offsets.bvh_nodes_offset as usize;
        let node_offset_u32 = base_offset + (node_index as usize * (core::mem::size_of::<BvhNode>() / 4));
        
        vec3(
            f32::from_bits(scene_metadata[node_offset_u32 + 4]),
            f32::from_bits(scene_metadata[node_offset_u32 + 5]),
            f32::from_bits(scene_metadata[node_offset_u32 + 6]),
        )
    };
    
    let get_bvh_node_left_child = |node_index: u32| -> u32 {
        let base_offset = push_constants.metadata_offsets.bvh_nodes_offset as usize;
        let node_offset_u32 = base_offset + (node_index as usize * (core::mem::size_of::<BvhNode>() / 4));
        scene_metadata[node_offset_u32 + 8]
    };
    
    let get_bvh_node_right_child = |node_index: u32| -> u32 {
        let base_offset = push_constants.metadata_offsets.bvh_nodes_offset as usize;
        let node_offset_u32 = base_offset + (node_index as usize * (core::mem::size_of::<BvhNode>() / 4));
        scene_metadata[node_offset_u32 + 9]
    };
    
    let get_bvh_node_triangle_start = |node_index: u32| -> u32 {
        let base_offset = push_constants.metadata_offsets.bvh_nodes_offset as usize;
        let node_offset_u32 = base_offset + (node_index as usize * (core::mem::size_of::<BvhNode>() / 4));
        scene_metadata[node_offset_u32 + 10]
    };
    
    let get_bvh_node_triangle_count = |node_index: u32| -> u32 {
        let base_offset = push_constants.metadata_offsets.bvh_nodes_offset as usize;
        let node_offset_u32 = base_offset + (node_index as usize * (core::mem::size_of::<BvhNode>() / 4));
        scene_metadata[node_offset_u32 + 11]
    };
    
    let get_triangle_index = |index: u32| -> u32 {
        let base_offset = push_constants.metadata_offsets.triangle_indices_offset as usize;
        let triangle_index_offset = base_offset + (index as usize);
        scene_metadata[triangle_index_offset]
    };
    
    // Ray-AABB intersection test for BVH traversal
    let ray_aabb_intersect = |ray_origin: Vec3, ray_dir: Vec3, aabb_min: Vec3, aabb_max: Vec3| -> bool {
        let inv_dir = vec3(1.0 / ray_dir.x, 1.0 / ray_dir.y, 1.0 / ray_dir.z);
        
        let t1 = (aabb_min - ray_origin) * inv_dir;
        let t2 = (aabb_max - ray_origin) * inv_dir;
        
        let tmin = t1.min(t2);
        let tmax = t1.max(t2);
        
        let tmin_max = tmin.x.max(tmin.y).max(tmin.z);
        let tmax_min = tmax.x.min(tmax.y).min(tmax.z);
        
        tmax_min >= 0.0 && tmin_max <= tmax_min
    };

    // BRANCHLESS
    // Raytracing - find closest intersection (spheres and triangles)
    let mut color = vec3(0.0, 0.0, 0.0); // Sky color
    let mut closest_t = f32::MAX - 2.0;
    let mut hit_material_id: u32 = u32::MAX;
    let mut hit_normal = vec3(0.0, 0.0, 0.0);
    let mut hit_point = vec3(0.0, 0.0, 0.0);
    let camera_pos = vec3(push_constants.camera.position[0], push_constants.camera.position[1], push_constants.camera.position[2]);


    // Test sphere intersections with AABB culling for acceleration
    for i in 0..push_constants.metadata_offsets.spheres_count {
        // Quick AABB test for sphere
        let sphere_center = get_sphere_center(i);
        let sphere_radius = get_sphere_radius(i);
        let sphere_min = sphere_center - Vec3::splat(sphere_radius);
        let sphere_max = sphere_center + Vec3::splat(sphere_radius);
        
        // Only test intersection if ray intersects sphere's AABB
        if ray_aabb_intersect(camera_pos, ray_direction_normalized, sphere_min, sphere_max) {
            // Inline sphere intersection to avoid closure return types
            let oc = camera_pos - sphere_center;
            let a = ray_direction_normalized.dot(ray_direction_normalized);
            let b = 2.0 * oc.dot(ray_direction_normalized);
            let c = oc.dot(oc) - sphere_radius * sphere_radius;
            let discriminant = b * b - 4.0 * a * c;
            
            if discriminant >= 0.0 {
                let sqrt_discriminant = discriminant.sqrt();
                let t1 = (-b - sqrt_discriminant) / (2.0 * a);
                let t2 = (-b + sqrt_discriminant) / (2.0 * a);
                
                let t = if t1 > RaytracerConfig::MIN_RAY_DISTANCE { t1 } else { t2 };
                
                if t > RaytracerConfig::MIN_RAY_DISTANCE && t < closest_t {
                    closest_t = t;
                    hit_material_id = get_sphere_material_id(i);
                    hit_point = camera_pos + ray_direction_normalized * t;
                    hit_normal = (hit_point - sphere_center).normalize();
                }
            }
        }
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


    // BVH traversal using explicit stack to avoid recursion
    if push_constants.metadata_offsets.bvh_nodes_count > 0 {
        // Stack for BVH traversal (GPU-friendly - fixed size)
        let mut node_stack: [u32; 64] = [0xFFFFFFFF; 64];
        let mut stack_ptr = 0;
        node_stack[0] = 0; // Start with root node
        stack_ptr += 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let current_node = node_stack[stack_ptr];
            
            if current_node == 0xFFFFFFFF || current_node >= push_constants.metadata_offsets.bvh_nodes_count {
                continue;
            }

            // Test current node AABB
            let bounds_min = get_bvh_node_bounds_min(current_node);
            let bounds_max = get_bvh_node_bounds_max(current_node);
            
            if !ray_aabb_intersect(camera_pos, ray_direction_normalized, bounds_min, bounds_max) {
                continue;
            }

            // Check if leaf node
            let left_child = get_bvh_node_left_child(current_node);
            let right_child = get_bvh_node_right_child(current_node);
            
            if left_child == 0xFFFFFFFF { // Leaf node
                let triangle_start = get_bvh_node_triangle_start(current_node);
                let triangle_count = get_bvh_node_triangle_count(current_node);
                
                // Test all triangles in this leaf - inline intersection to avoid closures
                for i in 0..triangle_count {
                    if triangle_start + i >= push_constants.metadata_offsets.triangle_indices_count {
                        break;
                    }
                    
                    let triangle_index = get_triangle_index(triangle_start + i);
                    let buffer_index = triangle_index / push_constants.triangles_per_buffer;
                    let local_index = (triangle_index % push_constants.triangles_per_buffer) as usize;

                    let triangle = match buffer_index {
                        0 => {
                            if local_index < triangles_buffer_0.len() {
                                triangles_buffer_0[local_index]
                            } else {
                                continue;
                            }
                        }
                        1 => {
                            if local_index < triangles_buffer_1.len() {
                                triangles_buffer_1[local_index]
                            } else {
                                continue;
                            }
                        }
                        2 => {
                            if local_index < triangles_buffer_2.len() {
                                triangles_buffer_2[local_index]
                            } else {
                                continue;
                            }
                        }
                        _ => continue
                    };

                    let v0 = vec3(triangle.v0[0], triangle.v0[1], triangle.v0[2]);
                    let v1 = vec3(triangle.v1[0], triangle.v1[1], triangle.v1[2]);
                    let v2 = vec3(triangle.v2[0], triangle.v2[1], triangle.v2[2]);

                    // Möller-Trumbore ray-triangle intersection
                    let edge1 = v1 - v0;
                    let edge2 = v2 - v0;
                    let h = ray_direction_normalized.cross(edge2);
                    let a = edge1.dot(h);

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
            } else { // Internal node
                // Add children to stack (right first for left-first traversal)
                if right_child != 0xFFFFFFFF && stack_ptr < 63 {
                    node_stack[stack_ptr] = right_child;
                    stack_ptr += 1;
                }
                if left_child != 0xFFFFFFFF && stack_ptr < 63 {
                    node_stack[stack_ptr] = left_child;
                    stack_ptr += 1;
                }
            }
        }
    } else {
        // Fallback to brute-force if no BVH available
        for i in 0..push_constants.triangle_count {
            let buffer_index = i / push_constants.triangles_per_buffer;
            let local_index = (i % push_constants.triangles_per_buffer) as usize;

            let triangle = match buffer_index {
                0 => {
                    if local_index < triangles_buffer_0.len() {
                        triangles_buffer_0[local_index]
                    } else {
                        continue;
                    }
                }
                1 => {
                    if local_index < triangles_buffer_1.len() {
                        triangles_buffer_1[local_index]
                    } else {
                        continue;
                    }
                }
                2 => {
                    if local_index < triangles_buffer_2.len() {
                        triangles_buffer_2[local_index]
                    } else {
                        continue;
                    }
                }
                _ => continue
            };

            let v0 = vec3(triangle.v0[0], triangle.v0[1], triangle.v0[2]);
            let v1 = vec3(triangle.v1[0], triangle.v1[1], triangle.v1[2]);
            let v2 = vec3(triangle.v2[0], triangle.v2[1], triangle.v2[2]);

            // Möller-Trumbore ray-triangle intersection
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let h = ray_direction_normalized.cross(edge2);
            let a = edge1.dot(h);

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
    }

    

    // Apply PBR material if we hit something
    if closest_t < (f32::MAX - 2.0) && hit_material_id < push_constants.material_count {
        if hit_material_id < materials.len() as u32 {
            let material = materials[hit_material_id as usize];
            let albedo = vec3(material.albedo[0], material.albedo[1], material.albedo[2]);
            let emission = vec3(material.emission[0], material.emission[1], material.emission[2]);
            
            // Unpack and convert f16 material properties back to f32 for calculations
            let metallic = f16_to_f32(material.metallic_roughness_f16 & 0xFFFF); // Low 16 bits
            let roughness = f16_to_f32(material.metallic_roughness_f16 >> 16);   // High 16 bits
            let ior = f16_to_f32(material.ior_transmission_f16 & 0xFFFF);        // Low 16 bits
            let transmission = f16_to_f32(material.ior_transmission_f16 >> 16);   // High 16 bits

            // Calculate lighting from all scene lights
            let _view_dir = -ray_direction_normalized;
            let mut total_lighting = vec3(0.0, 0.0, 0.0);

            // Ambient lighting
            let ambient = albedo * 0.1;
            total_lighting += ambient;

            // Process all lights in the scene using scene metadata buffer
            for light_idx in 0..push_constants.metadata_offsets.lights_count {
                let index_valid = (light_idx < push_constants.metadata_offsets.lights_count) as u32 as f32;
                
                let light_pos = get_light_position(light_idx);
                let light_type = get_light_type(light_idx);
                let light_color = get_light_color(light_idx);
                let light_intensity = get_light_intensity(light_idx);
                let light_dir_vec = get_light_direction(light_idx);
            
                // Calculate directional light properties
                let dir_light_dir = -light_dir_vec.normalize();
                let dir_light_intensity = hit_normal.dot(dir_light_dir).max(0.0) * light_intensity;
            
                // Calculate point/spot light properties
                let to_light = light_pos - hit_point;
                let distance = to_light.length();
                let point_light_dir = to_light.normalize();
                
                // Use f16 for attenuation calculation (0.0-1.0 range, perfect for f16)
                let attenuation_f32 = 1.0 / (1.0 + distance * distance * 0.01);
                let attenuation_f16 = f32_to_f16(attenuation_f32);
                let attenuation = f16_to_f32(attenuation_f16);
                
                let point_light_intensity = hit_normal.dot(point_light_dir).max(0.0) * light_intensity * attenuation;
            
                // Calculate spot light factor
                let spot_factor = (-light_dir_vec.normalize()).dot(point_light_dir).max(0.0);
                let spot_light_intensity = point_light_intensity * spot_factor;
            
                // Branchless light type selection using floats
                let is_directional = (light_type == 0) as u32 as f32;
                let is_point = (light_type == 1) as u32 as f32;
                let is_spot = (light_type == 2) as u32 as f32;
            
                // Combine light calculations branchlessly
                let _light_dir = dir_light_dir * is_directional + point_light_dir * (is_point + is_spot);
                let light_intensity_final = dir_light_intensity * is_directional +
                                           point_light_intensity * is_point +
                                           spot_light_intensity * is_spot;
            
                // Branchless BRDF evaluation using converted f16 values
                let diffuse = albedo / core::f32::consts::PI;
                let is_metallic = (metallic > 0.5) as u32 as f32;
                let metallic_contrib = albedo * light_intensity_final * 0.5;
                let dielectric_contrib = diffuse * light_intensity_final;
                let light_contribution = metallic_contrib * is_metallic + dielectric_contrib * (1.0 - is_metallic);
            
                // Only add contribution if light intensity is positive and index is valid
                let contribution_valid = ((light_intensity_final > 0.0) as u32 as f32) * index_valid;
                total_lighting += light_contribution * light_color * contribution_valid;
            }

            // Add emission
            color = total_lighting + emission;

            // Handle transmission (simplified) - branchless using converted f16 value
            let transmission_factor = transmission.max(0.0).min(1.0);
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