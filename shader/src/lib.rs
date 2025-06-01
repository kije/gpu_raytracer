#![no_std]

mod ray;
mod intersection;
mod bvh;
mod lighting;
mod material;
mod scene_access;

use spirv_std::spirv;
use spirv_std::glam::{vec2, vec3, vec4, Vec2, Vec4, UVec2, Vec3};
use raytracer_shared::{Triangle, Material, TextureInfo, PushConstants};

use ray::Ray;
use intersection::{Intersection, IntersectionResult};
use bvh::BvhTraverser;
use lighting::LightingCalculator;
use material::MaterialEvaluator;
use scene_access::SceneAccessor;

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
    // Early bounds checking
    if !is_pixel_in_bounds(id, push_constants) {
        return;
    }

    let pixel_coords = calculate_pixel_coordinates(id, push_constants);
    let camera_ray = Ray::from_screen_coordinates(pixel_coords, push_constants);
    
    // Initialize scene accessor for buffer access
    let scene_accessor = SceneAccessor::new(scene_metadata, push_constants);
    
    // Initialize BVH traverser
    let bvh_traverser = BvhTraverser::new(&scene_accessor);
    
    // Find closest intersection
    let intersection_result = find_closest_intersection(
        &camera_ray, 
        &scene_accessor, 
        &bvh_traverser,
        triangles_buffer_0,
        triangles_buffer_1, 
        triangles_buffer_2,
        push_constants
    );
    
    // Calculate final color
    let final_color = if intersection_result.hit {
        calculate_shading(&intersection_result.intersection, &camera_ray, &scene_accessor, materials, push_constants)
    } else {
        vec3(0.0, 0.0, 0.0) // Sky color
    };
    
    // Write to output
    let output_color = vec4(final_color.x, final_color.y, final_color.z, 1.0);
    unsafe {
        output_image.write(UVec2::new(pixel_coords.x, pixel_coords.y), output_color);
    }
}

/// Check if pixel is within bounds
fn is_pixel_in_bounds(id: spirv_std::glam::UVec3, push_constants: &PushConstants) -> bool {
    let pixel_x = push_constants.tile_offset[0] + id.x;
    let pixel_y = push_constants.tile_offset[1] + id.y;
    let width = push_constants.resolution[0] as u32;
    let height = push_constants.resolution[1] as u32;
    
    id.x < push_constants.tile_size[0] &&
    id.y < push_constants.tile_size[1] &&
    pixel_x < width &&
    pixel_y < height
}

/// Calculate pixel coordinates
fn calculate_pixel_coordinates(id: spirv_std::glam::UVec3, push_constants: &PushConstants) -> UVec2 {
    UVec2::new(
        push_constants.tile_offset[0] + id.x,
        push_constants.tile_offset[1] + id.y
    )
}

/// Find the closest intersection for a ray
fn find_closest_intersection(
    ray: &Ray,
    scene_accessor: &SceneAccessor,
    bvh_traverser: &BvhTraverser,
    triangles_buffer_0: &[Triangle],
    triangles_buffer_1: &[Triangle],
    triangles_buffer_2: &[Triangle],
    push_constants: &PushConstants
) -> IntersectionResult {
    let mut closest_t = f32::MAX - 2.0;

    // Test sphere intersections
    let sphere_result = test_sphere_intersections(ray, scene_accessor, closest_t);
    if sphere_result.hit {
        closest_t = sphere_result.intersection.t;
    }

    // Test triangle intersections via BVH
    let triangle_result = if push_constants.metadata_offsets.bvh_nodes_count > 0 {
        bvh_traverser.traverse_and_intersect(
            ray, 
            triangles_buffer_0,
            triangles_buffer_1, 
            triangles_buffer_2,
            push_constants,
            closest_t
        )
    } else {
        test_all_triangles_brute_force(
            ray,
            triangles_buffer_0,
            triangles_buffer_1,
            triangles_buffer_2, 
            push_constants,
            closest_t
        )
    };

    // Return the closest intersection
    if triangle_result.hit && sphere_result.hit {
        if sphere_result.intersection.t < triangle_result.intersection.t {
            sphere_result
        } else {
            triangle_result
        }
    } else if sphere_result.hit {
        sphere_result
    } else if triangle_result.hit {
        triangle_result
    } else {
        IntersectionResult::miss()
    }
}

/// Test intersections with all spheres
fn test_sphere_intersections(
    ray: &Ray, 
    scene_accessor: &SceneAccessor, 
    max_t: f32
) -> IntersectionResult {
    let mut result = IntersectionResult::miss();
    let mut closest_t = max_t;

    for i in 0..scene_accessor.sphere_count() {
        let test_result = intersection::test_sphere_intersection(ray, scene_accessor, i, closest_t);
        if test_result.hit {
            closest_t = test_result.intersection.t;
            result = test_result;
        }
    }

    result
}

/// Brute force triangle intersection testing (fallback)
fn test_all_triangles_brute_force(
    ray: &Ray,
    triangles_buffer_0: &[Triangle],
    triangles_buffer_1: &[Triangle], 
    triangles_buffer_2: &[Triangle],
    push_constants: &PushConstants,
    max_t: f32
) -> IntersectionResult {
    let mut result = IntersectionResult::miss();
    let mut closest_t = max_t;

    for i in 0..push_constants.triangle_count {
        let (triangle_valid, triangle) = get_triangle_from_buffers(i, triangles_buffer_0, triangles_buffer_1, triangles_buffer_2, push_constants);
        if triangle_valid {
            let test_result = intersection::test_triangle_intersection(ray, &triangle, closest_t);
            if test_result.hit {
                closest_t = test_result.intersection.t;
                result = test_result;
            }
        }
    }

    result
}

/// Get triangle from multiple buffers with GPU-friendly result
fn get_triangle_from_buffers(
    index: u32,
    triangles_buffer_0: &[Triangle],
    triangles_buffer_1: &[Triangle],
    triangles_buffer_2: &[Triangle],
    push_constants: &PushConstants
) -> (bool, Triangle) {
    let buffer_index = index / push_constants.triangles_per_buffer;
    let local_index = (index % push_constants.triangles_per_buffer) as usize;
    
    let default_triangle = Triangle {
        v0: [0.0; 3],
        _padding0: 0.0,
        v1: [0.0; 3], 
        _padding1: 0.0,
        v2: [0.0; 3],
        _padding2: 0.0,
        material_id: 0,
        _padding3: [0.0; 3],
    };

    match buffer_index {
        0 => {
            if local_index < triangles_buffer_0.len() {
                (true, triangles_buffer_0[local_index])
            } else {
                (false, default_triangle)
            }
        },
        1 => {
            if local_index < triangles_buffer_1.len() {
                (true, triangles_buffer_1[local_index])
            } else {
                (false, default_triangle)
            }
        },
        2 => {
            if local_index < triangles_buffer_2.len() {
                (true, triangles_buffer_2[local_index])
            } else {
                (false, default_triangle)
            }
        },
        _ => (false, default_triangle)
    }
}

/// Calculate shading for an intersection
fn calculate_shading(
    intersection: &Intersection,
    ray: &Ray,
    scene_accessor: &SceneAccessor,
    materials: &[Material],
    push_constants: &PushConstants
) -> Vec3 {
    if intersection.material_id >= materials.len() as u32 {
        return vec3(1.0, 0.0, 1.0); // Magenta for invalid material
    }

    let material = &materials[intersection.material_id as usize];
    let material_eval = MaterialEvaluator::new(material);
    let lighting_calc = LightingCalculator::new(scene_accessor);

    let lighting = lighting_calc.calculate_lighting(
        intersection, 
        ray, 
        &material_eval, 
        push_constants
    );

    // Apply transmission
    let transmission_factor = material_eval.transmission().max(0.0).min(1.0);
    let transmitted_color = vec3(0.2, 0.2, 0.3);
    
    lighting * (1.0 - transmission_factor) + transmitted_color * transmission_factor
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