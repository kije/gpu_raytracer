#![no_std]

mod ray;
mod intersection;
mod bvh;
mod lighting;
mod material;
mod scene_access;
mod triangle_access;
mod wavefront;

use spirv_std::spirv;
use spirv_std::glam::{vec2, vec3, vec4, Vec2, Vec4, UVec2, Vec3};
use raytracer_shared::{Triangle, Material, TextureInfo, PushConstants, branchless_u32_if};

use ray::Ray;
use intersection::{Intersection, IntersectionResult};
use bvh::BvhTraverser;
use lighting::LightingCalculator;
use material::MaterialEvaluator;
use scene_access::SceneAccessor;
use triangle_access::TriangleAccessor;
use wavefront::{SimpleRng, generate_camera_ray, process_wavefront_ray};

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
    
    // Initialize scene accessor for buffer access
    let scene_accessor = SceneAccessor::new(scene_metadata, push_constants);
    
    // Initialize BVH traverser
    let bvh_traverser = BvhTraverser::new(&scene_accessor);
    
    let final_color = if push_constants.wavefront_mode() != 0 {
        // Wavefront raytracing mode
        run_wavefront_raytracing(
            pixel_coords, &scene_accessor, &bvh_traverser,
            triangles_buffer_0, triangles_buffer_1, triangles_buffer_2,
            materials, push_constants
        )
    } else {
        // Legacy single-ray mode
        let camera_ray = Ray::from_screen_coordinates(pixel_coords, push_constants);
        
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
        
        // Calculate final color based on color channel
        if intersection_result.hit {
            calculate_shading(&intersection_result.intersection, &camera_ray, &scene_accessor, materials, push_constants)
        } else {
            vec3(0.0, 0.0, 0.0) // Sky color
        }
    };
    
    // Filter color by channel for chromatic aberration
    let channel_filtered_color = filter_color_by_channel(final_color, push_constants.color_channel());
    
    // Write to output
    let output_color = vec4(channel_filtered_color.x, channel_filtered_color.y, channel_filtered_color.z, 1.0);
    unsafe {
        output_image.write(UVec2::new(pixel_coords.x, pixel_coords.y), output_color);
    }
}

/// Run wavefront raytracing for a pixel
fn run_wavefront_raytracing(
    pixel_coords: UVec2,
    scene_accessor: &SceneAccessor,
    bvh_traverser: &BvhTraverser,
    triangles_buffer_0: &[Triangle],
    triangles_buffer_1: &[Triangle],
    triangles_buffer_2: &[Triangle],
    materials: &[Material],
    push_constants: &PushConstants,
) -> Vec3 {
    // Create RNG with pixel-specific seed
    let pixel_seed = push_constants.frame_seed
        .wrapping_add(pixel_coords.x)
        .wrapping_add(pixel_coords.y.wrapping_mul(push_constants.resolution[0] as u32));
    let mut rng = SimpleRng::new(pixel_seed);
    
    // Generate camera ray for this pixel  
    let mut wavefront_ray = generate_camera_ray(
        pixel_coords.x, pixel_coords.y, push_constants, push_constants.color_channel()
    );
    
    let mut accumulated_color = vec3(0.0, 0.0, 0.0);
    
    // Process ray for the current bounce depth only
    // In a full wavefront implementation, this would be done across all pixels for each bounce depth
    for bounce in 0..=push_constants.max_bounce_depth() {
        if bounce != push_constants.current_bounce_depth() {
            continue;
        }
        
        if wavefront_ray.active == 0 {
            break;
        }
        
        let (color_contribution, contributed) = process_wavefront_ray(
            &wavefront_ray,
            scene_accessor,
            bvh_traverser,
            triangles_buffer_0,
            triangles_buffer_1,
            triangles_buffer_2,
            materials,
            push_constants,
            &mut rng,
        );
        
        if contributed {
            accumulated_color += color_contribution;
        }
        
        // For this simplified implementation, terminate after first intersection
        // In a full implementation, this would generate continuation rays
        wavefront_ray.active = 0;
        break;
    }
    
    accumulated_color
}

/// Check if pixel is within bounds
fn is_pixel_in_bounds(id: spirv_std::glam::UVec3, push_constants: &PushConstants) -> bool {
    let pixel_x = push_constants.tile_offset[0] + id.x;
    let pixel_y = push_constants.tile_offset[1] + id.y;
    let width = push_constants.resolution[0] as u32;
    let height = push_constants.resolution[1] as u32;

    let (tile_width, tile_height) = push_constants.unpack_tile_size();
    id.x < tile_width &&
        id.y < tile_height &&
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
            scene_accessor,
            triangles_buffer_0,
            triangles_buffer_1,
            triangles_buffer_2, 
            push_constants,
            closest_t
        )
    };

    // Branchless closest intersection selection using comparison masks
    let sphere_closer = branchless_u32_if!(
        sphere_result.intersection.t < triangle_result.intersection.t,
        1u32,
        0u32
    );
    
    let both_hit = branchless_u32_if!(
        triangle_result.hit && sphere_result.hit,
        1u32,
        0u32
    );
    
    let sphere_only = branchless_u32_if!(
        sphere_result.hit && !triangle_result.hit,
        1u32,
        0u32
    );
    
    let triangle_only = branchless_u32_if!(
        triangle_result.hit && !sphere_result.hit,
        1u32,
        0u32
    );
    
    // Select result branchlessly
    let use_sphere = both_hit * sphere_closer + sphere_only;
    let use_triangle = both_hit * (1u32 - sphere_closer) + triangle_only;
    
    if use_sphere != 0 {
        sphere_result
    } else if use_triangle != 0 {
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
    scene_accessor: &SceneAccessor,
    triangles_buffer_0: &[Triangle],
    triangles_buffer_1: &[Triangle], 
    triangles_buffer_2: &[Triangle],
    push_constants: &PushConstants,
    max_t: f32
) -> IntersectionResult {
    let mut result = IntersectionResult::miss();
    let mut closest_t = max_t;

    for i in 0..push_constants.triangle_count {
        let (triangle_valid, v0, v1, v2, material_id) = TriangleAccessor::get_triangle_vertices_direct(i, triangles_buffer_0, triangles_buffer_1, triangles_buffer_2, push_constants, &scene_accessor);
        if triangle_valid {
            let test_result = intersection::test_triangle_intersection_direct(ray, v0, v1, v2, material_id, closest_t);
            if test_result.hit {
                closest_t = test_result.intersection.t;
                result = test_result;
            }
        }
    }

    result
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

    // Apply transmission with wavelength-dependent refraction for chromatic aberration
    let transmission_factor = material_eval.transmission().max(0.0).min(1.0);
    
    // Calculate color contribution based on wavelength-dependent IOR
    if transmission_factor > 0.0 {
        let wavelength_ior = material_eval.ior_for_channel(push_constants.color_channel());
        
        // Simple chromatic dispersion simulation
        // Higher IOR means more bending, which affects color intensity
        let dispersion_factor = (wavelength_ior - 1.0) / (material_eval.ior() - 1.0);
        let transmitted_color = vec3(0.2, 0.2, 0.3) * dispersion_factor;
        
        lighting * (1.0 - transmission_factor) + transmitted_color * transmission_factor
    } else {
        lighting
    }
}

/// Filter color by channel for chromatic aberration
/// Returns color for only the specified channel (0=red, 1=green, 2=blue)
fn filter_color_by_channel(color: Vec3, channel: u32) -> Vec3 {
    match channel {
        0 => vec3(color.x, 0.0, 0.0), // Red channel only
        1 => vec3(0.0, color.y, 0.0), // Green channel only  
        2 => vec3(0.0, 0.0, color.z), // Blue channel only
        _ => color, // Fallback to full color
    }
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

// Fragment shader to combine and display 3 chromatic aberration textures
#[spirv(fragment)]
pub fn main_fs(
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] texture_red: &spirv_std::image::Image!(2D, type=f32, sampled),
    #[spirv(descriptor_set = 0, binding = 1)] texture_green: &spirv_std::image::Image!(2D, type=f32, sampled),
    #[spirv(descriptor_set = 0, binding = 2)] texture_blue: &spirv_std::image::Image!(2D, type=f32, sampled),
    #[spirv(descriptor_set = 0, binding = 3)] sampler: &spirv_std::Sampler,
    output: &mut Vec4,
) {
    // Sample each color channel texture
    let red_sample = texture_red.sample(*sampler, uv);
    let green_sample = texture_green.sample(*sampler, uv);
    let blue_sample = texture_blue.sample(*sampler, uv);
    
    // Combine RGB channels into final color
    // Each texture contains the filtered color for that channel, so we extract the relevant component
    let final_color = vec4(
        red_sample.x,    // Red component from red channel texture
        green_sample.y,  // Green component from green channel texture
        blue_sample.z,   // Blue component from blue channel texture
        1.0              // Alpha channel
    );
    
    *output = final_color;
}