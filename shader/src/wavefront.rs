use spirv_std::glam::{vec2, vec3, Vec3};
use raytracer_shared::{WavefrontRay, PushConstants, branchless_u32_if};
use spirv_std::num_traits::Float;

use crate::ray::Ray;
use crate::intersection::{Intersection, IntersectionResult};
use crate::bvh::BvhTraverser;
use crate::lighting::LightingCalculator;
use crate::material::MaterialEvaluator;
use crate::scene_access::SceneAccessor;
use crate::triangle_access::TriangleAccessor;

/// Extension trait to add deactivate method to WavefrontRay
pub trait WavefrontRayExt {
    fn deactivate(&mut self);
}

impl WavefrontRayExt for WavefrontRay {
    fn deactivate(&mut self) {
        unsafe {
            // Safe to cast since we're working with the same memory layout
            let ptr = self as *mut WavefrontRay as *mut u32;
            // The active field is at offset 19 u32s (76 bytes / 4)
            *ptr.add(19) = 0;
        }
    }
}

/// Wavefront raytracing implementation
/// This module handles multi-bounce raytracing using a wavefront model where rays are processed
/// in groups per bounce depth, which is more efficient on GPUs than recursive raytracing.

/// Convert between WavefrontRay and Ray for intersection testing
impl Ray {
    /// Create a Ray from a WavefrontRay for intersection testing
    pub fn from_wavefront_ray(wavefront_ray: &WavefrontRay) -> Self {
        Self {
            origin: vec3(wavefront_ray.origin[0], wavefront_ray.origin[1], wavefront_ray.origin[2]),
            direction: vec3(wavefront_ray.direction[0], wavefront_ray.direction[1], wavefront_ray.direction[2]),
        }
    }
}

/// Simple pseudo-random number generator for GPU use
/// Uses a linear congruential generator for simplicity and speed
pub struct SimpleRng {
    seed: u32,
}

impl SimpleRng {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }
    
    /// Generate next random u32
    pub fn next_u32(&mut self) -> u32 {
        // Linear congruential generator: x_n+1 = (a * x_n + c) mod m
        // Using constants from Numerical Recipes
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        self.seed
    }
    
    /// Generate random f32 in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16777216.0 // 2^24 for good precision
    }
    
    /// Generate random f32 in [-1, 1)
    pub fn next_f32_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

/// Generate a camera ray for wavefront raytracing
pub fn generate_camera_ray(
    pixel_x: u32,
    pixel_y: u32,
    push_constants: &PushConstants,
    wavelength_channel: u32,
) -> WavefrontRay {
    // Convert screen coordinates to camera ray
    let uv = vec2(
        (pixel_x as f32 + 0.5) / push_constants.resolution[0],
        (pixel_y as f32 + 0.5) / push_constants.resolution[1]
    );

    // Calculate camera space coordinates
    let aspect_ratio = push_constants.resolution[0] / push_constants.resolution[1];
    let fov_scale = (push_constants.camera.fov * 0.5 * core::f32::consts::PI / 180.0).tan();

    let camera_x = (uv.x * 2.0 - 1.0) * aspect_ratio * fov_scale;
    let camera_y = (1.0 - uv.y * 2.0) * fov_scale;

    // Calculate camera vectors
    let forward = vec3(push_constants.camera.direction[0], push_constants.camera.direction[1], push_constants.camera.direction[2]);
    let up = vec3(push_constants.camera.up[0], push_constants.camera.up[1], push_constants.camera.up[2]);
    let right = forward.cross(up);
    let true_up = right.cross(forward);

    // Calculate ray direction
    let ray_direction = forward + right * camera_x + true_up * camera_y;
    let ray_direction_normalized = ray_direction.normalize();

    let origin = vec3(push_constants.camera.position[0], push_constants.camera.position[1], push_constants.camera.position[2]);

    WavefrontRay::camera_ray(
        [origin.x, origin.y, origin.z],
        [ray_direction_normalized.x, ray_direction_normalized.y, ray_direction_normalized.z],
        [pixel_x, pixel_y],
        wavelength_channel,
    )
}

/// Process a wavefront ray and generate new rays if needed
/// Returns true if the ray contributed to the final image
pub fn process_wavefront_ray(
    wavefront_ray: &WavefrontRay,
    scene_accessor: &SceneAccessor,
    bvh_traverser: &BvhTraverser,
    triangles_buffer_0: &[raytracer_shared::Triangle],
    triangles_buffer_1: &[raytracer_shared::Triangle],
    triangles_buffer_2: &[raytracer_shared::Triangle],
    materials: &[raytracer_shared::Material],
    push_constants: &PushConstants,
    rng: &mut SimpleRng,
) -> (Vec3, bool) {
    // Skip inactive rays
    if wavefront_ray.active == 0 {
        return (vec3(0.0, 0.0, 0.0), false);
    }

    // Convert to Ray for intersection testing
    let ray = Ray::from_wavefront_ray(wavefront_ray);
    
    // Find closest intersection using existing intersection logic
    let intersection_result = find_closest_intersection(
        &ray, 
        scene_accessor, 
        bvh_traverser,
        triangles_buffer_0,
        triangles_buffer_1, 
        triangles_buffer_2,
        push_constants
    );
    
    if !intersection_result.hit {
        // Ray hit the sky - return sky color weighted by throughput
        let sky_color = vec3(0.1, 0.2, 0.3); // Simple sky color
        let throughput = vec3(wavefront_ray.throughput[0], wavefront_ray.throughput[1], wavefront_ray.throughput[2]);
        return (sky_color * throughput, true);
    }

    // Calculate shading at intersection
    let color = calculate_wavefront_shading(
        &intersection_result.intersection,
        &ray,
        scene_accessor,
        materials,
        push_constants,
        wavefront_ray,
        rng
    );
    
    (color, true)
}

/// Calculate shading for a wavefront ray intersection
fn calculate_wavefront_shading(
    intersection: &Intersection,
    ray: &Ray,
    scene_accessor: &SceneAccessor,
    materials: &[raytracer_shared::Material],
    push_constants: &PushConstants,
    wavefront_ray: &WavefrontRay,
    _rng: &mut SimpleRng,
) -> Vec3 {
    if intersection.material_id >= materials.len() as u32 {
        return vec3(1.0, 0.0, 1.0); // Magenta for invalid material
    }

    let material = &materials[intersection.material_id as usize];
    let material_eval = MaterialEvaluator::new(material);
    let lighting_calc = LightingCalculator::new(scene_accessor);

    // Calculate basic lighting (same as current implementation)
    let lighting = lighting_calc.calculate_lighting(
        intersection, 
        ray, 
        &material_eval, 
        push_constants
    );
    
    // Apply throughput from wavefront ray
    let throughput = vec3(wavefront_ray.throughput[0], wavefront_ray.throughput[1], wavefront_ray.throughput[2]);
    
    // Apply transmission with wavelength-dependent refraction for chromatic aberration
    let transmission_factor = material_eval.transmission().max(0.0).min(1.0);
    
    if transmission_factor > 0.0 {
        let wavelength_ior = material_eval.ior_for_channel(push_constants.color_channel());
        
        // Simple chromatic dispersion simulation
        let dispersion_factor = (wavelength_ior - 1.0) / (material_eval.ior() - 1.0);
        let transmitted_color = vec3(0.2, 0.2, 0.3) * dispersion_factor;
        
        let final_color = lighting * (1.0 - transmission_factor) + transmitted_color * transmission_factor;
        final_color * throughput
    } else {
        lighting * throughput
    }
}

/// Find the closest intersection for a ray (reusing existing logic)
fn find_closest_intersection(
    ray: &Ray,
    scene_accessor: &SceneAccessor,
    bvh_traverser: &BvhTraverser,
    triangles_buffer_0: &[raytracer_shared::Triangle],
    triangles_buffer_1: &[raytracer_shared::Triangle],
    triangles_buffer_2: &[raytracer_shared::Triangle],
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

/// Test intersections with all spheres (reusing existing logic)
fn test_sphere_intersections(
    ray: &Ray, 
    scene_accessor: &SceneAccessor, 
    max_t: f32
) -> IntersectionResult {
    let mut result = IntersectionResult::miss();
    let mut closest_t = max_t;

    for i in 0..scene_accessor.sphere_count() {
        let test_result = crate::intersection::test_sphere_intersection(ray, scene_accessor, i, closest_t);
        if test_result.hit {
            closest_t = test_result.intersection.t;
            result = test_result;
        }
    }

    result
}

/// Brute force triangle intersection testing (reusing existing logic)
fn test_all_triangles_brute_force(
    ray: &Ray,
    scene_accessor: &SceneAccessor,
    triangles_buffer_0: &[raytracer_shared::Triangle],
    triangles_buffer_1: &[raytracer_shared::Triangle], 
    triangles_buffer_2: &[raytracer_shared::Triangle],
    push_constants: &PushConstants,
    max_t: f32
) -> IntersectionResult {
    let mut result = IntersectionResult::miss();
    let mut closest_t = max_t;

    for i in 0..push_constants.triangle_count {
        let (triangle_valid, v0, v1, v2, material_id) = TriangleAccessor::get_triangle_vertices_direct(i, triangles_buffer_0, triangles_buffer_1, triangles_buffer_2, push_constants, &scene_accessor);
        if triangle_valid {
            let test_result = crate::intersection::test_triangle_intersection_direct(ray, v0, v1, v2, material_id, closest_t);
            if test_result.hit {
                closest_t = test_result.intersection.t;
                result = test_result;
            }
        }
    }

    result
}

/// Generate continuation rays from an intersection point
/// This would be used in a full wavefront implementation to generate reflection/transmission rays
pub fn generate_continuation_rays(
    _intersection: &Intersection,
    _ray: &Ray,
    _material: &raytracer_shared::Material,
    _wavefront_ray: &WavefrontRay,
    _rng: &mut SimpleRng,
) -> u32 {
    // Placeholder for generating reflection and transmission rays
    // In a full implementation, this would:
    // 1. Evaluate the BRDF/BTDF
    // 2. Sample new ray directions based on material properties
    // 3. Apply Russian roulette for ray termination
    // 4. Create new WavefrontRay instances with updated throughput and bounce depth
    
    0 // For now, return no continuation rays (count)
}