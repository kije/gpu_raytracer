use spirv_std::glam::{vec3, Vec3};
use spirv_std::num_traits::Float;
use raytracer_shared::RaytracerConfig;
use crate::ray::Ray;
use crate::scene_access::SceneAccessor;
use crate::triangle_access::ExpandedTriangle;

/// Intersection data
#[derive(Clone, Copy)]
pub struct Intersection {
    pub t: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub material_id: u32,
}

/// Result of intersection test - using GPU-friendly representation
#[derive(Clone, Copy)]
pub struct IntersectionResult {
    pub hit: bool,
    pub intersection: Intersection,
}

impl IntersectionResult {
    pub fn miss() -> Self {
        Self {
            hit: false,
            intersection: Intersection::new(f32::MAX, Vec3::ZERO, Vec3::ZERO, u32::MAX),
        }
    }

    pub fn hit(intersection: Intersection) -> Self {
        Self {
            hit: true,
            intersection,
        }
    }
}

impl Intersection {
    pub fn new(t: f32, point: Vec3, normal: Vec3, material_id: u32) -> Self {
        Self {
            t,
            point,
            normal,
            material_id,
        }
    }
}

/// Test ray-sphere intersection
pub fn test_sphere_intersection(
    ray: &Ray,
    scene_accessor: &SceneAccessor,
    sphere_index: u32,
    max_t: f32
) -> IntersectionResult {
    let sphere_center = scene_accessor.get_sphere_center(sphere_index);
    let sphere_radius = scene_accessor.get_sphere_radius(sphere_index);
    
    // Direct sphere intersection - AABB test removed as it's redundant for spheres
    let oc = ray.origin - sphere_center;
    let a = ray.direction.dot(ray.direction);
    let b = 2.0 * oc.dot(ray.direction);
    let c = oc.dot(oc) - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;
    
    if discriminant < 0.0 {
        return IntersectionResult::miss();
    }

    let sqrt_discriminant = discriminant.sqrt();
    let t1 = (-b - sqrt_discriminant) / (2.0 * a);
    let t2 = (-b + sqrt_discriminant) / (2.0 * a);
    
    let t = if t1 > RaytracerConfig::MIN_RAY_DISTANCE { t1 } else { t2 };
    
    if t > RaytracerConfig::MIN_RAY_DISTANCE && t < max_t {
        let hit_point = ray.at(t);
        let hit_normal = (hit_point - sphere_center).normalize();
        let material_id = scene_accessor.get_sphere_material_id(sphere_index);
        
        IntersectionResult::hit(Intersection::new(t, hit_point, hit_normal, material_id))
    } else {
        IntersectionResult::miss()
    }
}

/// Test ray-triangle intersection using Möller-Trumbore algorithm
/// Optimized version that takes vertex positions directly to avoid intermediate struct allocation
pub fn test_triangle_intersection_direct(
    ray: &Ray,
    v0: [f32; 3],
    v1: [f32; 3], 
    v2: [f32; 3],
    material_id: u32,
    max_t: f32
) -> IntersectionResult {
    let v0_vec = vec3(v0[0], v0[1], v0[2]);
    let v1_vec = vec3(v1[0], v1[1], v1[2]);
    let v2_vec = vec3(v2[0], v2[1], v2[2]);

    // Möller-Trumbore ray-triangle intersection
    let edge1 = v1_vec - v0_vec;
    let edge2 = v2_vec - v0_vec;
    let h = ray.direction.cross(edge2);
    let a = edge1.dot(h);

    if a.abs() < RaytracerConfig::MIN_RAY_DISTANCE {
        return IntersectionResult::miss();
    }

    let f = 1.0 / a;
    let s = ray.origin - v0_vec;
    let u = f * s.dot(h);

    if u < 0.0 || u > 1.0 {
        return IntersectionResult::miss();
    }

    let q = s.cross(edge1);
    let v = f * ray.direction.dot(q);

    if v < 0.0 || u + v > 1.0 {
        return IntersectionResult::miss();
    }

    let t = f * edge2.dot(q);

    if t > RaytracerConfig::MIN_RAY_DISTANCE && t < max_t {
        let hit_point = ray.at(t);
        let hit_normal = edge1.cross(edge2).normalize();
        
        IntersectionResult::hit(Intersection::new(t, hit_point, hit_normal, material_id))
    } else {
        IntersectionResult::miss()
    }
}

/// Test ray-triangle intersection using Möller-Trumbore algorithm
/// Legacy version for backward compatibility with ExpandedTriangle
pub fn test_triangle_intersection(
    ray: &Ray,
    triangle: &ExpandedTriangle,
    max_t: f32
) -> IntersectionResult {
    test_triangle_intersection_direct(ray, triangle.v0, triangle.v1, triangle.v2, triangle.material_id, max_t)
}

/// Ray-AABB intersection test for BVH traversal
pub fn ray_aabb_intersect(ray_origin: Vec3, ray_dir: Vec3, aabb_min: Vec3, aabb_max: Vec3) -> bool {
    let inv_dir = vec3(1.0 / ray_dir.x, 1.0 / ray_dir.y, 1.0 / ray_dir.z);
    
    let t1 = (aabb_min - ray_origin) * inv_dir;
    let t2 = (aabb_max - ray_origin) * inv_dir;
    
    let tmin = t1.min(t2);
    let tmax = t1.max(t2);
    
    let tmin_max = tmin.x.max(tmin.y).max(tmin.z);
    let tmax_min = tmax.x.min(tmax.y).min(tmax.z);
    
    tmax_min >= 0.0 && tmin_max <= tmax_min
}