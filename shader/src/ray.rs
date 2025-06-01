use spirv_std::glam::{vec2, vec3, Vec3, UVec2};
use spirv_std::num_traits::Float;
use raytracer_shared::PushConstants;

/// Ray representation for raytracing
#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    /// Create a new ray
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Create ray from screen coordinates
    pub fn from_screen_coordinates(pixel_coords: UVec2, push_constants: &PushConstants) -> Self {
        let width = push_constants.resolution[0] as u32;
        let height = push_constants.resolution[1] as u32;

        // Convert screen coordinates to camera ray with full precision
        let uv = vec2(
            (pixel_coords.x as f32 + 0.5) / width as f32,
            (pixel_coords.y as f32 + 0.5) / height as f32
        );

        // Convert UV coordinates to camera space with full f32 precision
        let aspect_ratio = width as f32 / height as f32;
        let fov_scale = (push_constants.camera.fov * 0.5 * core::f32::consts::PI / 180.0).tan();

        // Calculate camera space coordinates directly with f32 precision
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

        let origin = vec3(push_constants.camera.position[0], push_constants.camera.position[1], push_constants.camera.position[2]);

        Self::new(origin, ray_direction_normalized)
    }

    /// Get point along ray at parameter t
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}