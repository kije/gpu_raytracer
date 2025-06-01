use spirv_std::float::{f32_to_f16, f16_to_f32, vec2_to_f16x2};
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

        // Convert screen coordinates to camera ray using f16 for better memory efficiency
        let uv_f32 = vec2(
            (pixel_coords.x as f32 + 0.5) / width as f32,
            (pixel_coords.y as f32 + 0.5) / height as f32
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

        let origin = vec3(push_constants.camera.position[0], push_constants.camera.position[1], push_constants.camera.position[2]);

        Self::new(origin, ray_direction_normalized)
    }

    /// Get point along ray at parameter t
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}