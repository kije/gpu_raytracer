use raytracer_shared::{Triangle, PushConstants};

/// Shared triangle accessor utilities to avoid code duplication
pub struct TriangleAccessor;

impl TriangleAccessor {
    /// Get triangle from multiple buffers with GPU-friendly result
    pub fn get_triangle_from_buffers(
        index: u32,
        triangles_buffer_0: &[Triangle],
        triangles_buffer_1: &[Triangle],
        triangles_buffer_2: &[Triangle],
        push_constants: &PushConstants
    ) -> (bool, Triangle) {
        let buffer_index = index / push_constants.triangles_per_buffer;
        let local_index = (index % push_constants.triangles_per_buffer) as usize;
        
        // Use a constant for the default triangle to avoid repeated construction
        const DEFAULT_TRIANGLE: Triangle = Triangle {
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
                    (false, DEFAULT_TRIANGLE)
                }
            },
            1 => {
                if local_index < triangles_buffer_1.len() {
                    (true, triangles_buffer_1[local_index])
                } else {
                    (false, DEFAULT_TRIANGLE)
                }
            },
            2 => {
                if local_index < triangles_buffer_2.len() {
                    (true, triangles_buffer_2[local_index])
                } else {
                    (false, DEFAULT_TRIANGLE)
                }
            },
            _ => (false, DEFAULT_TRIANGLE)
        }
    }
}