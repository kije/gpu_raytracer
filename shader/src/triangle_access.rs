use raytracer_shared::{Triangle, PushConstants, Vertex};
use crate::scene_access::SceneAccessor;

/// Shared triangle accessor utilities to avoid code duplication
pub struct TriangleAccessor;

/// Triangle with vertex positions expanded (for compatibility with intersection code)
pub struct ExpandedTriangle {
    pub v0: [f32; 3],
    pub v1: [f32; 3], 
    pub v2: [f32; 3],
    pub material_id: u32,
}

impl TriangleAccessor {
    /// Get triangle from multiple buffers with GPU-friendly result
    /// Returns an expanded triangle with vertex positions resolved from the vertex buffer
    pub fn get_triangle_from_buffers(
        index: u32,
        triangles_buffer_0: &[Triangle],
        triangles_buffer_1: &[Triangle],
        triangles_buffer_2: &[Triangle],
        push_constants: &PushConstants,
        scene_accessor: &SceneAccessor
    ) -> (bool, ExpandedTriangle) {
        let buffer_index = index / push_constants.triangles_per_buffer;
        let local_index = (index % push_constants.triangles_per_buffer) as usize;
        
        // Use a constant for the default triangle to avoid repeated construction
        const DEFAULT_TRIANGLE: ExpandedTriangle = ExpandedTriangle {
            v0: [0.0; 3],
            v1: [0.0; 3], 
            v2: [0.0; 3],
            material_id: 0,
        };

        // GPU-friendly triangle lookup without Option types
        let mut triangle_valid = false;
        let mut triangle = Triangle {
            v0_index: 0,
            v1_index: 0,
            v2_index: 0,
            material_id: 0,
        };

        if buffer_index == 0 && local_index < triangles_buffer_0.len() {
            triangle = triangles_buffer_0[local_index];
            triangle_valid = true;
        } else if buffer_index == 1 && local_index < triangles_buffer_1.len() {
            triangle = triangles_buffer_1[local_index];
            triangle_valid = true;
        } else if buffer_index == 2 && local_index < triangles_buffer_2.len() {
            triangle = triangles_buffer_2[local_index];
            triangle_valid = true;
        }

        if triangle_valid {
            // Get vertex positions from scene metadata buffer
            let v0 = scene_accessor.get_vertex_position(triangle.v0_index);
            let v1 = scene_accessor.get_vertex_position(triangle.v1_index);
            let v2 = scene_accessor.get_vertex_position(triangle.v2_index);
            
            (true, ExpandedTriangle {
                v0,
                v1,
                v2,
                material_id: triangle.material_id,
            })
        } else {
            (false, DEFAULT_TRIANGLE)
        }
    }
}