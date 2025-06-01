use raytracer_shared::{Triangle, PushConstants};
use crate::ray::Ray;
use crate::intersection::{IntersectionResult, ray_aabb_intersect, test_triangle_intersection};
use crate::scene_access::SceneAccessor;
use crate::triangle_access::TriangleAccessor;

/// BVH traverser for accelerated triangle intersection
pub struct BvhTraverser<'a> {
    scene_accessor: &'a SceneAccessor<'a>,
}

impl<'a> BvhTraverser<'a> {
    pub fn new(scene_accessor: &'a SceneAccessor<'a>) -> Self {
        Self { scene_accessor }
    }

    /// Traverse BVH and find closest triangle intersection
    pub fn traverse_and_intersect(
        &self,
        ray: &Ray,
        triangles_buffer_0: &[Triangle],
        triangles_buffer_1: &[Triangle],
        triangles_buffer_2: &[Triangle],
        push_constants: &PushConstants,
        max_t: f32
    ) -> IntersectionResult {
        if self.scene_accessor.bvh_node_count() == 0 {
            return IntersectionResult::miss();
        }

        let mut result = IntersectionResult::miss();
        let mut closest_t = max_t;

        // Stack for BVH traversal (GPU-friendly - fixed size)
        let mut node_stack: [u32; 64] = [0xFFFFFFFF; 64];
        let mut stack_ptr = 0;
        node_stack[0] = 0; // Start with root node
        stack_ptr += 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let current_node = node_stack[stack_ptr];
            
            if current_node == 0xFFFFFFFF || current_node >= self.scene_accessor.bvh_node_count() {
                continue;
            }

            // Test current node AABB
            let bounds_min = self.scene_accessor.get_bvh_node_bounds_min(current_node);
            let bounds_max = self.scene_accessor.get_bvh_node_bounds_max(current_node);
            
            if !ray_aabb_intersect(ray.origin, ray.direction, bounds_min, bounds_max) {
                continue;
            }

            // Check if leaf node
            let left_child = self.scene_accessor.get_bvh_node_left_child(current_node);
            let right_child = self.scene_accessor.get_bvh_node_right_child(current_node);
            
            if left_child == 0xFFFFFFFF { // Leaf node
                let leaf_result = self.test_leaf_triangles(
                    ray,
                    current_node,
                    triangles_buffer_0,
                    triangles_buffer_1,
                    triangles_buffer_2,
                    push_constants,
                    closest_t
                );
                if leaf_result.hit {
                    closest_t = leaf_result.intersection.t;
                    result = leaf_result;
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

        result
    }

    /// Test all triangles in a BVH leaf node
    fn test_leaf_triangles(
        &self,
        ray: &Ray,
        node_index: u32,
        triangles_buffer_0: &[Triangle],
        triangles_buffer_1: &[Triangle],
        triangles_buffer_2: &[Triangle],
        push_constants: &PushConstants,
        max_t: f32
    ) -> IntersectionResult {
        let triangle_start = self.scene_accessor.get_bvh_node_triangle_start(node_index);
        let triangle_count = self.scene_accessor.get_bvh_node_triangle_count(node_index);
        
        let mut result = IntersectionResult::miss();
        let mut closest_t = max_t;

        // Test all triangles in this leaf
        for i in 0..triangle_count {
            if triangle_start + i >= push_constants.metadata_offsets.triangle_indices_count {
                break;
            }
            
            let triangle_index = self.scene_accessor.get_triangle_index(triangle_start + i);
            
            let (triangle_valid, triangle) = TriangleAccessor::get_triangle_from_buffers(
                triangle_index,
                triangles_buffer_0,
                triangles_buffer_1,
                triangles_buffer_2,
                push_constants
            );
            if triangle_valid {
                let test_result = test_triangle_intersection(ray, &triangle, closest_t);
                if test_result.hit {
                    closest_t = test_result.intersection.t;
                    result = test_result;
                }
            }
        }

        result
    }

}