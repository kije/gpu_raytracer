use bvh::{
    aabb::{Aabb as BvhAabb, Bounded},
    bounding_hierarchy::BHShape,
    bvh::Bvh as BVH,
};
use raytracer_shared::{Triangle, Sphere, Aabb, BvhNode};

// Type aliases for specific f32 3D types
type Point3f = nalgebra::Point3<f32>;
type BvhAabbf = BvhAabb<f32, 3>;
type BVHf = BVH<f32, 3>;

/// Wrapper for Triangle to implement BHShape trait for BVH construction
#[derive(Clone, Copy, Debug)]
pub struct BvhTriangle {
    pub triangle: Triangle,
    pub node_index: usize,
}

impl BvhTriangle {
    pub fn new(triangle: Triangle, node_index: usize) -> Self {
        Self { triangle, node_index }
    }
    
    /// Get the center of the triangle
    pub fn centroid(&self) -> Point3f {
        let v0 = Point3f::new(self.triangle.v0[0], self.triangle.v0[1], self.triangle.v0[2]);
        let v1 = Point3f::new(self.triangle.v1[0], self.triangle.v1[1], self.triangle.v1[2]);
        let v2 = Point3f::new(self.triangle.v2[0], self.triangle.v2[1], self.triangle.v2[2]);
        
        Point3f::new(
            (v0.x + v1.x + v2.x) / 3.0,
            (v0.y + v1.y + v2.y) / 3.0,
            (v0.z + v1.z + v2.z) / 3.0,
        )
    }
}

impl Bounded<f32, 3> for BvhTriangle {
    fn aabb(&self) -> BvhAabbf {
        let v0 = Point3f::new(self.triangle.v0[0], self.triangle.v0[1], self.triangle.v0[2]);
        let v1 = Point3f::new(self.triangle.v1[0], self.triangle.v1[1], self.triangle.v1[2]);
        let v2 = Point3f::new(self.triangle.v2[0], self.triangle.v2[1], self.triangle.v2[2]);
        
        let min = Point3f::new(
            v0.x.min(v1.x).min(v2.x),
            v0.y.min(v1.y).min(v2.y),
            v0.z.min(v1.z).min(v2.z),
        );
        
        let max = Point3f::new(
            v0.x.max(v1.x).max(v2.x),
            v0.y.max(v1.y).max(v2.y),
            v0.z.max(v1.z).max(v2.z),
        );
        
        BvhAabbf::with_bounds(min, max)
    }
}

impl BHShape<f32, 3> for BvhTriangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
    
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

/// BVH builder for triangle acceleration
pub struct BvhBuilder {
    bvh_triangles: Vec<BvhTriangle>,
    triangle_indices: Vec<u32>,
}

impl BvhBuilder {
    /// Maximum triangles per leaf node to reduce memory usage
    const MAX_TRIANGLES_PER_LEAF: usize = 8;
    
    /// Create a new BVH builder
    pub fn new() -> Self {
        Self {
            bvh_triangles: Vec::new(),
            triangle_indices: Vec::new(),
        }
    }
    
    /// Build BVH from triangles with optimized leaf sizes
    pub fn build(triangles: &[Triangle]) -> BvhResult {
        if triangles.is_empty() {
            return BvhResult {
                nodes: vec![BvhNode::leaf(
                    Aabb::empty(),
                    0,
                    0,
                )],
                triangle_indices: Vec::new(),
            };
        }
        
        // Use chunked approach for very large scenes
        if triangles.len() > 100_000 {
            Self::build_chunked(triangles)
        } else {
            Self::build_standard(triangles)
        }
    }
    
    /// Standard BVH build for smaller scenes
    fn build_standard(triangles: &[Triangle]) -> BvhResult {
        let mut bvh_triangles: Vec<BvhTriangle> = triangles
            .iter()
            .enumerate()
            .map(|(i, triangle)| BvhTriangle::new(*triangle, i))
            .collect();
        
        // Build BVH using the bvh crate
        let bvh = BVHf::build(&mut bvh_triangles);
        
        // Convert BVH to our format
        Self::convert_bvh_nodes(&bvh, &bvh_triangles)
    }
    
    /// Chunked BVH build for large scenes to reduce memory usage
    fn build_chunked(triangles: &[Triangle]) -> BvhResult {
        // For very large scenes, create much larger leaf nodes to dramatically reduce node count
        let triangles_per_leaf = (triangles.len() / 10_000).max(32); // Aim for ~10k nodes max
        let mut nodes = Vec::new();
        let mut triangle_indices = Vec::new();
        
        // Create leaf nodes with multiple triangles each
        for (chunk_idx, chunk) in triangles.chunks(triangles_per_leaf).enumerate() {
            // Calculate bounding box for this chunk
            let mut chunk_aabb = Aabb::empty();
            for triangle in chunk {
                let tri_aabb = Self::triangle_aabb(triangle);
                chunk_aabb = Aabb::union(&chunk_aabb, &tri_aabb);
            }
            
            // Add triangle indices for this chunk
            let triangle_start = triangle_indices.len() as u32;
            let base_triangle_index = chunk_idx * triangles_per_leaf;
            for i in 0..chunk.len() {
                triangle_indices.push((base_triangle_index + i) as u32);
            }
            
            // Create leaf node for this chunk
            nodes.push(BvhNode::leaf(chunk_aabb, triangle_start, chunk.len() as u32));
        }
        
        // If we have more than one leaf, build a simple top-level BVH
        if nodes.len() > 1 {
            nodes = Self::build_simple_top_level_bvh(nodes);
        }
        
        BvhResult {
            nodes,
            triangle_indices,
        }
    }
    
    /// Build a simple binary tree over existing leaf nodes
    fn build_simple_top_level_bvh(leaf_nodes: Vec<BvhNode>) -> Vec<BvhNode> {
        if leaf_nodes.len() <= 1 {
            return leaf_nodes;
        }
        
        let mut nodes = Vec::new();
        let mut current_level = leaf_nodes;
        
        // Build tree bottom-up
        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            
            // Pair up nodes and create parents
            for chunk in current_level.chunks(2) {
                if chunk.len() == 2 {
                    // Combine two nodes
                    let combined_aabb = Aabb::union(&chunk[0].bounds, &chunk[1].bounds);
                    let left_idx = nodes.len() as u32;
                    let right_idx = left_idx + 1;
                    
                    nodes.extend_from_slice(chunk);
                    next_level.push(BvhNode::internal(combined_aabb, left_idx, right_idx));
                } else {
                    // Odd node out, promote to next level
                    let node_idx = nodes.len() as u32;
                    nodes.push(chunk[0]);
                    next_level.push(BvhNode::internal(chunk[0].bounds, node_idx, 0xFFFFFFFF));
                }
            }
            
            current_level = next_level;
        }
        
        // Add the final root
        if !current_level.is_empty() {
            nodes.push(current_level[0]);
        }
        
        // Reverse to put root first
        nodes.reverse();
        
        // Fix indices after reversal
        let offset = nodes.len() as u32 - 1;
        for node in &mut nodes {
            if !node.is_leaf() {
                if node.left_child != 0xFFFFFFFF {
                    node.left_child = offset - node.left_child;
                }
                if node.right_child != 0xFFFFFFFF {
                    node.right_child = offset - node.right_child;
                }
            }
        }
        
        nodes
    }
    
    /// Build a simple top-level BVH over chunk bounding boxes
    fn build_top_level_bvh(
        chunk_bounds: &[(Aabb, usize, usize)], 
        _nodes: &mut Vec<BvhNode>
    ) -> BvhNode {
        // For now, create a simple binary tree over chunks
        let mut combined_aabb = Aabb::empty();
        for (aabb, _, _) in chunk_bounds {
            combined_aabb = Aabb::union(&combined_aabb, aabb);
        }
        
        if chunk_bounds.len() <= 2 {
            // Create internal node pointing to first chunk roots
            let left_child = if chunk_bounds.len() > 0 { (chunk_bounds[0].1 + 1) as u32 } else { 0xFFFFFFFF };
            let right_child = if chunk_bounds.len() > 1 { (chunk_bounds[1].1 + 1) as u32 } else { 0xFFFFFFFF };
            
            BvhNode::internal(combined_aabb, left_child, right_child)
        } else {
            // For more chunks, just point to first chunk (simplified)
            BvhNode::internal(combined_aabb, (chunk_bounds[0].1 + 1) as u32, 0xFFFFFFFF)
        }
    }
    
    /// Calculate AABB for a single triangle
    fn triangle_aabb(triangle: &Triangle) -> Aabb {
        let min = [
            triangle.v0[0].min(triangle.v1[0]).min(triangle.v2[0]),
            triangle.v0[1].min(triangle.v1[1]).min(triangle.v2[1]),
            triangle.v0[2].min(triangle.v1[2]).min(triangle.v2[2]),
        ];
        let max = [
            triangle.v0[0].max(triangle.v1[0]).max(triangle.v2[0]),
            triangle.v0[1].max(triangle.v1[1]).max(triangle.v2[1]),
            triangle.v0[2].max(triangle.v1[2]).max(triangle.v2[2]),
        ];
        Aabb::new(min, max)
    }
    
    /// Convert bvh crate nodes to our BvhNode format
    fn convert_bvh_nodes(bvh: &BVHf, bvh_triangles: &[BvhTriangle]) -> BvhResult {
        let mut nodes = Vec::new();
        let mut triangle_indices = Vec::new();
        
        Self::convert_node_recursive(
            &bvh.nodes,
            0,
            bvh_triangles,
            &mut nodes,
            &mut triangle_indices,
        );
        
        BvhResult {
            nodes,
            triangle_indices,
        }
    }
    
    /// Recursively convert BVH nodes
    fn convert_node_recursive(
        bvh_nodes: &[bvh::bvh::BvhNode<f32, 3>],
        node_index: usize,
        bvh_triangles: &[BvhTriangle],
        result_nodes: &mut Vec<BvhNode>,
        triangle_indices: &mut Vec<u32>,
    ) -> u32 {
        if node_index >= bvh_nodes.len() {
            panic!("Invalid BVH node index: {}", node_index);
        }
        
        let current_result_index = result_nodes.len() as u32;
        let bvh_node = &bvh_nodes[node_index];
        
        // Convert AABB
        let bvh_aabb = bvh_node.get_node_aabb(bvh_triangles);
        let aabb = Aabb::new(
            bvh_aabb.min.coords.into(),
            bvh_aabb.max.coords.into(),
        );
        
        match bvh_node {
            bvh::bvh::BvhNode::Node {
                child_l_index,
                child_r_index,
                ..
            } => {
                // Create placeholder internal node
                result_nodes.push(BvhNode::internal(aabb, 0, 0));
                
                // Process children
                let left_index = Self::convert_node_recursive(
                    bvh_nodes,
                    *child_l_index,
                    bvh_triangles,
                    result_nodes,
                    triangle_indices,
                );
                
                let right_index = Self::convert_node_recursive(
                    bvh_nodes,
                    *child_r_index,
                    bvh_triangles,
                    result_nodes,
                    triangle_indices,
                );
                
                // Update the internal node with correct child indices
                result_nodes[current_result_index as usize] = BvhNode::internal(aabb, left_index, right_index);
            }
            bvh::bvh::BvhNode::Leaf {
                shape_index,
                ..
            } => {
                // Leaf node contains one triangle
                let triangle_start = triangle_indices.len() as u32;
                triangle_indices.push(*shape_index as u32);
                
                result_nodes.push(BvhNode::leaf(aabb, triangle_start, 1));
            }
        }
        
        current_result_index
    }
}

/// Result of BVH construction
pub struct BvhResult {
    pub nodes: Vec<BvhNode>,
    pub triangle_indices: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use raytracer_shared::Triangle;
    
    #[test]
    fn test_bvh_triangle_creation() {
        let triangle = Triangle::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            0,
        );
        
        let bvh_triangle = BvhTriangle::new(triangle, 0);
        assert_eq!(bvh_triangle.node_index, 0);
        
        let center = bvh_triangle.centroid();
        assert_eq!(center, Point3f::new(0.5, 1.0/3.0, 0.0));
    }
    
    #[test]
    fn test_bvh_triangle_bounding_box() {
        let triangle = Triangle::new(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            0,
        );
        
        let bvh_triangle = BvhTriangle::new(triangle, 0);
        let bbox = bvh_triangle.aabb();
        
        assert_eq!(bbox.min.coords, nalgebra::Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(bbox.max.coords, nalgebra::Vector3::new(2.0, 2.0, 0.0));
    }
    
    #[test]
    fn test_bvh_build_empty() {
        let triangles = Vec::new();
        let result = BvhBuilder::build(&triangles);
        
        assert_eq!(result.nodes.len(), 1);
        assert!(result.nodes[0].is_leaf());
        assert_eq!(result.nodes[0].triangle_count, 0);
        assert_eq!(result.triangle_indices.len(), 0);
    }
    
    #[test]
    fn test_bvh_build_single_triangle() {
        let triangles = vec![Triangle::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            0,
        )];
        
        let result = BvhBuilder::build(&triangles);
        
        assert_eq!(result.nodes.len(), 1);
        assert!(result.nodes[0].is_leaf());
        assert_eq!(result.nodes[0].triangle_count, 1);
        assert_eq!(result.triangle_indices.len(), 1);
        assert_eq!(result.triangle_indices[0], 0);
    }
    
    #[test]
    fn test_bvh_build_multiple_triangles() {
        let triangles = vec![
            Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], 0),
            Triangle::new([2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.5, 1.0, 0.0], 1),
            Triangle::new([4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [4.5, 1.0, 0.0], 2),
        ];
        
        let result = BvhBuilder::build(&triangles);
        
        // Should have some internal nodes for 3 triangles
        assert!(!result.nodes.is_empty());
        assert_eq!(result.triangle_indices.len(), 3);
        
        // Verify all triangle indices are present
        let mut indices = result.triangle_indices.clone();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2]);
    }
    
    #[test]
    fn test_bvh_node_bounds() {
        let triangles = vec![
            Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], 0),
            Triangle::new([2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.5, 1.0, 0.0], 1),
        ];
        
        let result = BvhBuilder::build(&triangles);
        
        // Root node should encompass all triangles
        let root = &result.nodes[0];
        assert!(root.bounds.min[0] <= 0.0);
        assert!(root.bounds.max[0] >= 3.0);
        assert!(root.bounds.min[1] <= 0.0);
        assert!(root.bounds.max[1] >= 1.0);
    }
}