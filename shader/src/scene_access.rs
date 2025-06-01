use spirv_std::glam::{vec3, Vec3};
use raytracer_shared::{Sphere, Light, BvhNode, Vertex, PushConstants};

/// Provides access to scene data stored in the combined metadata buffer
pub struct SceneAccessor<'a> {
    scene_metadata: &'a [u32],
    push_constants: &'a PushConstants,
}

impl<'a> SceneAccessor<'a> {
    pub fn new(scene_metadata: &'a [u32], push_constants: &'a PushConstants) -> Self {
        Self {
            scene_metadata,
            push_constants,
        }
    }

    /// Get number of spheres in the scene
    pub fn sphere_count(&self) -> u32 {
        self.push_constants.metadata_offsets.spheres_count
    }

    /// Get number of lights in the scene
    pub fn light_count(&self) -> u32 {
        self.push_constants.metadata_offsets.lights_count
    }

    /// Get number of BVH nodes
    pub fn bvh_node_count(&self) -> u32 {
        self.push_constants.metadata_offsets.bvh_nodes_count
    }

    /// Get sphere center
    pub fn get_sphere_center(&self, sphere_index: u32) -> Vec3 {
        let base_offset = self.push_constants.metadata_offsets.spheres_offset as usize;
        let sphere_offset_u32 = base_offset + (sphere_index as usize * (core::mem::size_of::<Sphere>() / 4));
        
        vec3(
            f32::from_bits(self.scene_metadata[sphere_offset_u32]),
            f32::from_bits(self.scene_metadata[sphere_offset_u32 + 1]),
            f32::from_bits(self.scene_metadata[sphere_offset_u32 + 2]),
        )
    }
    
    /// Get sphere radius
    pub fn get_sphere_radius(&self, sphere_index: u32) -> f32 {
        let base_offset = self.push_constants.metadata_offsets.spheres_offset as usize;
        let sphere_offset_u32 = base_offset + (sphere_index as usize * (core::mem::size_of::<Sphere>() / 4));
        f32::from_bits(self.scene_metadata[sphere_offset_u32 + 3])
    }
    
    /// Get sphere material ID
    pub fn get_sphere_material_id(&self, sphere_index: u32) -> u32 {
        let base_offset = self.push_constants.metadata_offsets.spheres_offset as usize;
        let sphere_offset_u32 = base_offset + (sphere_index as usize * (core::mem::size_of::<Sphere>() / 4));
        self.scene_metadata[sphere_offset_u32 + 4]
    }
    
    /// Get light position
    pub fn get_light_position(&self, light_index: u32) -> Vec3 {
        let base_offset = self.push_constants.metadata_offsets.lights_offset as usize;
        let light_offset_u32 = base_offset + (light_index as usize * (core::mem::size_of::<Light>() / 4));
        
        vec3(
            f32::from_bits(self.scene_metadata[light_offset_u32]),
            f32::from_bits(self.scene_metadata[light_offset_u32 + 1]),
            f32::from_bits(self.scene_metadata[light_offset_u32 + 2]),
        )
    }
    
    /// Get light type
    pub fn get_light_type(&self, light_index: u32) -> u32 {
        let base_offset = self.push_constants.metadata_offsets.lights_offset as usize;
        let light_offset_u32 = base_offset + (light_index as usize * (core::mem::size_of::<Light>() / 4));
        self.scene_metadata[light_offset_u32 + 3]
    }
    
    /// Get light color
    pub fn get_light_color(&self, light_index: u32) -> Vec3 {
        let base_offset = self.push_constants.metadata_offsets.lights_offset as usize;
        let light_offset_u32 = base_offset + (light_index as usize * (core::mem::size_of::<Light>() / 4));
        
        vec3(
            f32::from_bits(self.scene_metadata[light_offset_u32 + 4]),
            f32::from_bits(self.scene_metadata[light_offset_u32 + 5]),
            f32::from_bits(self.scene_metadata[light_offset_u32 + 6]),
        )
    }
    
    /// Get light intensity
    pub fn get_light_intensity(&self, light_index: u32) -> f32 {
        let base_offset = self.push_constants.metadata_offsets.lights_offset as usize;
        let light_offset_u32 = base_offset + (light_index as usize * (core::mem::size_of::<Light>() / 4));
        f32::from_bits(self.scene_metadata[light_offset_u32 + 7])
    }
    
    /// Get light direction
    pub fn get_light_direction(&self, light_index: u32) -> Vec3 {
        let base_offset = self.push_constants.metadata_offsets.lights_offset as usize;
        let light_offset_u32 = base_offset + (light_index as usize * (core::mem::size_of::<Light>() / 4));
        
        vec3(
            f32::from_bits(self.scene_metadata[light_offset_u32 + 8]),
            f32::from_bits(self.scene_metadata[light_offset_u32 + 9]),
            f32::from_bits(self.scene_metadata[light_offset_u32 + 10]),
        )
    }
    
    /// Get BVH node bounds minimum
    pub fn get_bvh_node_bounds_min(&self, node_index: u32) -> Vec3 {
        let base_offset = self.push_constants.metadata_offsets.bvh_nodes_offset as usize;
        let node_offset_u32 = base_offset + (node_index as usize * (core::mem::size_of::<BvhNode>() / 4));
        
        vec3(
            f32::from_bits(self.scene_metadata[node_offset_u32]),
            f32::from_bits(self.scene_metadata[node_offset_u32 + 1]),
            f32::from_bits(self.scene_metadata[node_offset_u32 + 2]),
        )
    }
    
    /// Get BVH node bounds maximum
    pub fn get_bvh_node_bounds_max(&self, node_index: u32) -> Vec3 {
        let base_offset = self.push_constants.metadata_offsets.bvh_nodes_offset as usize;
        let node_offset_u32 = base_offset + (node_index as usize * (core::mem::size_of::<BvhNode>() / 4));
        
        vec3(
            f32::from_bits(self.scene_metadata[node_offset_u32 + 4]),
            f32::from_bits(self.scene_metadata[node_offset_u32 + 5]),
            f32::from_bits(self.scene_metadata[node_offset_u32 + 6]),
        )
    }
    
    /// Get BVH node left child
    pub fn get_bvh_node_left_child(&self, node_index: u32) -> u32 {
        let base_offset = self.push_constants.metadata_offsets.bvh_nodes_offset as usize;
        let node_offset_u32 = base_offset + (node_index as usize * (core::mem::size_of::<BvhNode>() / 4));
        self.scene_metadata[node_offset_u32 + 8]
    }
    
    /// Get BVH node right child
    pub fn get_bvh_node_right_child(&self, node_index: u32) -> u32 {
        let base_offset = self.push_constants.metadata_offsets.bvh_nodes_offset as usize;
        let node_offset_u32 = base_offset + (node_index as usize * (core::mem::size_of::<BvhNode>() / 4));
        self.scene_metadata[node_offset_u32 + 9]
    }
    
    /// Get BVH node triangle start index
    pub fn get_bvh_node_triangle_start(&self, node_index: u32) -> u32 {
        let base_offset = self.push_constants.metadata_offsets.bvh_nodes_offset as usize;
        let node_offset_u32 = base_offset + (node_index as usize * (core::mem::size_of::<BvhNode>() / 4));
        self.scene_metadata[node_offset_u32 + 10]
    }
    
    /// Get BVH node triangle count
    pub fn get_bvh_node_triangle_count(&self, node_index: u32) -> u32 {
        let base_offset = self.push_constants.metadata_offsets.bvh_nodes_offset as usize;
        let node_offset_u32 = base_offset + (node_index as usize * (core::mem::size_of::<BvhNode>() / 4));
        self.scene_metadata[node_offset_u32 + 11]
    }
    
    /// Get triangle index from triangle indices buffer
    pub fn get_triangle_index(&self, index: u32) -> u32 {
        let base_offset = self.push_constants.metadata_offsets.triangle_indices_offset as usize;
        let triangle_index_offset = base_offset + (index as usize);
        self.scene_metadata[triangle_index_offset]
    }
    
    /// Get vertex position from vertices buffer
    pub fn get_vertex_position(&self, vertex_index: u32) -> [f32; 3] {
        let base_offset = self.push_constants.metadata_offsets.vertices_offset as usize;
        let vertex_offset_u32 = base_offset + (vertex_index as usize * (core::mem::size_of::<Vertex>() / 4));
        
        [
            f32::from_bits(self.scene_metadata[vertex_offset_u32]),
            f32::from_bits(self.scene_metadata[vertex_offset_u32 + 1]),
            f32::from_bits(self.scene_metadata[vertex_offset_u32 + 2]),
        ]
    }
}