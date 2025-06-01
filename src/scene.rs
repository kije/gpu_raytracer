use raytracer_shared::{Camera, Sphere, Triangle, Vertex, Material, Light, TextureInfo, SceneBuilder, BvhNode};
use crate::gltf_loader::{GltfLoader, GltfError};
use crate::bvh::{BvhBuilder, BvhResult};

/// Scene geometry and camera
pub struct SceneState {
    pub camera: Camera,
    pub spheres: Vec<Sphere>,
    pub triangles: Vec<Triangle>,
    pub vertices: Vec<Vertex>,
    pub materials: Vec<Material>,
    pub lights: Vec<Light>,
    pub textures: Vec<TextureInfo>,
    pub texture_data: Vec<u8>,
    pub bvh_nodes: Vec<BvhNode>,
    pub triangle_indices: Vec<u32>,
}

impl SceneState {
    pub fn new() -> Self {
        let (spheres, triangles, vertices, materials, lights) = SceneBuilder::build_default_scene();
        let BvhResult { nodes, triangle_indices } = BvhBuilder::build(&triangles, &vertices);
        
        let scene = Self {
            camera: Camera::new(),
            spheres,
            triangles,
            vertices,
            materials,
            lights,
            textures: Vec::new(),
            texture_data: Vec::new(),
            bvh_nodes: nodes,
            triangle_indices,
        };
        
        // Print memory usage for default scene
        scene.print_memory_usage("Default Scene");
        scene
    }

    /// Load scene from glTF file
    pub fn load_from_gltf<P: AsRef<std::path::Path>>(path: P) -> Result<Self, GltfError> {
        let loader = GltfLoader::load_from_path(&path)?;
        let loaded_scene = loader.extract_scene(None)?;
        
        // Use the first camera from glTF if available, otherwise default camera
        let camera = loaded_scene.cameras.first().copied().unwrap_or_else(Camera::new);
        
        // Build BVH for triangles
        let BvhResult { nodes, triangle_indices } = BvhBuilder::build(&loaded_scene.triangles, &loaded_scene.vertices);
        
        let scene = Self {
            camera,
            spheres: loaded_scene.spheres,
            triangles: loaded_scene.triangles,
            vertices: loaded_scene.vertices,
            materials: loaded_scene.materials,
            lights: loaded_scene.lights,
            textures: loaded_scene.textures,
            texture_data: loaded_scene.texture_data,
            bvh_nodes: nodes,
            triangle_indices,
        };
        
        // Print memory usage for loaded glTF scene
        scene.print_memory_usage("glTF Scene");
        Ok(scene)
    }

    /// Load scene from glTF file with fallback to default scene
    pub fn load_from_gltf_or_default<P: AsRef<std::path::Path>>(path: P) -> Self {
        match Self::load_from_gltf(path.as_ref()) {
            Ok(scene) => {
                println!("Successfully loaded glTF scene from: {:?}", path.as_ref());
                scene
            }
            Err(e) => {
                println!("Failed to load glTF scene from {:?}, using default scene. Error: {:?}", 
                         path.as_ref(), e);
                Self::new()
            }
        }
    }

    /// Replace current scene with glTF data
    pub fn replace_with_gltf<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<(), GltfError> {
        let loader = GltfLoader::load_from_path(&path)?;
        let loaded_scene = loader.extract_scene(None)?;
        
        // Use the first camera from glTF if available, otherwise keep current camera
        if let Some(new_camera) = loaded_scene.cameras.first() {
            self.camera = *new_camera;
            println!("Loaded camera from glTF: pos={:?}, dir={:?}, fov={}", 
                     self.camera.position, self.camera.direction, self.camera.fov);
        }
        
        self.spheres = loaded_scene.spheres;
        self.triangles = loaded_scene.triangles;
        self.vertices = loaded_scene.vertices;
        self.materials = loaded_scene.materials;
        self.lights = loaded_scene.lights;
        self.textures = loaded_scene.textures;
        self.texture_data = loaded_scene.texture_data;
        
        // Rebuild BVH for new triangles
        let BvhResult { nodes, triangle_indices } = BvhBuilder::build(&self.triangles, &self.vertices);
        self.bvh_nodes = nodes;
        self.triangle_indices = triangle_indices;
        
        println!("Loaded glTF scene: {} spheres, {} triangles, {} materials, {} lights, {} textures, {} BVH nodes",
                 self.spheres.len(), self.triangles.len(), self.materials.len(),
                 self.lights.len(), self.textures.len(), self.bvh_nodes.len());
        
        // Print memory usage for replaced scene
        self.print_memory_usage("Replaced glTF Scene");
        
        Ok(())
    }
    
    /// Rebuild BVH when triangles change
    pub fn rebuild_bvh(&mut self) {
        let BvhResult { nodes, triangle_indices } = BvhBuilder::build(&self.triangles, &self.vertices);
        self.bvh_nodes = nodes;
        self.triangle_indices = triangle_indices;
        println!("Rebuilt BVH: {} nodes, {} triangle indices", self.bvh_nodes.len(), self.triangle_indices.len());
    }
    
    /// Print detailed memory usage breakdown for scene data
    pub fn print_memory_usage(&self, scene_name: &str) {
        use std::mem::size_of;
        
        // Calculate memory usage for each component
        let spheres_bytes = self.spheres.len() * size_of::<Sphere>();
        let triangles_bytes = self.triangles.len() * size_of::<Triangle>();
        let vertices_bytes = self.vertices.len() * size_of::<Vertex>();
        let materials_bytes = self.materials.len() * size_of::<Material>();
        let lights_bytes = self.lights.len() * size_of::<Light>();
        let textures_bytes = self.textures.len() * size_of::<TextureInfo>();
        let texture_data_bytes = self.texture_data.len();
        let bvh_nodes_bytes = self.bvh_nodes.len() * size_of::<BvhNode>();
        let triangle_indices_bytes = self.triangle_indices.len() * size_of::<u32>();
        let camera_bytes = size_of::<Camera>();
        
        let total_bytes = spheres_bytes + triangles_bytes + vertices_bytes + materials_bytes + 
                         lights_bytes + textures_bytes + texture_data_bytes + bvh_nodes_bytes + 
                         triangle_indices_bytes + camera_bytes;
        
        // Helper function to format bytes as KB
        let to_kb = |bytes: usize| bytes as f64 / 1024.0;
        
        println!("\n=== 📊 {} Memory Usage Breakdown ===", scene_name);
        println!("┌─────────────────────┬───────────┬──────────┬─────────┐");
        println!("│ Component           │ Count     │ Size KB  │ % Total │");
        println!("├─────────────────────┼───────────┼──────────┼─────────┤");
        println!("│ Triangles          │ {:>9} │ {:>8.1} │ {:>6.1}% │", 
                 self.triangles.len(), to_kb(triangles_bytes), 
                 (triangles_bytes as f64 / total_bytes as f64) * 100.0);
        println!("│ Vertices           │ {:>9} │ {:>8.1} │ {:>6.1}% │", 
                 self.vertices.len(), to_kb(vertices_bytes), 
                 (vertices_bytes as f64 / total_bytes as f64) * 100.0);
        println!("│ Triangle Indices   │ {:>9} │ {:>8.1} │ {:>6.1}% │", 
                 self.triangle_indices.len(), to_kb(triangle_indices_bytes), 
                 (triangle_indices_bytes as f64 / total_bytes as f64) * 100.0);
        println!("│ BVH Nodes          │ {:>9} │ {:>8.1} │ {:>6.1}% │", 
                 self.bvh_nodes.len(), to_kb(bvh_nodes_bytes), 
                 (bvh_nodes_bytes as f64 / total_bytes as f64) * 100.0);
        println!("│ Materials          │ {:>9} │ {:>8.1} │ {:>6.1}% │", 
                 self.materials.len(), to_kb(materials_bytes), 
                 (materials_bytes as f64 / total_bytes as f64) * 100.0);
        println!("│ Spheres            │ {:>9} │ {:>8.1} │ {:>6.1}% │", 
                 self.spheres.len(), to_kb(spheres_bytes), 
                 (spheres_bytes as f64 / total_bytes as f64) * 100.0);
        println!("│ Lights             │ {:>9} │ {:>8.1} │ {:>6.1}% │", 
                 self.lights.len(), to_kb(lights_bytes), 
                 (lights_bytes as f64 / total_bytes as f64) * 100.0);
        println!("│ Textures (meta)    │ {:>9} │ {:>8.1} │ {:>6.1}% │", 
                 self.textures.len(), to_kb(textures_bytes), 
                 (textures_bytes as f64 / total_bytes as f64) * 100.0);
        println!("│ Texture Data       │ {:>9} │ {:>8.1} │ {:>6.1}% │", 
                 self.texture_data.len(), to_kb(texture_data_bytes), 
                 (texture_data_bytes as f64 / total_bytes as f64) * 100.0);
        println!("│ Camera             │ {:>9} │ {:>8.1} │ {:>6.1}% │", 
                 1, to_kb(camera_bytes), 
                 (camera_bytes as f64 / total_bytes as f64) * 100.0);
        println!("├─────────────────────┼───────────┼──────────┼─────────┤");
        println!("│ 🎯 TOTAL SCENE     │ {:>9} │ {:>8.1} │ {:>6.1}% │", 
                 "items", to_kb(total_bytes), 100.0);
        println!("└─────────────────────┴───────────┴──────────┴─────────┘");
        
        // Additional memory efficiency insights
        if !self.triangles.is_empty() && !self.vertices.is_empty() {
            let vertex_dedup_ratio = self.vertices.len() as f64 / (self.triangles.len() * 3) as f64;
            let memory_savings = (1.0 - vertex_dedup_ratio) * 100.0;
            println!("💡 Vertex deduplication: {:.1}% memory savings ({} unique vertices vs {} if duplicated)", 
                     memory_savings, self.vertices.len(), self.triangles.len() * 3);
        }
        
        if !self.bvh_nodes.is_empty() {
            let bvh_overhead = (bvh_nodes_bytes as f64 / triangles_bytes as f64) * 100.0;
            println!("⚡ BVH acceleration overhead: {:.1}% of triangle memory ({} nodes for {} triangles)", 
                     bvh_overhead, self.bvh_nodes.len(), self.triangles.len());
        }
        
        println!();
    }
}