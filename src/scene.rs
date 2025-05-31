use raytracer_shared::{Camera, Sphere, Triangle, Material, Light, TextureInfo, SceneBuilder};
use crate::gltf_loader::{GltfLoader, GltfError};

/// Scene geometry and camera
pub struct SceneState {
    pub camera: Camera,
    pub spheres: Vec<Sphere>,
    pub triangles: Vec<Triangle>,
    pub materials: Vec<Material>,
    pub lights: Vec<Light>,
    pub textures: Vec<TextureInfo>,
    pub texture_data: Vec<u8>,
}

impl SceneState {
    pub fn new() -> Self {
        let (spheres, triangles, materials) = SceneBuilder::build_default_scene();
        
        Self {
            camera: Camera::new(),
            spheres,
            triangles,
            materials,
            lights: Vec::new(),
            textures: Vec::new(),
            texture_data: Vec::new(),
        }
    }

    /// Load scene from glTF file
    pub fn load_from_gltf<P: AsRef<std::path::Path>>(path: P) -> Result<Self, GltfError> {
        let loader = GltfLoader::load_from_path(&path)?;
        let loaded_scene = loader.extract_scene(None)?;
        
        // Use the first camera from glTF if available, otherwise default camera
        let camera = loaded_scene.cameras.first().copied().unwrap_or_else(Camera::new);
        
        Ok(Self {
            camera,
            spheres: loaded_scene.spheres,
            triangles: loaded_scene.triangles,
            materials: loaded_scene.materials,
            lights: loaded_scene.lights,
            textures: loaded_scene.textures,
            texture_data: loaded_scene.texture_data,
        })
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
        self.materials = loaded_scene.materials;
        self.lights = loaded_scene.lights;
        self.textures = loaded_scene.textures;
        self.texture_data = loaded_scene.texture_data;
        
        println!("Loaded glTF scene: {} spheres, {} triangles, {} materials, {} lights, {} textures",
                 self.spheres.len(), self.triangles.len(), self.materials.len(),
                 self.lights.len(), self.textures.len());
        
        Ok(())
    }
}