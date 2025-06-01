use std::collections::HashMap;
use std::path::Path;
use gltf::{Document, Node, Primitive, Camera as GltfCamera};
use raytracer_shared::{Triangle, Vertex, Material, Sphere, Camera, Light, TextureInfo};
use image::DynamicImage;

/// Enhanced glTF loader with support for GLB, textures, lights, and advanced materials
pub struct GltfLoader {
    document: Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<gltf::image::Data>,
}

/// Error types for glTF loading
#[derive(Debug)]
pub enum GltfError {
    IoError(std::io::Error),
    GltfError(gltf::Error),
    ValidationError(String),
    ImageError(image::ImageError),
}

impl From<std::io::Error> for GltfError {
    fn from(err: std::io::Error) -> Self {
        GltfError::IoError(err)
    }
}

impl From<gltf::Error> for GltfError {
    fn from(err: gltf::Error) -> Self {
        GltfError::GltfError(err)
    }
}

impl From<image::ImageError> for GltfError {
    fn from(err: image::ImageError) -> Self {
        GltfError::ImageError(err)
    }
}

/// Enhanced loaded scene data with lights, cameras, and textures
pub struct LoadedScene {
    pub triangles: Vec<Triangle>,
    pub vertices: Vec<Vertex>,
    pub materials: Vec<Material>,
    pub spheres: Vec<Sphere>,
    pub lights: Vec<Light>,
    pub cameras: Vec<Camera>,
    pub textures: Vec<TextureInfo>,
    pub texture_data: Vec<u8>,
}

impl GltfLoader {
    /// Load a glTF or GLB file from path
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, GltfError> {
        let (document, buffers, images) = gltf::import(path)?;
        
        Ok(Self {
            document,
            buffers,
            images,
        })
    }
    
    /// Load GLB from binary data
    pub fn load_from_glb(data: &[u8]) -> Result<Self, GltfError> {
        let (document, buffers, images) = gltf::import_slice(data)?;
        
        Ok(Self {
            document,
            buffers,
            images,
        })
    }

    /// Extract comprehensive scene data for raytracing
    pub fn extract_scene(&self, scene_index: Option<usize>) -> Result<LoadedScene, GltfError> {
        let scene = if let Some(index) = scene_index {
            self.document.scenes().nth(index)
                .ok_or_else(|| GltfError::ValidationError(format!("Scene {} not found", index)))?
        } else {
            self.document.default_scene()
                .or_else(|| self.document.scenes().next())
                .ok_or_else(|| GltfError::ValidationError("No scenes found in glTF file".to_string()))?
        };

        let mut triangles = Vec::new();
        let mut vertices = Vec::new();
        let mut materials = Vec::new();
        let mut spheres = Vec::new();
        let mut lights = Vec::new();
        let mut cameras = Vec::new();
        let mut textures = Vec::new();
        let mut texture_data = Vec::new();
        let mut material_map = HashMap::new();

        // Process textures first
        self.process_textures(&mut textures, &mut texture_data)?;

        // Process materials with texture support
        for (material_index, gltf_material) in self.document.materials().enumerate() {
            let material = self.convert_material(&gltf_material, &textures)?;
            material_map.insert(material_index, materials.len() as u32);
            materials.push(material);
        }

        // Process nodes recursively
        for node in scene.nodes() {
            self.process_node(&node, &glam::Mat4::IDENTITY, &material_map, &mut triangles, &mut vertices, &mut spheres, &mut lights, &mut cameras)?;
        }

        println!("Loaded glTF scene: {} triangles, {} vertices, {} materials, {} spheres, {} lights, {} cameras, {} textures", 
                 triangles.len(), vertices.len(), materials.len(), spheres.len(), lights.len(), cameras.len(), textures.len());

        Ok(LoadedScene {
            triangles,
            vertices,
            materials,
            spheres,
            lights,
            cameras,
            textures,
            texture_data,
        })
    }

    /// Process textures and build texture data buffer
    fn process_textures(&self, textures: &mut Vec<TextureInfo>, texture_data: &mut Vec<u8>) -> Result<(), GltfError> {
        for gltf_texture in self.document.textures() {
            let image_index = gltf_texture.source().index();
            
            if image_index < self.images.len() {
                let image_data = &self.images[image_index];
                let offset = texture_data.len() as u32;
                
                // Convert image data to RGBA8
                let dynamic_image = match image_data.format {
                    gltf::image::Format::R8 => {
                        DynamicImage::ImageLuma8(
                            image::ImageBuffer::from_raw(image_data.width, image_data.height, image_data.pixels.clone())
                                .ok_or_else(|| GltfError::ValidationError("Invalid R8 image data".to_string()))?
                        )
                    }
                    gltf::image::Format::R8G8 => {
                        DynamicImage::ImageLumaA8(
                            image::ImageBuffer::from_raw(image_data.width, image_data.height, image_data.pixels.clone())
                                .ok_or_else(|| GltfError::ValidationError("Invalid R8G8 image data".to_string()))?
                        )
                    }
                    gltf::image::Format::R8G8B8 => {
                        DynamicImage::ImageRgb8(
                            image::ImageBuffer::from_raw(image_data.width, image_data.height, image_data.pixels.clone())
                                .ok_or_else(|| GltfError::ValidationError("Invalid R8G8B8 image data".to_string()))?
                        )
                    }
                    gltf::image::Format::R8G8B8A8 => {
                        DynamicImage::ImageRgba8(
                            image::ImageBuffer::from_raw(image_data.width, image_data.height, image_data.pixels.clone())
                                .ok_or_else(|| GltfError::ValidationError("Invalid R8G8B8A8 image data".to_string()))?
                        )
                    }
                    _ => {
                        return Err(GltfError::ValidationError(format!("Unsupported image format: {:?}", image_data.format)));
                    }
                };
                
                let rgba_image = dynamic_image.to_rgba8();
                let rgba_data = rgba_image.as_raw();
                
                let texture_info = TextureInfo::new(
                    image_data.width,
                    image_data.height,
                    3, // RGBA8 format
                    offset,
                    rgba_data.len() as u32,
                );
                
                textures.push(texture_info);
                texture_data.extend_from_slice(rgba_data);
            }
        }
        
        Ok(())
    }

    /// Process a single node and its children recursively
    fn process_node(
        &self,
        node: &Node,
        parent_transform: &glam::Mat4,
        material_map: &HashMap<usize, u32>,
        triangles: &mut Vec<Triangle>,
        vertices: &mut Vec<Vertex>,
        spheres: &mut Vec<Sphere>,
        lights: &mut Vec<Light>,
        cameras: &mut Vec<Camera>,
    ) -> Result<(), GltfError> {
        // Calculate node transform
        let local_transform = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
        let transform = *parent_transform * local_transform;

        // Process mesh if present
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                self.process_primitive(&primitive, &transform, material_map, triangles, vertices)?;
            }
        }

        // Process camera if present
        if let Some(gltf_camera) = node.camera() {
            let camera = self.convert_camera(&gltf_camera, &transform);
            cameras.push(camera);
        }

        // Process lights if present (KHR_lights_punctual extension)
        if let Some(gltf_light) = node.light() {
            let light = self.convert_light(&gltf_light, &transform);
            lights.push(light);
        }

        // Process children recursively
        for child in node.children() {
            self.process_node(&child, &transform, material_map, triangles, vertices, spheres, lights, cameras)?;
        }

        Ok(())
    }

    /// Convert glTF camera to raytracer camera
    fn convert_camera(&self, gltf_camera: &GltfCamera, transform: &glam::Mat4) -> Camera {
        let position = transform.transform_point3(glam::Vec3::ZERO);
        let direction = transform.transform_vector3(-glam::Vec3::Z).normalize();
        let up = transform.transform_vector3(glam::Vec3::Y).normalize();
        
        let fov = match gltf_camera.projection() {
            gltf::camera::Projection::Perspective(persp) => {
                persp.yfov().to_degrees()
            }
            gltf::camera::Projection::Orthographic(_) => {
                45.0 // Default FOV for orthographic cameras
            }
        };

        Camera {
            position: position.to_array(),
            direction: direction.to_array(),
            up: up.to_array(),
            fov,
        }
    }

    /// Convert glTF light to raytracer light (KHR_lights_punctual)
    fn convert_light(&self, gltf_light: &gltf::khr_lights_punctual::Light, transform: &glam::Mat4) -> Light {
        let position = transform.transform_point3(glam::Vec3::ZERO);
        let direction = transform.transform_vector3(-glam::Vec3::Z).normalize();
        
        let color = gltf_light.color();
        let intensity = gltf_light.intensity();
        
        match gltf_light.kind() {
            gltf::khr_lights_punctual::Kind::Directional => {
                Light::directional(direction.to_array(), color, intensity)
            }
            gltf::khr_lights_punctual::Kind::Point => {
                let range = gltf_light.range().unwrap_or(f32::INFINITY);
                Light::point(position.to_array(), color, intensity, range)
            }
            gltf::khr_lights_punctual::Kind::Spot { 
                inner_cone_angle, 
                outer_cone_angle 
            } => {
                let range = gltf_light.range().unwrap_or(f32::INFINITY);
                Light::spot(
                    position.to_array(),
                    direction.to_array(),
                    color,
                    intensity,
                    range,
                    inner_cone_angle,
                    outer_cone_angle
                )
            }
        }
    }

    /// Process a mesh primitive (convert to indexed triangles with vertex deduplication)
    fn process_primitive(
        &self,
        primitive: &Primitive,
        transform: &glam::Mat4,
        material_map: &HashMap<usize, u32>,
        triangles: &mut Vec<Triangle>,
        vertices: &mut Vec<Vertex>,
    ) -> Result<(), GltfError> {
        // Get material ID
        let material_id = primitive.material().index()
            .and_then(|idx| material_map.get(&idx).copied())
            .unwrap_or(0); // Default to material 0 if no material

        // Get position data
        let positions = primitive.get(&gltf::Semantic::Positions)
            .ok_or_else(|| GltfError::ValidationError("Primitive missing position data".to_string()))?;

        let positions_data = self.get_accessor_data::<[f32; 3]>(&positions)?;

        // Use HashMap for vertex deduplication (position-based)
        let mut vertex_map: HashMap<[u32; 3], u32> = HashMap::new(); // position bits -> vertex index
        
        // Helper function to get or create vertex index with deduplication
        let mut get_vertex_index = |position: [f32; 3]| -> u32 {
            let transformed_pos = self.transform_vertex(position, transform);
            
            // Convert to bits for exact comparison
            let pos_bits = [
                transformed_pos[0].to_bits(),
                transformed_pos[1].to_bits(), 
                transformed_pos[2].to_bits(),
            ];
            
            if let Some(&existing_index) = vertex_map.get(&pos_bits) {
                existing_index
            } else {
                let new_index = vertices.len() as u32;
                vertices.push(Vertex {
                    position: transformed_pos,
                });
                vertex_map.insert(pos_bits, new_index);
                new_index
            }
        };

        // Handle different topology types
        match primitive.mode() {
            gltf::mesh::Mode::Triangles => {
                if let Some(indices) = primitive.indices() {
                    // Indexed triangles
                    let indices_data = self.get_indices_data(&indices)?;
                    
                    for chunk in indices_data.chunks(3) {
                        if chunk.len() == 3 {
                            let v0_idx = get_vertex_index(positions_data[chunk[0] as usize]);
                            let v1_idx = get_vertex_index(positions_data[chunk[1] as usize]);
                            let v2_idx = get_vertex_index(positions_data[chunk[2] as usize]);
                            
                            triangles.push(Triangle::new_indexed(v0_idx, v1_idx, v2_idx, material_id));
                        }
                    }
                } else {
                    // Non-indexed triangles
                    for chunk in positions_data.chunks(3) {
                        if chunk.len() == 3 {
                            let v0_idx = get_vertex_index(chunk[0]);
                            let v1_idx = get_vertex_index(chunk[1]);
                            let v2_idx = get_vertex_index(chunk[2]);
                            
                            triangles.push(Triangle::new_indexed(v0_idx, v1_idx, v2_idx, material_id));
                        }
                    }
                }
            }
            gltf::mesh::Mode::TriangleFan => {
                // Triangle fan: first vertex is shared by all triangles
                if positions_data.len() >= 3 {
                    let center_idx = get_vertex_index(positions_data[0]);
                    for i in 1..positions_data.len() - 1 {
                        let v1_idx = get_vertex_index(positions_data[i]);
                        let v2_idx = get_vertex_index(positions_data[i + 1]);
                        
                        triangles.push(Triangle::new_indexed(center_idx, v1_idx, v2_idx, material_id));
                    }
                }
            }
            gltf::mesh::Mode::TriangleStrip => {
                // Triangle strip: each new vertex forms a triangle with previous two
                for i in 0..positions_data.len().saturating_sub(2) {
                    let v0_idx = get_vertex_index(positions_data[i]);
                    let v1_idx = get_vertex_index(positions_data[i + 1]);
                    let v2_idx = get_vertex_index(positions_data[i + 2]);
                    
                    // Alternate winding order for strips
                    if i % 2 == 0 {
                        triangles.push(Triangle::new_indexed(v0_idx, v1_idx, v2_idx, material_id));
                    } else {
                        triangles.push(Triangle::new_indexed(v0_idx, v2_idx, v1_idx, material_id));
                    }
                }
            }
            _ => {
                println!("Warning: Unsupported primitive mode: {:?}", primitive.mode());
            }
        }

        Ok(())
    }

    /// Convert glTF material to enhanced raytracer material
    fn convert_material(&self, gltf_material: &gltf::Material, _textures: &[TextureInfo]) -> Result<Material, GltfError> {
        let mut material = if let Some(spec_gloss) = gltf_material.pbr_specular_glossiness() {
            // KHR_materials_pbrSpecularGlossiness workflow
            let diffuse = spec_gloss.diffuse_factor();
            let specular = spec_gloss.specular_factor();
            let glossiness = spec_gloss.glossiness_factor();
            
            Material::specular_glossiness(
                [diffuse[0], diffuse[1], diffuse[2]],
                specular,
                glossiness
            )
        } else {
            // Standard metallic-roughness workflow
            let pbr = gltf_material.pbr_metallic_roughness();
            
            let base_color = pbr.base_color_factor();
            let albedo = [base_color[0], base_color[1], base_color[2]];
            let metallic = pbr.metallic_factor();
            let roughness = pbr.roughness_factor();
            
            Material::new(albedo, metallic, roughness, [0.0; 3], 1.5, 0.0)
        };

        // Handle emission
        let emission_factor = gltf_material.emissive_factor();
        material.emission = [emission_factor[0], emission_factor[1], emission_factor[2]];

        // Handle transmission (KHR_materials_transmission)
        if let Some(transmission) = gltf_material.transmission() {
            material.set_transmission(transmission.transmission_factor());
        }

        // Handle IOR (KHR_materials_ior)
        if let Some(ior) = gltf_material.ior() {
            material.set_ior(ior);
        }

        // Handle specular (KHR_materials_specular)
        if let Some(specular) = gltf_material.specular() {
            material.specular_factor = specular.specular_factor();
            let spec_color = specular.specular_color_factor();
            material.specular_color = [spec_color[0], spec_color[1], spec_color[2]];
        }

        // Handle volume (KHR_materials_volume)
        if let Some(volume) = gltf_material.volume() {
            material.thickness_factor = volume.thickness_factor();
            material.attenuation_distance = volume.attenuation_distance();
            let att_color = volume.attenuation_color();
            material.attenuation_color = [att_color[0], att_color[1], att_color[2]];
        }

        // Handle texture indices
        let mut texture_indices = [u32::MAX; 8];
        let mut tex_index = 0;

        // Base color/diffuse texture
        if let Some(base_color_texture) = gltf_material.pbr_metallic_roughness().base_color_texture() {
            if tex_index < 8 {
                texture_indices[tex_index] = base_color_texture.texture().index() as u32;
                tex_index += 1;
            }
        }

        // Metallic-roughness texture
        if let Some(mr_texture) = gltf_material.pbr_metallic_roughness().metallic_roughness_texture() {
            if tex_index < 8 {
                texture_indices[tex_index] = mr_texture.texture().index() as u32;
                tex_index += 1;
            }
        }

        // Normal texture
        if let Some(normal_texture) = gltf_material.normal_texture() {
            if tex_index < 8 {
                texture_indices[tex_index] = normal_texture.texture().index() as u32;
                tex_index += 1;
            }
        }

        // Emissive texture
        if let Some(emissive_texture) = gltf_material.emissive_texture() {
            if tex_index < 8 {
                texture_indices[tex_index] = emissive_texture.texture().index() as u32;
                tex_index += 1;
            }
        }

        material.texture_indices = texture_indices;

        Ok(material)
    }

    /// Transform a vertex by a 4x4 matrix
    fn transform_vertex(&self, vertex: [f32; 3], transform: &glam::Mat4) -> [f32; 3] {
        let v = glam::Vec3::from_array(vertex);
        let transformed = transform.transform_point3(v);
        transformed.to_array()
    }

    /// Get accessor data as a specific type
    fn get_accessor_data<T: Clone + Default + bytemuck::Pod>(&self, accessor: &gltf::Accessor) -> Result<Vec<T>, GltfError> {
        let buffer_view = accessor.view()
            .ok_or_else(|| GltfError::ValidationError("Accessor missing buffer view".to_string()))?;
        
        let buffer_index = buffer_view.buffer().index();
        let buffer_data = &self.buffers[buffer_index];
        
        let start = buffer_view.offset() + accessor.offset();
        let stride = buffer_view.stride().unwrap_or(accessor.size());
        
        let mut result = Vec::with_capacity(accessor.count());
        
        for i in 0..accessor.count() {
            let offset = start + i * stride;
            let end = offset + accessor.size();
            
            if end <= buffer_data.len() {
                // Safe casting using bytemuck for well-defined types
                if std::mem::size_of::<T>() == 12 { // [f32; 3]
                    let slice = &buffer_data[offset..offset + 12];
                    let array: [f32; 3] = [
                        f32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]),
                        f32::from_le_bytes([slice[4], slice[5], slice[6], slice[7]]),
                        f32::from_le_bytes([slice[8], slice[9], slice[10], slice[11]]),
                    ];
                    // Safe casting using bytemuck - only works for compatible types
                    if let Ok(value) = bytemuck::try_cast::<[f32; 3], T>(array) {
                        result.push(value);
                    } else {
                        return Err(GltfError::ValidationError(format!(
                            "Type conversion failed for accessor data: expected size {}, got size {}", 
                            std::mem::size_of::<T>(), 
                            std::mem::size_of::<[f32; 3]>()
                        )));
                    }
                } else {
                    result.push(T::default());
                }
            } else {
                return Err(GltfError::ValidationError("Buffer access out of bounds".to_string()));
            }
        }
        
        Ok(result)
    }

    /// Get indices data
    fn get_indices_data(&self, accessor: &gltf::Accessor) -> Result<Vec<u32>, GltfError> {
        let buffer_view = accessor.view()
            .ok_or_else(|| GltfError::ValidationError("Indices accessor missing buffer view".to_string()))?;
        
        let buffer_index = buffer_view.buffer().index();
        let buffer_data = &self.buffers[buffer_index];
        
        let start = buffer_view.offset() + accessor.offset();
        let mut result = Vec::with_capacity(accessor.count());
        
        match accessor.data_type() {
            gltf::accessor::DataType::U8 => {
                for i in 0..accessor.count() {
                    let offset = start + i;
                    if offset < buffer_data.len() {
                        result.push(buffer_data[offset] as u32);
                    }
                }
            }
            gltf::accessor::DataType::U16 => {
                for i in 0..accessor.count() {
                    let offset = start + i * 2;
                    if offset + 1 < buffer_data.len() {
                        let value = u16::from_le_bytes([buffer_data[offset], buffer_data[offset + 1]]);
                        result.push(value as u32);
                    }
                }
            }
            gltf::accessor::DataType::U32 => {
                for i in 0..accessor.count() {
                    let offset = start + i * 4;
                    if offset + 3 < buffer_data.len() {
                        let value = u32::from_le_bytes([
                            buffer_data[offset],
                            buffer_data[offset + 1],
                            buffer_data[offset + 2],
                            buffer_data[offset + 3],
                        ]);
                        result.push(value);
                    }
                }
            }
            _ => {
                return Err(GltfError::ValidationError("Unsupported index data type".to_string()));
            }
        }
        
        Ok(result)
    }

    /// Get list of available scenes
    pub fn list_scenes(&self) -> Vec<(usize, Option<String>)> {
        self.document
            .scenes()
            .enumerate()
            .map(|(index, scene)| (index, scene.name().map(|s| s.to_string())))
            .collect()
    }

    /// Get scene count
    pub fn scene_count(&self) -> usize {
        self.document.scenes().len()
    }

    /// Get list of available cameras
    pub fn list_cameras(&self) -> Vec<(usize, Option<String>)> {
        self.document
            .cameras()
            .enumerate()
            .map(|(index, camera)| (index, camera.name().map(|s| s.to_string())))
            .collect()
    }

    /// Get list of available lights (KHR_lights_punctual)
    pub fn list_lights(&self) -> Vec<(usize, Option<String>)> {
        if let Some(lights) = self.document.lights() {
            lights
                .enumerate()
                .map(|(index, light)| (index, light.name().map(|s| s.to_string())))
                .collect()
        } else {
            Vec::new()
        }
    }
}