use spirv_std::float::{f32_to_f16, f16_to_f32};
use spirv_std::glam::{vec3, Vec3};
use raytracer_shared::PushConstants;
use crate::ray::Ray;
use crate::intersection::Intersection;
use crate::material::MaterialEvaluator;
use crate::scene_access::SceneAccessor;

/// Lighting calculator for PBR shading
pub struct LightingCalculator<'a> {
    scene_accessor: &'a SceneAccessor<'a>,
}

impl<'a> LightingCalculator<'a> {
    pub fn new(scene_accessor: &'a SceneAccessor<'a>) -> Self {
        Self { scene_accessor }
    }

    /// Calculate lighting for an intersection point
    pub fn calculate_lighting(
        &self,
        intersection: &Intersection,
        _ray: &Ray,
        material_eval: &MaterialEvaluator,
        push_constants: &PushConstants
    ) -> Vec3 {
        let mut total_lighting = vec3(0.0, 0.0, 0.0);

        // Ambient lighting
        let ambient = material_eval.albedo() * 0.1;
        total_lighting += ambient;

        // Process all lights in the scene
        for light_idx in 0..self.scene_accessor.light_count() {
            let light_contribution = self.calculate_light_contribution(
                intersection,
                material_eval,
                light_idx,
                push_constants
            );
            
            total_lighting += light_contribution;
        }

        // Add emission
        total_lighting + material_eval.emission()
    }

    /// Calculate contribution from a single light
    fn calculate_light_contribution(
        &self,
        intersection: &Intersection,
        material_eval: &MaterialEvaluator,
        light_idx: u32,
        push_constants: &PushConstants
    ) -> Vec3 {
        let index_valid = (light_idx < push_constants.metadata_offsets.lights_count) as u32 as f32;
        
        let light_pos = self.scene_accessor.get_light_position(light_idx);
        let light_type = self.scene_accessor.get_light_type(light_idx);
        let light_color = self.scene_accessor.get_light_color(light_idx);
        let light_intensity = self.scene_accessor.get_light_intensity(light_idx);
        let light_dir_vec = self.scene_accessor.get_light_direction(light_idx);
    
        // Calculate different light type contributions
        let directional_contribution = self.calculate_directional_light(
            intersection,
            &light_dir_vec,
            light_intensity
        );

        let point_spot_contribution = self.calculate_point_spot_light(
            intersection,
            &light_pos,
            &light_dir_vec,
            light_intensity
        );
    
        // Branchless light type selection
        let is_directional = (light_type == 0) as u32 as f32;
        let is_point = (light_type == 1) as u32 as f32;
        let is_spot = (light_type == 2) as u32 as f32;
        
        let light_intensity_final = directional_contribution.intensity * is_directional +
                                   point_spot_contribution.intensity * is_point +
                                   point_spot_contribution.spot_intensity * is_spot;
    
        // BRDF evaluation
        let light_contribution = material_eval.evaluate_brdf(light_intensity_final);
    
        // Only add contribution if light intensity is positive and index is valid
        let contribution_valid = ((light_intensity_final > 0.0) as u32 as f32) * index_valid;
        light_contribution * light_color * contribution_valid
    }

    /// Calculate directional light contribution
    fn calculate_directional_light(
        &self,
        intersection: &Intersection,
        light_direction: &Vec3,
        light_intensity: f32
    ) -> LightContribution {
        let dir_light_dir = -light_direction.normalize();
        let intensity = intersection.normal.dot(dir_light_dir).max(0.0) * light_intensity;
        
        LightContribution {
            intensity,
            spot_intensity: 0.0,
        }
    }

    /// Calculate point/spot light contribution with f16 optimization for attenuation
    fn calculate_point_spot_light(
        &self,
        intersection: &Intersection,
        light_position: &Vec3,
        light_direction: &Vec3,
        light_intensity: f32
    ) -> LightContribution {
        let to_light = *light_position - intersection.point;
        let distance = to_light.length();
        let point_light_dir = to_light.normalize();
        
        // Use f16 for attenuation calculation (0.0-1.0 range, perfect for f16)
        let attenuation_f32 = 1.0 / (1.0 + distance * distance * 0.01);
        let attenuation_f16 = f32_to_f16(attenuation_f32);
        let attenuation = f16_to_f32(attenuation_f16);
        
        let point_intensity = intersection.normal.dot(point_light_dir).max(0.0) * light_intensity * attenuation;
    
        // Calculate spot light factor
        let spot_factor = (-light_direction.normalize()).dot(point_light_dir).max(0.0);
        let spot_intensity = point_intensity * spot_factor;

        LightContribution {
            intensity: point_intensity,
            spot_intensity,
        }
    }
}

/// Helper struct for light contribution calculations
struct LightContribution {
    intensity: f32,
    spot_intensity: f32,
}