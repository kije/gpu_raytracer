use spirv_std::float::f16_to_f32;
use spirv_std::glam::{vec3, Vec3};
use raytracer_shared::Material;

/// Material property evaluator with f16 optimization support
pub struct MaterialEvaluator<'a> {
    material: &'a Material,
}

impl<'a> MaterialEvaluator<'a> {
    pub fn new(material: &'a Material) -> Self {
        Self { material }
    }

    /// Get albedo color
    pub fn albedo(&self) -> Vec3 {
        vec3(self.material.albedo[0], self.material.albedo[1], self.material.albedo[2])
    }

    /// Get emission color
    pub fn emission(&self) -> Vec3 {
        vec3(self.material.emission[0], self.material.emission[1], self.material.emission[2])
    }

    /// Get metallic factor (converted from packed f16)
    pub fn metallic(&self) -> f32 {
        f16_to_f32(self.material.metallic_roughness_f16 & 0xFFFF) // Low 16 bits
    }

    /// Get roughness factor (converted from packed f16)
    pub fn roughness(&self) -> f32 {
        f16_to_f32(self.material.metallic_roughness_f16 >> 16) // High 16 bits
    }

    /// Get index of refraction (converted from packed f16)
    pub fn ior(&self) -> f32 {
        f16_to_f32(self.material.ior_transmission_f16 & 0xFFFF) // Low 16 bits
    }

    /// Get wavelength-dependent IOR for chromatic aberration
    /// channel: 0=red, 1=green, 2=blue
    pub fn ior_for_channel(&self, channel: u32) -> f32 {
        let base_ior = self.ior();
        
        // Apply wavelength-dependent dispersion (Cauchy equation approximation)
        // Red: ~650nm, Green: ~540nm, Blue: ~450nm
        // Using simplified dispersion formula: n(λ) = A + B/λ²
        match channel {
            0 => base_ior - 0.02, // Red (less dispersion)
            1 => base_ior,        // Green (reference)
            2 => base_ior + 0.04, // Blue (more dispersion)
            _ => base_ior,        // Fallback
        }
    }

    /// Get transmission factor (converted from packed f16)
    pub fn transmission(&self) -> f32 {
        f16_to_f32(self.material.ior_transmission_f16 >> 16) // High 16 bits
    }

    /// Check if material is metallic
    pub fn is_metallic(&self) -> bool {
        self.metallic() > 0.5
    }

    /// Get diffuse component
    pub fn diffuse(&self) -> Vec3 {
        self.albedo() / core::f32::consts::PI
    }

    /// Evaluate BRDF for lighting calculations
    pub fn evaluate_brdf(&self, light_intensity: f32) -> Vec3 {
        let diffuse = self.diffuse();
        let is_metallic = self.is_metallic() as u32 as f32;
        let metallic_contrib = self.albedo() * light_intensity * 0.5;
        let dielectric_contrib = diffuse * light_intensity;
        
        metallic_contrib * is_metallic + dielectric_contrib * (1.0 - is_metallic)
    }
}