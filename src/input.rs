use winit::{
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState, MouseButton},
};
use raytracer_shared::{RaytracerConfig};

/// Input handling state
pub struct InputState {
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            mouse_pressed: false,
            last_mouse_pos: None,
        }
    }

    /// Handle mouse input events
    pub fn handle_mouse_input(&mut self, button: MouseButton, button_state: ElementState) {
        if button == MouseButton::Left {
            self.mouse_pressed = button_state == ElementState::Pressed;
        }
    }

    /// Handle cursor movement and return camera rotation delta if mouse is pressed
    pub fn handle_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) -> Option<(f64, f64)> {
        let mut delta = None;
        
        if self.mouse_pressed {
            if let Some(last_pos) = self.last_mouse_pos {
                let delta_x = position.x - last_pos.0;
                let delta_y = position.y - last_pos.1;
                delta = Some((delta_x, delta_y));
            }
        }
        
        self.last_mouse_pos = Some((position.x, position.y));
        delta
    }
}

/// Camera movement and rotation
pub struct CameraController;

impl CameraController {
    /// Apply camera rotation based on mouse delta
    pub fn rotate_camera(camera: &mut raytracer_shared::Camera, delta_x: f64, delta_y: f64) {
        let sensitivity = RaytracerConfig::CAMERA_ROTATE_SENSITIVITY;
        let delta_x = delta_x as f32 * sensitivity;
        let delta_y = delta_y as f32 * sensitivity;
        
        // Create rotation around Y axis (yaw)
        let cos_yaw = delta_x.cos();
        let sin_yaw = delta_x.sin();
        
        // Rotate direction vector around Y axis
        let old_dir_x = camera.direction[0];
        let old_dir_z = camera.direction[2];
        camera.direction[0] = old_dir_x * cos_yaw - old_dir_z * sin_yaw;
        camera.direction[2] = old_dir_x * sin_yaw + old_dir_z * cos_yaw;
        
        // Simple pitch rotation (just modify Y component)
        camera.direction[1] = (camera.direction[1] - delta_y).clamp(-RaytracerConfig::CAMERA_PITCH_CLAMP, RaytracerConfig::CAMERA_PITCH_CLAMP);
        
        // Normalize direction
        let len = (camera.direction[0].powi(2) + 
                  camera.direction[1].powi(2) + 
                  camera.direction[2].powi(2)).sqrt();
        if len > 0.0 {
            camera.direction[0] /= len;
            camera.direction[1] /= len;
            camera.direction[2] /= len;
        }
    }
    
    /// Apply camera movement
    pub fn move_camera(camera: &mut raytracer_shared::Camera, forward: f32, right: f32) {
        let speed = RaytracerConfig::CAMERA_MOVE_SPEED;
        
        // Forward/backward movement
        camera.position[0] += camera.direction[0] * forward * speed;
        camera.position[1] += camera.direction[1] * forward * speed;
        camera.position[2] += camera.direction[2] * forward * speed;
        
        // Right/left movement (cross product of direction and up)
        let right_vec = [
            camera.direction[1] * camera.up[2] - camera.direction[2] * camera.up[1],
            camera.direction[2] * camera.up[0] - camera.direction[0] * camera.up[2],
            camera.direction[0] * camera.up[1] - camera.direction[1] * camera.up[0],
        ];
        
        camera.position[0] += right_vec[0] * right * speed;
        camera.position[1] += right_vec[1] * right * speed;
        camera.position[2] += right_vec[2] * right * speed;
    }
}