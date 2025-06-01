#![feature(build_hasher_simple_hash_one)]
#![feature(inline_const)]

use std::mem;
use winit::{
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState, MouseButton},
    event_loop::EventLoop,
    window::WindowBuilder,
};
use raytracer_shared::PushConstants;

mod gltf_loader;
mod input;
mod renderer;
mod scene;
mod buffers;
mod compute;
mod bvh;

use input::{InputState, CameraController};
use renderer::{RenderState, ProgressiveState, PerformanceState};
use scene::SceneState;
use buffers::BufferManager;
use compute::ComputeRenderer;

/// Graphics and GPU management
struct GraphicsManager {
    render: RenderState,
    progressive: ProgressiveState,
    performance: PerformanceState,
}

impl GraphicsManager {
    fn new(render: RenderState, progressive: ProgressiveState, performance: PerformanceState) -> Self {
        Self { render, progressive, performance }
    }
    
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, buffers: &BufferManager) {
        if new_size.width > 0 && new_size.height > 0 {
            self.render.resize(new_size);
            self.progressive.resize(new_size.width, new_size.height);
            self.render.recreate_bind_groups(buffers);
        }
    }
    
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.performance.update_frame_count();
        self.render.render()
    }
}

/// Scene and content management
struct ContentManager {
    scene: SceneState,
    buffers: BufferManager,
}

impl ContentManager {
    fn new(scene: SceneState, buffers: BufferManager) -> Self {
        Self { scene, buffers }
    }
    
    fn load_gltf(&mut self, path: &str) -> Result<(), gltf_loader::GltfError> {
        self.scene.replace_with_gltf(path)?;
        self.buffers.mark_scene_metadata_dirty();
        self.buffers.mark_triangles_dirty();
        self.buffers.mark_materials_dirty();
        self.buffers.mark_textures_dirty();
        self.buffers.mark_texture_data_dirty();
        Ok(())
    }
}

/// User interaction management
struct InteractionManager {
    input: InputState,
}

impl InteractionManager {
    fn new(input: InputState) -> Self {
        Self { input }
    }
    
    fn handle_mouse_input(&mut self, button: MouseButton, button_state: ElementState) {
        self.input.handle_mouse_input(button, button_state);
    }
    
    fn handle_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) -> Option<(f64, f64)> {
        self.input.handle_cursor_moved(position)
    }
}

/// Main application state coordinator
struct State {
    graphics: GraphicsManager,
    content: ContentManager,
    interaction: InteractionManager,
}

impl State {
    async fn new(window: &winit::window::Window) -> Self {
        let size = window.inner_size();

        // Initialize render state
        let mut render = RenderState::new(window).await;
        
        // Initialize buffer manager
        let buffers = BufferManager::new(&render.device);
        
        // Initialize scene state
        let scene = SceneState::new();
        
        // Initialize progressive rendering state
        let progressive = ProgressiveState::new(size.width, size.height);
        
        // Initialize input state
        let input = InputState::new();
        
        // Initialize performance state
        let performance = PerformanceState::new();

        // Update the render state's bind groups with the proper buffers
        render.recreate_bind_groups(&buffers);

        Self {
            graphics: GraphicsManager::new(render, progressive, performance),
            content: ContentManager::new(scene, buffers),
            interaction: InteractionManager::new(input),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.graphics.resize(new_size, &self.content.buffers);
    }

    fn run_compute(&mut self) {
        ComputeRenderer::run_compute(
            &mut self.graphics.render,
            &mut self.content.buffers,
            &self.content.scene,
            &mut self.graphics.progressive,
            &mut self.graphics.performance,
        );
    }

    fn trigger_recompute(&mut self) {
        self.graphics.progressive.trigger_recompute();
    }

    fn handle_keyboard(&mut self, key: VirtualKeyCode) {
        match key {
            VirtualKeyCode::Space => {
                self.trigger_recompute();
            }
            VirtualKeyCode::W => {
                CameraController::move_camera(&mut self.content.scene.camera, 1.0, 0.0);
                self.trigger_recompute();
            }
            VirtualKeyCode::S => {
                CameraController::move_camera(&mut self.content.scene.camera, -1.0, 0.0);
                self.trigger_recompute();
            }
            VirtualKeyCode::A => {
                CameraController::move_camera(&mut self.content.scene.camera, 0.0, -1.0);
                self.trigger_recompute();
            }
            VirtualKeyCode::D => {
                CameraController::move_camera(&mut self.content.scene.camera, 0.0, 1.0);
                self.trigger_recompute();
            }
            VirtualKeyCode::L => {
                // Try to load a glTF file
                if let Err(e) = self.content.load_gltf("model.gltf") {
                    println!("Failed to load glTF file: {:?}", e);
                } else {
                    self.trigger_recompute();
                }
            }
            _ => {}
        }
    }

    fn handle_mouse_input(&mut self, button: MouseButton, button_state: ElementState) {
        self.interaction.handle_mouse_input(button, button_state);
    }

    fn handle_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        if let Some((delta_x, delta_y)) = self.interaction.handle_cursor_moved(position) {
            CameraController::rotate_camera(&mut self.content.scene.camera, delta_x, delta_y);
            self.trigger_recompute();
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.graphics.render()
    }
}

fn main() {
    const {
        assert!(mem::size_of::<PushConstants>() <= 128, "Push constant must be smaller than 128 bytes (as per Vulkan spec)");
    };
    pollster::block_on(run());
}

async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("GPU Raytracer")
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => control_flow.set_exit(),
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    state.resize(**new_inner_size);
                }
                WindowEvent::KeyboardInput {
                    input: KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(key),
                        ..
                    },
                    ..
                } => {
                    if *key == VirtualKeyCode::Escape {
                        control_flow.set_exit();
                    } else {
                        state.handle_keyboard(*key);
                    }
                }
                WindowEvent::MouseInput {
                    button,
                    state: button_state,
                    ..
                } => {
                    state.handle_mouse_input(*button, *button_state);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    state.handle_cursor_moved(*position);
                }
                _ => {}
            },
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                state.run_compute();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.graphics.render.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => control_flow.set_exit(),
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}