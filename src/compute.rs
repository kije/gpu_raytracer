use bytemuck;
use raytracer_shared::{PushConstants, RaytracerConfig};
use crate::renderer::{RenderState, ProgressiveState, PerformanceState};
use crate::buffers::BufferManager;
use crate::scene::SceneState;

/// Compute shader execution and progressive rendering logic
pub struct ComputeRenderer;

impl ComputeRenderer {
    /// Run the compute shader for progressive rendering
    pub fn run_compute(
        render: &mut RenderState,
        buffers: &mut BufferManager,
        scene: &SceneState,
        progressive: &mut ProgressiveState,
        performance: &mut PerformanceState,
    ) {
        // Start progressive rendering if we need to recompute
        if progressive.needs_recompute && !progressive.is_progressive_rendering {
            progressive.is_progressive_rendering = true;
            progressive.current_tile = 0;
            progressive.progressive_start_time = std::time::Instant::now();
            progressive.needs_recompute = false;
            
            println!("Starting parallel progressive render: {}x{} tiles, {} tiles per frame", 
                     progressive.tiles_x, progressive.tiles_y, progressive.tiles_per_frame);
        }
        
        // Continue progressive rendering if in progress
        if !progressive.is_progressive_rendering {
            return;
        }

        let total_tiles = progressive.tiles_x * progressive.tiles_y;
        if progressive.current_tile >= total_tiles {
            // Progressive rendering completed
            progressive.is_progressive_rendering = false;
            let total_time = progressive.progressive_start_time.elapsed();
            println!("Progressive render completed in {:.2}ms", total_time.as_secs_f32() * RaytracerConfig::MILLISECONDS_PER_SECOND);
            return;
        }

        let compute_start = std::time::Instant::now();
        
        // Calculate how many tiles to process this frame
        let tiles_remaining = total_tiles - progressive.current_tile;
        let tiles_this_frame = std::cmp::min(progressive.tiles_per_frame, tiles_remaining);
        
        let mut encoder = render.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Parallel Compute Encoder"),
            });
        
        // Update geometry buffers using smart buffer management
        let spheres_resized = buffers.update_spheres(&render.device, &render.queue, &scene.spheres);
        let triangles_resized = buffers.update_triangles(&render.device, &render.queue, &scene.triangles);
        let materials_resized = buffers.update_materials(&render.device, &render.queue, &scene.materials);
        let lights_resized = buffers.update_lights(&render.device, &render.queue, &scene.lights);
        let textures_resized = buffers.update_textures(&render.device, &render.queue, &scene.textures);
        let texture_data_resized = buffers.update_texture_data(&render.device, &render.queue, &scene.texture_data);
        
        // If buffers were resized, recreate bind groups with new buffers
        if spheres_resized || triangles_resized || materials_resized || lights_resized || textures_resized || texture_data_resized {
            render.recreate_bind_groups(buffers);
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });

            compute_pass.set_pipeline(&render.compute_pipeline);
            compute_pass.set_bind_group(0, &render.compute_bind_group, &[]);

            // Process multiple tiles in parallel
            for i in 0..tiles_this_frame {
                let tile_index = progressive.current_tile + i;
                let tile_x = tile_index % progressive.tiles_x;
                let tile_y = tile_index / progressive.tiles_x;
                
                let tile_offset_x = tile_x * RaytracerConfig::TILE_SIZE;
                let tile_offset_y = tile_y * RaytracerConfig::TILE_SIZE;
                
                // Calculate actual tile size (handle edge tiles)
                let actual_tile_width = std::cmp::min(RaytracerConfig::TILE_SIZE, render.size.width - tile_offset_x);
                let actual_tile_height = std::cmp::min(RaytracerConfig::TILE_SIZE, render.size.height - tile_offset_y);
                
                let push_constants = PushConstants::new(
                    [render.size.width as f32, render.size.height as f32],
                    performance.start_time.elapsed().as_secs_f32(),
                    scene.camera,
                    scene.spheres.len() as u32,
                    scene.triangles.len() as u32, // All triangles accessible
                    scene.materials.len() as u32,
                    scene.lights.len() as u32,
                    [tile_offset_x, tile_offset_y],
                    [actual_tile_width, actual_tile_height],
                    [progressive.tiles_x, progressive.tiles_y],
                    buffers.triangles_per_buffer as u32,
                );
                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[push_constants]));

                // Dispatch workgroups for current tile
                let workgroup_x = (actual_tile_width + RaytracerConfig::THREAD_GROUP_SIZE.0 - 1) / RaytracerConfig::THREAD_GROUP_SIZE.0;
                let workgroup_y = (actual_tile_height + RaytracerConfig::THREAD_GROUP_SIZE.1 - 1) / RaytracerConfig::THREAD_GROUP_SIZE.1;
                compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
            }
        }

        render.queue.submit(std::iter::once(encoder.finish()));
        
        // Don't wait for GPU - allow frame to be displayed immediately
        performance.last_compute_time = compute_start.elapsed();
        
        // Move to next set of tiles
        progressive.current_tile += tiles_this_frame;
        
        // Progress feedback
        if progressive.current_tile % progressive.tiles_per_frame == 0 || progressive.current_tile == total_tiles {
            let progress = (progressive.current_tile as f32 / total_tiles as f32) * RaytracerConfig::PROGRESS_PERCENTAGE_SCALE;
            println!("Progress: {:.1}% ({}/{}) - {} tiles/frame", 
                     progress, progressive.current_tile, total_tiles, tiles_this_frame);
        }
    }
}