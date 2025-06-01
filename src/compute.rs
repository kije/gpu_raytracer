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
        // Handle progressive rendering state
        if !Self::handle_progressive_rendering_setup(progressive) {
            return;
        }

        let total_tiles = progressive.tiles_x * progressive.tiles_y;
        if Self::check_rendering_completion(progressive, total_tiles) {
            return;
        }

        let compute_start = std::time::Instant::now();
        let tiles_this_frame = Self::calculate_tiles_for_frame(progressive, total_tiles);
        
        // Update buffers and recreate bind groups if needed
        let metadata_offsets = Self::update_buffers_and_bind_groups(render, buffers, scene);
        
        // Execute compute pass for tiles
        Self::execute_compute_pass(render, buffers, scene, progressive, performance, tiles_this_frame, metadata_offsets);
        
        // Update performance metrics and progress
        Self::update_progress_and_performance(progressive, performance, compute_start, tiles_this_frame, total_tiles);
    }

    /// Setup progressive rendering if needed
    fn handle_progressive_rendering_setup(progressive: &mut ProgressiveState) -> bool {
        if progressive.needs_recompute && !progressive.is_progressive_rendering {
            progressive.is_progressive_rendering = true;
            progressive.current_tile = 0;
            progressive.progressive_start_time = std::time::Instant::now();
            progressive.needs_recompute = false;
            
            println!("Starting parallel progressive render: {}x{} tiles, {} tiles per frame", 
                     progressive.tiles_x, progressive.tiles_y, progressive.tiles_per_frame);
        }
        
        progressive.is_progressive_rendering
    }

    /// Check if progressive rendering is complete
    fn check_rendering_completion(progressive: &mut ProgressiveState, total_tiles: u32) -> bool {
        if progressive.current_tile >= total_tiles {
            progressive.is_progressive_rendering = false;
            let total_time = progressive.progressive_start_time.elapsed();
            println!("Progressive render completed in {:.2}ms", 
                     total_time.as_secs_f32() * RaytracerConfig::MILLISECONDS_PER_SECOND);
            return true;
        }
        false
    }

    /// Calculate how many tiles to process this frame
    fn calculate_tiles_for_frame(progressive: &ProgressiveState, total_tiles: u32) -> u32 {
        let tiles_remaining = total_tiles - progressive.current_tile;
        std::cmp::min(progressive.tiles_per_frame, tiles_remaining)
    }

    /// Update all buffers and recreate bind groups if any were resized
    fn update_buffers_and_bind_groups(
        render: &mut RenderState,
        buffers: &mut BufferManager,
        scene: &SceneState,
    ) -> raytracer_shared::SceneMetadataOffsets {
        // Check if bind groups need recreation BEFORE updating buffers
        // This way we can see the dirty flags before they get cleared by buffer updates
        let needs_bind_group_recreation = buffers.needs_update();
        
        
        let (_scene_metadata_resized, metadata_offsets) = buffers.update_scene_metadata(
            &render.device, &render.queue, &scene.spheres, &scene.lights, 
            &scene.bvh_nodes, &scene.triangle_indices
        );
        let _triangles_resized = buffers.update_triangles(&render.device, &render.queue, &scene.triangles);
        let _materials_resized = buffers.update_materials(&render.device, &render.queue, &scene.materials);
        let _textures_resized = buffers.update_textures(&render.device, &render.queue, &scene.textures);
        let _texture_data_resized = buffers.update_texture_data(&render.device, &render.queue, &scene.texture_data);
        
        // Recreate bind groups if content changed OR if any buffer was resized
        if needs_bind_group_recreation {
            render.recreate_bind_groups(buffers);
        }

        metadata_offsets
    }

    /// Execute the compute pass for multiple tiles and color channels
    fn execute_compute_pass(
        render: &mut RenderState,
        buffers: &BufferManager,
        scene: &SceneState,
        progressive: &ProgressiveState,
        performance: &PerformanceState,
        tiles_this_frame: u32,
        metadata_offsets: raytracer_shared::SceneMetadataOffsets,
    ) {
        let mut encoder = render.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Parallel Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });

            compute_pass.set_pipeline(&render.compute_pipeline);

            for i in 0..tiles_this_frame {
                Self::process_tile(
                    &mut compute_pass, render, buffers, scene, progressive, 
                    performance, i, metadata_offsets
                );
            }
        }

        render.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Process a single tile across all color channels
    fn process_tile<'a>(
        compute_pass: &mut wgpu::ComputePass<'a>,
        render: &'a RenderState,
        buffers: &BufferManager,
        scene: &SceneState,
        progressive: &ProgressiveState,
        performance: &PerformanceState,
        tile_index_offset: u32,
        metadata_offsets: raytracer_shared::SceneMetadataOffsets,
    ) {
        let tile_index = progressive.current_tile + tile_index_offset;
        let (tile_offset_x, tile_offset_y, actual_tile_width, actual_tile_height) = 
            Self::calculate_tile_dimensions(tile_index, progressive, render);

        // Process all three color channels for each tile
        for color_channel in 0..3 {
            Self::process_color_channel(
                compute_pass, render, buffers, scene, progressive, performance,
                tile_offset_x, tile_offset_y, actual_tile_width, actual_tile_height,
                color_channel, metadata_offsets, tile_index_offset
            );
        }
    }

    /// Calculate tile position and dimensions
    fn calculate_tile_dimensions(
        tile_index: u32,
        progressive: &ProgressiveState,
        render: &RenderState,
    ) -> (u32, u32, u32, u32) {
        let tile_x = tile_index % progressive.tiles_x;
        let tile_y = tile_index / progressive.tiles_x;
        
        let tile_offset_x = tile_x * RaytracerConfig::TILE_SIZE;
        let tile_offset_y = tile_y * RaytracerConfig::TILE_SIZE;
        
        let actual_tile_width = std::cmp::min(RaytracerConfig::TILE_SIZE, render.size.width - tile_offset_x);
        let actual_tile_height = std::cmp::min(RaytracerConfig::TILE_SIZE, render.size.height - tile_offset_y);
        
        (tile_offset_x, tile_offset_y, actual_tile_width, actual_tile_height)
    }

    /// Process a single color channel for a tile
    fn process_color_channel<'a>(
        compute_pass: &mut wgpu::ComputePass<'a>,
        render: &'a RenderState,
        buffers: &BufferManager,
        scene: &SceneState,
        progressive: &ProgressiveState,
        performance: &PerformanceState,
        tile_offset_x: u32,
        tile_offset_y: u32,
        actual_tile_width: u32,
        actual_tile_height: u32,
        color_channel: u32,
        metadata_offsets: raytracer_shared::SceneMetadataOffsets,
        tile_index_offset: u32,
    ) {
        // Set the appropriate bind group for this color channel
        let bind_group = render.get_compute_bind_group(color_channel)
            .expect("Invalid color channel");
        compute_pass.set_bind_group(0, bind_group, &[]);
        
        let push_constants = PushConstants::new(
            [render.size.width as f32, render.size.height as f32],
            performance.start_time.elapsed().as_secs_f32(),
            scene.camera,
            scene.triangles.len() as u32,
            scene.materials.len() as u32,
            [tile_offset_x, tile_offset_y],
            [actual_tile_width, actual_tile_height],
            [progressive.tiles_x, progressive.tiles_y],
            buffers.triangles_per_buffer as u32,
            metadata_offsets,
            color_channel,
        );
        
        // Debug: Log push constants for first tile of first channel only to avoid spam
        if tile_index_offset == 0 && color_channel == 0 {
            println!("Push constants: triangles={}, materials={}, spheres={}, lights={}, bvh_nodes={}", 
                     scene.triangles.len(), scene.materials.len(), 
                     metadata_offsets.spheres_count, metadata_offsets.lights_count, metadata_offsets.bvh_nodes_count);
        }
        
        compute_pass.set_push_constants(0, bytemuck::cast_slice(&[push_constants]));

        // Dispatch workgroups for current tile and channel
        let workgroup_x = (actual_tile_width + RaytracerConfig::THREAD_GROUP_SIZE.0 - 1) / RaytracerConfig::THREAD_GROUP_SIZE.0;
        let workgroup_y = (actual_tile_height + RaytracerConfig::THREAD_GROUP_SIZE.1 - 1) / RaytracerConfig::THREAD_GROUP_SIZE.1;
        compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }

    /// Update performance metrics and progress tracking
    fn update_progress_and_performance(
        progressive: &mut ProgressiveState,
        performance: &mut PerformanceState,
        compute_start: std::time::Instant,
        tiles_this_frame: u32,
        total_tiles: u32,
    ) {
        performance.last_compute_time = compute_start.elapsed();
        progressive.current_tile += tiles_this_frame;
        
        // Progress feedback
        if progressive.current_tile % progressive.tiles_per_frame == 0 || progressive.current_tile == total_tiles {
            let progress = (progressive.current_tile as f32 / total_tiles as f32) * RaytracerConfig::PROGRESS_PERCENTAGE_SCALE;
            println!("Progress: {:.1}% ({}/{}) - {} tiles/frame", 
                     progress, progressive.current_tile, total_tiles, tiles_this_frame);
        }
    }
}