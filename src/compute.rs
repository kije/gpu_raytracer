use bytemuck;
use raytracer_shared::{PushConstants, RaytracerConfig, WavefrontRay, WavefrontCounters};
use crate::renderer::{RenderState, ProgressiveState, PerformanceState, TimingBreakdown, ProgressiveTiming};
use crate::buffers::BufferManager;
use crate::scene::SceneState;

/// Compute shader execution and progressive rendering logic
pub struct ComputeRenderer;

impl ComputeRenderer {
    /// Run the compute shader for progressive rendering with detailed timing
    pub fn run_compute(
        render: &mut RenderState,
        buffers: &mut BufferManager,
        scene: &SceneState,
        progressive: &mut ProgressiveState,
        performance: &mut PerformanceState,
    ) {
        let frame_start = std::time::Instant::now();
        
        // Handle progressive rendering state
        if !Self::handle_progressive_rendering_setup(progressive, performance) {
            return;
        }

        let total_tiles = progressive.tiles_x * progressive.tiles_y;
        if Self::check_rendering_completion(progressive, performance, total_tiles) {
            return;
        }

        let tiles_this_frame = Self::calculate_tiles_for_frame(progressive, total_tiles);
        
        // Time buffer updates
        let buffer_update_start = std::time::Instant::now();
        let metadata_offsets = Self::update_buffers_and_bind_groups(render, buffers, scene);
        let buffer_update_time = buffer_update_start.elapsed();
        
        // Time compute submission
        let compute_start = std::time::Instant::now();
        Self::execute_compute_pass(render, buffers, scene, progressive, performance, tiles_this_frame, metadata_offsets);
        let compute_submission_time = compute_start.elapsed();
        
        let total_frame_time = frame_start.elapsed();
        
        // Update performance metrics with detailed timing breakdown
        Self::update_performance_with_timing(
            progressive, performance, tiles_this_frame, total_tiles,
            buffer_update_time, compute_submission_time, total_frame_time
        );
    }

    /// Setup progressive rendering if needed and initialize timing
    fn handle_progressive_rendering_setup(progressive: &mut ProgressiveState, performance: &mut PerformanceState) -> bool {
        if progressive.needs_recompute && !progressive.is_progressive_rendering {
            progressive.is_progressive_rendering = true;
            progressive.current_tile = 0;
            progressive.progressive_start_time = std::time::Instant::now();
            progressive.needs_recompute = false;
            
            let total_tiles = progressive.tiles_x * progressive.tiles_y;
            
            // Initialize progressive timing state
            performance.progressive_timing = Some(ProgressiveTiming {
                start_time: std::time::Instant::now(),
                total_tiles,
                completed_tiles: 0,
                total_buffer_update_time: std::time::Duration::ZERO,
                total_compute_submission_time: std::time::Duration::ZERO,
                tile_times: Vec::with_capacity(total_tiles as usize),
            });
            
            println!("┌─────────────────────────────────────────────────────────────┐");
            println!("│ Starting progressive render: {}x{} tiles ({} total)        │", 
                     progressive.tiles_x, progressive.tiles_y, total_tiles);
            println!("│ Tiles per frame: {} - Advanced timing enabled             │", 
                     progressive.tiles_per_frame);
            println!("│ Note: Submit=GPU queue submission time, Rate=overall speed │");
            println!("└─────────────────────────────────────────────────────────────┘");
        }
        
        progressive.is_progressive_rendering
    }

    /// Check if progressive rendering is complete and print comprehensive timing summary
    fn check_rendering_completion(progressive: &mut ProgressiveState, performance: &mut PerformanceState, total_tiles: u32) -> bool {
        if progressive.current_tile >= total_tiles {
            progressive.is_progressive_rendering = false;
            let total_time = progressive.progressive_start_time.elapsed();
            
            // Print comprehensive timing summary
            if let Some(ref prog_timing) = performance.progressive_timing {
                Self::print_completion_summary(prog_timing, total_time, total_tiles);
            }
            
            // Clear progressive timing state
            performance.progressive_timing = None;
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
            &scene.bvh_nodes, &scene.triangle_indices, &scene.vertices
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
        _tile_index_offset: u32, // fixme remove
    ) {
        // Set the appropriate bind group for this color channel
        let bind_group = render.get_compute_bind_group(color_channel)
            .expect("Invalid color channel");
        compute_pass.set_bind_group(0, bind_group, &[]);
        
        let push_constants = PushConstants::new(
            [render.size.width as f32, render.size.height as f32],
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
        
        compute_pass.set_push_constants(0, bytemuck::cast_slice(&[push_constants]));

        // Dispatch workgroups for current tile and channel
        let workgroup_x = (actual_tile_width + RaytracerConfig::THREAD_GROUP_SIZE.0 - 1) / RaytracerConfig::THREAD_GROUP_SIZE.0;
        let workgroup_y = (actual_tile_height + RaytracerConfig::THREAD_GROUP_SIZE.1 - 1) / RaytracerConfig::THREAD_GROUP_SIZE.1;
        compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }

    /// Update performance metrics with detailed timing breakdown
    fn update_performance_with_timing(
        progressive: &mut ProgressiveState,
        performance: &mut PerformanceState,
        tiles_this_frame: u32,
        total_tiles: u32,
        buffer_update_time: std::time::Duration,
        compute_submission_time: std::time::Duration,
        total_frame_time: std::time::Duration,
    ) {
        // Update legacy timing for compatibility
        performance.last_compute_time = compute_submission_time;
        
        // Calculate more realistic tiles per second based on progressive rendering rate
        let realistic_tiles_per_sec = if let Some(ref prog_timing) = performance.progressive_timing {
            let elapsed_since_start = prog_timing.start_time.elapsed().as_secs_f32();
            if elapsed_since_start > 0.0 {
                prog_timing.completed_tiles as f32 / elapsed_since_start
            } else {
                0.0
            }
        } else {
            // Fallback to frame-based calculation (less accurate)
            if total_frame_time.as_secs_f32() > 0.0 {
                tiles_this_frame as f32 / total_frame_time.as_secs_f32()
            } else {
                0.0
            }
        };

        // Update detailed timing breakdown
        performance.last_timing_breakdown = TimingBreakdown {
            buffer_update_time,
            compute_submission_time,
            total_frame_time,
            tiles_processed: tiles_this_frame,
            tiles_per_second: realistic_tiles_per_sec,
        };
        
        // Update progressive timing state
        if let Some(ref mut prog_timing) = performance.progressive_timing {
            prog_timing.completed_tiles += tiles_this_frame;
            prog_timing.total_buffer_update_time += buffer_update_time;
            prog_timing.total_compute_submission_time += compute_submission_time;
            prog_timing.tile_times.push(total_frame_time);
        }
        
        progressive.current_tile += tiles_this_frame;
        
        // Enhanced progress feedback with timing details
        if progressive.current_tile % progressive.tiles_per_frame == 0 || progressive.current_tile == total_tiles {
            let progress = (progressive.current_tile as f32 / total_tiles as f32) * RaytracerConfig::PROGRESS_PERCENTAGE_SCALE;
            let timing = &performance.last_timing_breakdown;
            
            println!("Progress: {:.1}% ({}/{}) │ Buffer: {:.2}ms │ Submit: {:.2}ms │ Frame: {:.2}ms │ Rate: {:.1} tiles/sec", 
                progress, 
                progressive.current_tile, 
                total_tiles,
                timing.buffer_update_time.as_secs_f32() * RaytracerConfig::MILLISECONDS_PER_SECOND,
                timing.compute_submission_time.as_secs_f32() * RaytracerConfig::MILLISECONDS_PER_SECOND,
                timing.total_frame_time.as_secs_f32() * RaytracerConfig::MILLISECONDS_PER_SECOND,
                timing.tiles_per_second
            );
        }
    }
    
    /// Print comprehensive timing summary when rendering completes
    fn print_completion_summary(
        prog_timing: &ProgressiveTiming, 
        total_time: std::time::Duration, 
        total_tiles: u32
    ) {
        let total_secs = total_time.as_secs_f32();
        let avg_buffer_time = prog_timing.total_buffer_update_time.as_secs_f32() / total_tiles as f32;
        let avg_compute_time = prog_timing.total_compute_submission_time.as_secs_f32() / total_tiles as f32;
        let overall_tiles_per_sec = total_tiles as f32 / total_secs;
        
        // Calculate percentile timings
        let mut frame_times: Vec<f32> = prog_timing.tile_times.iter()
            .map(|d| d.as_secs_f32() * RaytracerConfig::MILLISECONDS_PER_SECOND)
            .collect();
        frame_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p50 = frame_times.get(frame_times.len() / 2).unwrap_or(&0.0);
        let p95 = frame_times.get((frame_times.len() * 95) / 100).unwrap_or(&0.0);
        let p99 = frame_times.get((frame_times.len() * 99) / 100).unwrap_or(&0.0);
        
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│ Progressive Rendering Complete - Timing Summary            │");
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Total Time: {:.2}s │ Tiles: {} │ Rate: {:.1} tiles/sec      │", 
                 total_secs, total_tiles, overall_tiles_per_sec);
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Average Times per Tile:                                    │");
        println!("│  • Buffer Updates: {:.3}ms                                 │", 
                 avg_buffer_time * RaytracerConfig::MILLISECONDS_PER_SECOND);
        println!("│  • Compute Submit: {:.3}ms                                 │", 
                 avg_compute_time * RaytracerConfig::MILLISECONDS_PER_SECOND);
        println!("│  • Total Buffer:   {:.2}ms ({:.1}% of total)               │", 
                 prog_timing.total_buffer_update_time.as_secs_f32() * RaytracerConfig::MILLISECONDS_PER_SECOND,
                 (prog_timing.total_buffer_update_time.as_secs_f32() / total_secs) * 100.0);
        println!("│  • Total Compute:  {:.2}ms ({:.1}% of total)               │", 
                 prog_timing.total_compute_submission_time.as_secs_f32() * RaytracerConfig::MILLISECONDS_PER_SECOND,
                 (prog_timing.total_compute_submission_time.as_secs_f32() / total_secs) * 100.0);
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Frame Time Percentiles:                                    │");
        println!("│  • P50 (median): {:.2}ms                                   │", p50);
        println!("│  • P95:          {:.2}ms                                   │", p95);
        println!("│  • P99:          {:.2}ms                                   │", p99);
        println!("└─────────────────────────────────────────────────────────────┘");
    }

    /// Run wavefront raytracing with multiple bounce depths
    /// This is the main entry point for the new wavefront raytracing system
    pub fn run_wavefront_compute(
        render: &mut RenderState,
        buffers: &mut BufferManager,
        scene: &SceneState,
        progressive: &mut ProgressiveState,
        performance: &mut PerformanceState,
        max_bounce_depth: u32,
    ) {
        let frame_start = std::time::Instant::now();
        let frame_seed = frame_start.elapsed().as_nanos() as u32; // Simple frame seed
        
        // Handle progressive rendering state
        if !Self::handle_progressive_rendering_setup(progressive, performance) {
            return;
        }

        let total_tiles = progressive.tiles_x * progressive.tiles_y;
        if Self::check_rendering_completion(progressive, performance, total_tiles) {
            return;
        }

        let tiles_this_frame = Self::calculate_tiles_for_frame(progressive, total_tiles);
        
        // Time buffer updates
        let buffer_update_start = std::time::Instant::now();
        let metadata_offsets = Self::update_buffers_and_bind_groups(render, buffers, scene);
        let buffer_update_time = buffer_update_start.elapsed();
        
        // Time wavefront compute execution
        let compute_start = std::time::Instant::now();
        Self::execute_wavefront_compute_pass(
            render, buffers, scene, progressive, performance, 
            tiles_this_frame, metadata_offsets, max_bounce_depth, frame_seed
        );
        let compute_submission_time = compute_start.elapsed();
        
        let total_frame_time = frame_start.elapsed();
        
        // Update performance metrics with detailed timing breakdown
        Self::update_performance_with_timing(
            progressive, performance, tiles_this_frame, total_tiles,
            buffer_update_time, compute_submission_time, total_frame_time
        );
    }

    /// Execute wavefront raytracing compute passes for each bounce depth
    fn execute_wavefront_compute_pass(
        render: &mut RenderState,
        buffers: &BufferManager,
        scene: &SceneState,
        progressive: &ProgressiveState,
        _performance: &PerformanceState,
        tiles_this_frame: u32,
        metadata_offsets: raytracer_shared::SceneMetadataOffsets,
        max_bounce_depth: u32,
        frame_seed: u32,
    ) {
        let mut encoder = render.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Wavefront Compute Encoder"),
        });

        // Initialize wavefront counters
        let mut wavefront_counters = WavefrontCounters::new(max_bounce_depth, frame_seed);
        
        // Calculate total rays needed for all tiles this frame
        let mut total_rays_needed = 0u32;
        for i in 0..tiles_this_frame {
            let tile_index = progressive.current_tile + i;
            let (_, _, actual_tile_width, actual_tile_height) = 
                Self::calculate_tile_dimensions(tile_index, progressive, render);
            total_rays_needed += actual_tile_width * actual_tile_height * 3; // 3 color channels
        }
        
        // Add camera rays for bounce depth 0
        wavefront_counters.add_rays(0, total_rays_needed);

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Wavefront Compute Pass"),
            });

            compute_pass.set_pipeline(&render.compute_pipeline);

            // Process each bounce depth sequentially
            for bounce_depth in 0..=max_bounce_depth {
                if !wavefront_counters.has_active_rays(bounce_depth) {
                    continue;
                }

                let ray_count = wavefront_counters.get_ray_count(bounce_depth);
                println!("Processing bounce depth {} with {} rays", bounce_depth, ray_count);

                // Process all tiles for this bounce depth
                for i in 0..tiles_this_frame {
                    Self::process_tile_wavefront(
                        &mut compute_pass, render, buffers, scene, progressive,
                        i, metadata_offsets, bounce_depth, max_bounce_depth, frame_seed
                    );
                }

                // Note: In a real implementation, we would update ray counts based on 
                // how many rays were generated for the next bounce depth.
                // For this initial implementation, we simulate termination.
                let continuation_rate = 0.7f32.powf(bounce_depth as f32); // Exponential decay
                let next_ray_count = (ray_count as f32 * continuation_rate) as u32;
                if bounce_depth < max_bounce_depth && next_ray_count > 0 {
                    wavefront_counters.add_rays(bounce_depth + 1, next_ray_count);
                }
            }
        }

        render.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Process a single tile with wavefront raytracing for a specific bounce depth
    fn process_tile_wavefront<'a>(
        compute_pass: &mut wgpu::ComputePass<'a>,
        render: &'a RenderState,
        buffers: &BufferManager,
        scene: &SceneState,
        progressive: &ProgressiveState,
        tile_index_offset: u32,
        metadata_offsets: raytracer_shared::SceneMetadataOffsets,
        current_bounce_depth: u32,
        max_bounce_depth: u32,
        frame_seed: u32,
    ) {
        let tile_index = progressive.current_tile + tile_index_offset;
        let (tile_offset_x, tile_offset_y, actual_tile_width, actual_tile_height) = 
            Self::calculate_tile_dimensions(tile_index, progressive, render);

        // Process all three color channels for each tile at this bounce depth
        for color_channel in 0..3 {
            Self::process_color_channel_wavefront(
                compute_pass, render, buffers, scene, progressive,
                tile_offset_x, tile_offset_y, actual_tile_width, actual_tile_height,
                color_channel, metadata_offsets, current_bounce_depth, max_bounce_depth, frame_seed
            );
        }
    }

    /// Process a single color channel for a tile with wavefront raytracing
    fn process_color_channel_wavefront<'a>(
        compute_pass: &mut wgpu::ComputePass<'a>,
        render: &'a RenderState,
        buffers: &BufferManager,
        scene: &SceneState,
        progressive: &ProgressiveState,
        tile_offset_x: u32,
        tile_offset_y: u32,
        actual_tile_width: u32,
        actual_tile_height: u32,
        color_channel: u32,
        metadata_offsets: raytracer_shared::SceneMetadataOffsets,
        current_bounce_depth: u32,
        max_bounce_depth: u32,
        frame_seed: u32,
    ) {
        // Set the appropriate bind group for this color channel
        let bind_group = render.get_compute_bind_group(color_channel)
            .expect("Invalid color channel");
        compute_pass.set_bind_group(0, bind_group, &[]);
        
        let push_constants = PushConstants::new_wavefront(
            [render.size.width as f32, render.size.height as f32],
            scene.camera,
            scene.triangles.len() as u32,
            scene.materials.len() as u32,
            [tile_offset_x, tile_offset_y],
            [actual_tile_width, actual_tile_height],
            [progressive.tiles_x, progressive.tiles_y],
            buffers.triangles_per_buffer as u32,
            metadata_offsets,
            color_channel,
            current_bounce_depth,
            max_bounce_depth,
            frame_seed,
        );
        
        compute_pass.set_push_constants(0, bytemuck::cast_slice(&[push_constants]));

        // Dispatch workgroups for current tile and channel
        let workgroup_x = (actual_tile_width + RaytracerConfig::THREAD_GROUP_SIZE.0 - 1) / RaytracerConfig::THREAD_GROUP_SIZE.0;
        let workgroup_y = (actual_tile_height + RaytracerConfig::THREAD_GROUP_SIZE.1 - 1) / RaytracerConfig::THREAD_GROUP_SIZE.1;
        compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }
}