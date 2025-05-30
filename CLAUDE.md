# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GPU-accelerated raytracer written in Rust using rust-gpu and WGPU for compute shaders. The project uses a workspace structure with three main components:

- **Main application** (`src/main.rs`): WGPU-based application that sets up compute pipeline, manages window/surface, and handles the main event loop
- **Compute shader** (`shader/`): SPIR-V compute shader written in Rust using rust-gpu's `spirv-std` for the actual raytracing calculations
- **Shared crate** (`shared/`): Contains shared data structures used by both the main application and shader

## Build System

The project uses a custom build process that compiles the shader to SPIR-V during the main build:

- `build.rs` uses `spirv-builder` to compile the shader crate to SPIR-V bytecode
- The compiled shader binary is embedded into the main application via `include_bytes!(env!("shader.spv"))`
- Requires nightly Rust toolchain `nightly-2023-05-27` with specific components

## Key Commands

```bash
# Build and run (compiles shader automatically)
cargo run

# Build only
cargo build

# Release build
cargo run --release
```

## Architecture Changes (Latest)

The project now uses a **progressive tile-based rendering architecture** with **PBR material system**:

- **Progressive Rendering**: Raytracing uses 64x64 pixel tiles processed incrementally for real-time feedback
- **PBR Materials**: Physically-based rendering with metallic/roughness workflow, ready for Microfacet BRDF
- **Material System**: Separate material buffer with albedo, metallic, roughness, emission, IOR, and transmission
- **Compute Phase**: Raytracing runs in compute shader, writes directly to storage texture
- **Render Phase**: Fragment shader samples the raytraced texture onto a fullscreen quad
- **Performance**: Raytracing only re-runs when needed (window resize, user input like spacebar)
- **Scalability**: Complex raytracing won't block the render loop
- **Shared Crate**: Common data structures eliminate code duplication between main app and shader

### Pipeline Structure
1. **Compute Pipeline**: `main_cs` → Storage texture (rgba8) via progressive tiles
2. **Render Pipeline**: `main_vs` + `main_fs` → Screen framebuffer

### Controls
- **Spacebar**: Trigger raytracer recomputation
- **WASD**: Camera movement
- **Mouse drag**: Camera rotation
- **Resize**: Automatically triggers recomputation

## Architecture

### GPU Pipeline
- **Compute shader**: Runs on GPU with 16x16 thread groups, writes to storage buffer
- **Host application**: Creates WGPU device, sets up compute pipeline, manages surface presentation
- **Data flow**: Push constants (resolution, time) → Compute shader → Storage buffer → Surface texture

### Shader Interface
- **Push constants**: `PushConstants` struct shared between host and shader for resolution/time/camera/tile/material data
- **Storage texture**: Direct writes to rgba8 texture for progressive tile rendering
- **Storage buffers**: Separate buffers for spheres, triangles, and materials
- **Bindings**: 
  - Descriptor set 0, binding 0: Storage texture (write)
  - Descriptor set 0, binding 1: Spheres buffer (read)
  - Descriptor set 0, binding 2: Triangles buffer (read)
  - Descriptor set 0, binding 3: Materials buffer (read)

### Shared Data Structures (`shared/` crate)
- **Camera**: Position, direction, up vector, FOV - uses `[f32; 3]` arrays for cross-platform compatibility
- **Material**: PBR material with albedo, metallic, roughness, emission, IOR, transmission properties
- **Sphere**: Center, radius, material_id - references material by index
- **Triangle**: Three vertices, material_id - triangle primitive with material reference
- **PushConstants**: Resolution, time, camera, tile info, material count - all shader parameters
- **Cross-platform compatibility**: Uses arrays instead of Vec types, works in both std and no_std environments

### Material System
- **PBR Workflow**: Metallic/roughness workflow compatible with standard PBR pipelines
- **Material Types**: 
  - `Material::diffuse()` - Simple Lambert diffuse materials
  - `Material::metallic()` - Metallic materials with controllable roughness
  - `Material::glass()` - Dielectric materials with IOR and transmission
  - `Material::emissive()` - Light-emitting materials for area lights
- **Shader Integration**: Materials indexed by primitives, evaluated in compute shader
- **Extensible**: Ready for advanced Microfacet BRDF models (GGX distribution, Fresnel terms, etc.)

### Key Dependencies
- WGPU 0.16.0 with SPIR-V support for GPU compute
- spirv-builder for shader compilation
- raytracer-shared workspace crate for shared data structures
- Specific dependency pinning for compatibility (ahash 0.7.8, gpu-descriptor 0.2.0)
- glam for vector math
- bytemuck for GPU data marshalling

## Shader Development

The compute shader (`shader/src/lib.rs`) is a `#![no_std]` crate that:
- Uses `spirv-std` for GPU programming primitives
- Imports shared data structures from `raytracer-shared` crate
- Implements progressive tile-based raytracing with sphere and triangle intersection
- Evaluates PBR materials with simplified BRDF (ready for Microfacet BRDF upgrade)
- Handles material types: diffuse, metallic, glass (transmission), and emissive
- Converts array-based shared structs to Vec3 types for mathematical operations
- Defines compute (`main_cs`), vertex (`main_vs`), and fragment (`main_fs`) shader entry points

## Current Demo Scene

The default scene showcases the material system with:
- **Red diffuse sphere** (material ID 0) - Lambert diffuse
- **Yellow metallic spheres** (material ID 1) - Low roughness metal
- **Blue glass spheres** (material ID 2) - Dielectric with transmission  
- **Blue emissive sphere** (material ID 3) - Acts as area light
- **Red and green triangles** - Different material demonstrations

# Summary instructions

When you are using compact, please focus on test output and code changes

