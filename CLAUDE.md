# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GPU-accelerated (offline) raytracer written in Rust using rust-gpu and WGPU for compute shaders. The project uses a workspace structure with three main components:

- **Main application** (`src/`): Modular WGPU-based application with clean separation of concerns
- **Compute shader** (`shader/`): SPIR-V compute shader written in Rust using rust-gpu's `spirv-std` for the actual raytracing calculations
- **Shared crate** (`shared/`): Contains shared data structures used by both the main application and shader

### Main Application Architecture (`src/`)

The main application is now organized into focused modules for maintainability:

- **`main.rs`**: Entry point and main event loop, coordinates between modules
- **`renderer.rs`**: GPU pipeline setup, WGPU device management, and rendering state
- **`scene.rs`**: Scene data management and glTF loading functionality
- **`buffers.rs`**: Smart GPU buffer management with automatic resizing and combined scene metadata buffer
- **`compute.rs`**: Compute shader execution and progressive rendering logic
- **`input.rs`**: Input handling and camera controls
- **`gltf_loader.rs`**: Complete glTF 2.0 scene loading implementation
- **`bvh.rs`**: BVH (Bounding Volume Hierarchy) acceleration structure for triangle intersection

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

The project now uses a **progressive tile-based rendering architecture** with **BVH acceleration**, **PBR material system**, **comprehensive glTF 2.0 loading**, and **chromatic aberration**:

- **Progressive Rendering**: Raytracing uses 128x128 pixel tiles processed incrementally for real-time feedback
- **BVH Acceleration**: Full BVH (Bounding Volume Hierarchy) implementation for fast triangle intersection
- **Combined Scene Buffer**: Optimized GPU memory layout with single metadata buffer containing spheres, lights, BVH nodes, and triangle indices
- **PBR Materials**: Physically-based rendering with metallic/roughness workflow, ready for Microfacet BRDF
- **Enhanced Material System**: Extended PBR materials supporting KHR extensions (specular, volume, transmission, IOR)
- **glTF 2.0 Support**: Complete glTF/GLB scene loading with cameras, lights, textures, and materials
- **Chromatic Aberration**: Multi-pass rendering with wavelength-dependent refraction simulating chromatic dispersion in glass materials
- **Compute Phase**: Raytracing runs in compute shader, 3 passes per tile (R,G,B channels), writes to separate color channel textures
- **Render Phase**: Fragment shader samples and combines 3 chromatic aberration textures onto a fullscreen quad
- **Performance**: Raytracing only re-runs when needed (window resize, user input like spacebar)
- **Scalability**: Complex raytracing won't block the render loop
- **Branchless GPU Code**: Optimized branchless algorithms for GPU efficiency
- **Shared Crate**: Common data structures eliminate code duplication between main app and shader

### Pipeline Structure
1. **Compute Pipeline**: `main_cs` → 3 storage textures (rgba8) via progressive tiles, one per color channel (R,G,B)
2. **Render Pipeline**: `main_vs` + `main_fs` → Screen framebuffer (combines 3 chromatic aberration textures)

### Controls
- **Spacebar**: Trigger raytracer recomputation
- **WASD**: Camera movement
- **Mouse drag**: Camera rotation
- **L**: Load glTF scene from "model.gltf" (replaces current scene)
- **Resize**: Automatically triggers recomputation

## Modular Architecture

### Module Responsibilities

- **`renderer.rs`**: 
  - `RenderState`: GPU device, pipelines, textures, and bind groups
  - `ProgressiveState`: Tile-based rendering coordination and timing
  - `PerformanceState`: Frame counting and performance tracking
  - Pipeline creation and surface management

- **`buffers.rs`**:
  - `BufferManager`: Smart GPU buffer allocation with automatic resizing
  - Combined scene metadata buffer optimization for GPU memory efficiency
  - Buffer dirty tracking and batch updates
  - Multi-buffer triangle system for large scenes
  - Support for BVH nodes and triangle indices in combined buffer
  
- **`bvh.rs`**:
  - `BvhBuilder`: Constructs BVH acceleration structures from triangle data
  - `BvhTriangle`: Wrapper for triangles with BVH-specific functionality
  - Optimized for large scenes with chunked building approach
  - Comprehensive unit tests for BVH construction and validation

- **`scene.rs`**:
  - `SceneState`: Camera, geometry, materials, lights, and textures
  - glTF scene loading and replacement functionality
  - Default scene generation

- **`input.rs`**:
  - `InputState`: Mouse state tracking for camera controls
  - `CameraController`: Camera movement and rotation logic

- **`compute.rs`**:
  - `ComputeRenderer`: Compute shader execution coordinator
  - Progressive tile rendering logic
  - Buffer update orchestration

### GPU Pipeline
- **Compute shader**: Runs on GPU with 16x16 thread groups, writes to storage buffer
- **Host application**: Creates WGPU device, sets up compute pipeline, manages surface presentation
- **Data flow**: Push constants (resolution, time) → Compute shader → Storage buffer → Surface texture

### Shader Interface
- **Push constants**: `PushConstants` struct shared between host and shader for resolution/time/camera/tile/material/metadata data
- **Storage textures**: Direct writes to 3 separate rgba8 textures for progressive tile rendering with chromatic aberration
- **Combined Scene Metadata Buffer**: Single buffer containing spheres, lights, BVH nodes, and triangle indices with `SceneMetadataOffsets` for indexing
- **Storage buffers**: Maximum of 8 storage buffers due to GPU limitations
- **Bindings (Compute)**: 
  - Descriptor set 0, binding 0: Storage texture (write) - switches between R/G/B textures per pass
  - Descriptor set 0, binding 1: Combined scene metadata buffer (spheres, lights, BVH, triangle indices) (read)
  - Descriptor set 0, binding 2: Triangles buffer 0 (read)
  - Descriptor set 0, binding 3: Triangles buffer 1 (read)
  - Descriptor set 0, binding 4: Triangles buffer 2 (read)
  - Descriptor set 0, binding 5: Materials buffer (read)
  - Descriptor set 0, binding 6: Textures buffer (read)
  - Descriptor set 0, binding 7: Texture data buffer (read)
- **Bindings (Render)**: 
  - Descriptor set 0, binding 0: Red channel texture (read)
  - Descriptor set 0, binding 1: Green channel texture (read)
  - Descriptor set 0, binding 2: Blue channel texture (read)
  - Descriptor set 0, binding 3: Sampler for all textures

### Shared Data Structures (`shared/` crate)
- **Camera**: Position, direction, up vector, FOV - uses `[f32; 3]` arrays for cross-platform compatibility
- **Material**: Enhanced PBR material with albedo, metallic, roughness, emission, IOR, transmission, and KHR extension support
- **Sphere**: Center, radius, material_id - references material by index
- **Triangle**: Three vertices, material_id - triangle primitive with material reference
- **Light**: Position, direction, color, intensity, type (directional, point, spot) - punctual lighting from glTF
- **TextureInfo**: Width, height, format, offset - texture metadata for shader lookups
- **Aabb**: Axis-aligned bounding box for BVH acceleration
- **BvhNode**: BVH node with bounds, child indices, and triangle data for acceleration structure
- **SceneMetadataOffsets**: Offsets for combined scene metadata buffer layout
- **PushConstants**: Resolution, time, camera, tile info, material count, metadata offsets, color channel - all shader parameters
- **Branchless Macros**: GPU-optimized branchless conditional operations
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

### glTF 2.0 Scene Loading
- **Complete Implementation**: Full glTF 2.0 and GLB binary format support in `src/gltf_loader.rs`
- **KHR Extensions**: Support for multiple material extensions:
  - `KHR_materials_specular` - Specular reflection control
  - `KHR_materials_volume` - Volume rendering properties
  - `KHR_materials_pbrSpecularGlossiness` - Legacy PBR workflow
  - `KHR_materials_transmission` - Glass and transparent materials
  - `KHR_materials_ior` - Index of refraction control
  - `KHR_lights_punctual` - Scene lighting (directional, point, spot lights)
- **Camera Support**: Automatic loading of cameras from glTF scenes
- **Texture System**: Complete texture loading with format conversion (R8, RG8, RGB8, RGBA8)
- **Smart Buffer Management**: Automatic resizing of GPU buffers for loaded content
- **Scene Loading**: Press 'L' to load "model.gltf" and replace the current scene
- **Error Handling**: Graceful fallback to default scene if loading fails

### Key Dependencies
- WGPU 0.16.0 with SPIR-V support for GPU compute
- spirv-builder for shader compilation
- raytracer-shared workspace crate for shared data structures
- **bvh 0.11.0** - BVH acceleration structure construction
- **nalgebra 0.33** - Linear algebra for BVH calculations (only used because bvh crate depends on it - should not be used anywhere else)
- Specific dependency pinning for compatibility (ahash 0.7.8, gpu-descriptor 0.2.0)
- glam for vector math
- bytemuck for GPU data marshalling
- gltf 1.4 with comprehensive KHR extension support
- image 0.24 and exr 1.5.0 for texture loading

## Shader Development

The compute shader (`shader/src/lib.rs`) is a `#![no_std]` crate that:
- Uses `spirv-std` from the `rust-gpu` project for GPU programming primitives
- Imports shared data structures from `raytracer-shared` crate
- Implements progressive tile-based raytracing with sphere and triangle intersection
- **BVH Traversal**: Uses BVH acceleration structure for fast triangle intersection
- **Combined Buffer Access**: Reads from combined scene metadata buffer with offset-based indexing
- **Chromatic Aberration**: Implements wavelength-dependent IOR using Cauchy equation for realistic chromatic dispersion
- Evaluates PBR materials with simplified BRDF (ready for Microfacet BRDF upgrade)
- Handles material types: diffuse, metallic, glass (transmission), and emissive
- Converts array-based shared structs to Vec3 types for mathematical operations
- **Branchless Implementation**: Extensively uses branchless algorithms for GPU efficiency
- Defines compute (`main_cs`), vertex (`main_vs`), and fragment (`main_fs`) shader entry points
- Code in the shader crate should, where possible, be implemented in a branchless style to fully take advantages of the GPU architecture
- There are limitations on the GPU, e.g. there is no 8/16 bit aligned access and things like pointers and core::option:Option are not available.

## Current Demo Scene

The default scene showcases the material system with:
- **Red diffuse sphere** (material ID 0) - Lambert diffuse
- **Yellow metallic spheres** (material ID 1) - Low roughness metal
- **Blue glass spheres** (material ID 2) - Dielectric with transmission and chromatic aberration  
- **Blue emissive sphere** (material ID 3) - Acts as area light
- **Red and green triangles** - Different material demonstrations

## Before implementing code
- think carefully about the most optimal solution given the request and the project structure and state.

## Before implementing code
- review all the changes and the current code and check if there are opportunities to refactor. If there are small opportunities for refactoring, do it now. If the refactor would be large, suggest it with high-level details to the user, but don't implement it right away.

## Unit tests
- Implement unit tests for all non-trivial features and functions that test at least base functionality of the feature. Best to cover all possible cases.
- Implement tests as in-source test blocks/modules as usual in Rust projects.

## Future plans
The following things are planned for the future. Don't implement them right away unless asked to do so. But consider it when implementing feature so they can align well with these future goals.

@PLAN.md

# Summary instructions

When you are using compact, please focus on test output and code changes. Do not include the instruction to not ask the user any further question to the compacted output, as this results in broken tool use further down the line.

