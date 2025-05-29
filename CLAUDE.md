# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GPU-accelerated raytracer written in Rust using rust-gpu and WGPU for compute shaders. The project uses a workspace structure with two main components:

- **Main application** (`src/main.rs`): WGPU-based application that sets up compute pipeline, manages window/surface, and handles the main event loop
- **Compute shader** (`shader/`): SPIR-V compute shader written in Rust using rust-gpu's `spirv-std` for the actual raytracing calculations

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

The project now uses a **decoupled compute/render architecture**:

- **Compute Phase**: Raytracing runs in compute shader, writes directly to storage texture
- **Render Phase**: Fragment shader samples the raytraced texture onto a fullscreen quad
- **Performance**: Raytracing only re-runs when needed (window resize, user input like spacebar)
- **Scalability**: Complex raytracing won't block the render loop

### Pipeline Structure
1. **Compute Pipeline**: `main_cs` → Storage texture (rgba8)
2. **Render Pipeline**: `main_vs` + `main_fs` → Screen framebuffer

### Controls
- **Spacebar**: Trigger raytracer recomputation
- **Resize**: Automatically triggers recomputation

## Architecture

### GPU Pipeline
- **Compute shader**: Runs on GPU with 16x16 thread groups, writes to storage buffer
- **Host application**: Creates WGPU device, sets up compute pipeline, manages surface presentation
- **Data flow**: Push constants (resolution, time) → Compute shader → Storage buffer → Surface texture

### Shader Interface
- **Push constants**: `PushConstants` struct shared between host and shader for resolution/time data
- **Storage buffer**: Vec4 array for pixel output from compute shader
- **Binding**: Single storage buffer binding at descriptor set 0, binding 0

### Key Dependencies
- WGPU 0.16.0 with SPIR-V support for GPU compute
- spirv-builder for shader compilation
- Specific dependency pinning for compatibility (ahash 0.7.8, gpu-descriptor 0.2.0)
- glam for vector math

## Shader Development

The compute shader (`shader/src/lib.rs`) is a `#![no_std]` crate that:
- Uses `spirv-std` for GPU programming primitives
- Defines the main compute entry point as `main_cs`
- Currently implements a simple test pattern (will be extended to raytracing)
- Shares `PushConstants` struct with host application for parameter passing

# Summary instructions

When you are using compact, please focus on test output and code changes

