# Specification: Deep Application Documentation

## Overview
Create a comprehensive, modular documentation suite for SilenceArc. This documentation will serve as both a technical wiki and a professional "whitepaper" describing the world's first native, real-time noise suppression engine built specifically for Intel Arc GPUs using pure SYCL and oneDNN (bypassing OpenVINO).

## Target Audience
- **End Users (Streamers/Gamers):** High-level benefits and simple setup.
- **Developers (Technical Deep-Dive):** In-depth logic of the GPU engine and DSP pipeline.
- **Maintainers (Architectural):** Design decisions, weight mapping, and layout permutations.
- **Intel Enthusiasts:** Showcasing the power of Arc/oneAPI ecosystem.

## Scope of Documentation
1.  **Project Philosophy:** Why Intel Arc? Why bypass OpenVINO? The "native performance" mandate.
2.  **Architecture Deep-Dive:**
    - High-level C4-style overview.
    - SYCL/oneDNN interop layer.
    - DeepFilterNet3 topology mapping (Encoder/Decoders).
    - Custom SYCL kernels for memory permutations (TNC <-> NCHW).
3.  **Engine Logic:** Detailed walkthrough of USM, zero-copy buffers, and primitive execution flow.
4.  **Setup & Build Guide:** Compiler requirements (Intel LLVM), environment variables, and CMake integration.
5.  **User Guide:** Signal monitoring, device selection, and attenuation limits.

## Functional Requirements
- **Modular Wiki Style:** Documentation divided into logical parts (e.g., `DOCS/ARCHITECTURE.md`, `DOCS/ENGINE.md`, `DOCS/SETUP.md`).
- **Technical Paper Component:** A synthesized summary (`WHITE_PAPER.md`) explaining the innovation and results.
- **Step-by-Step Breakdown:** Complex concepts explained in digestible phases.

## Acceptance Criteria
- [ ] Complete set of modular Markdown files in a dedicated `docs/` or `wiki/` directory.
- [ ] Clear explanation of weight mapping logic (133 tensors from PyTorch).
- [ ] Documentation of the custom SYCL reorder kernel stride formulas.
- [ ] Professional `WHITE_PAPER.md` summarizing the project's value proposition.
- [ ] All documentation matches the current working state of the C++ codebase.
