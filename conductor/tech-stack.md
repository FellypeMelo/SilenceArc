# Silence Arc - Technology Stack

## Core Language & Runtime
- **Primary Language:** C++ (C++20/23)
- **Rationale:** Absolute performance, direct memory management, and first-class support for audio and GPU compute APIs.

## GPU Compute & AI Inference
- **Backend:** SYCL / Intel oneAPI
- **Direct Hardware Access:** Level Zero (via SYCL)
- **Rationale:** Native acceleration for Intel Arc GPUs without the overhead or constraints of OpenVino. Allows for custom compute kernels for real-time DSP.

## Audio Processing
- **Primary API:** WASAPI (Windows Audio Session API)
- **Mode:** Exclusive Mode (where possible) for ultra-low latency.
- **Rationale:** Native Windows standard providing high-fidelity, low-latency audio capture and playback.

## User Interface (GUI)
- **Framework:** Dear ImGui
- **Rendering:** DX11 or DX12 (aligned with Windows/Arc ecosystem)
- **Rationale:** Extremely low memory footprint and CPU overhead. Ideal for a technical tool that stays resident in the background.

## Infrastructure & Build
- **Build System:** CMake (with oneAPI integration)
- **Version Control:** Git
- **Documentation:** Markdown (following AI-XP standards)
