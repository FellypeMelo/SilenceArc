# SilenceArc: Native GPU-Accelerated Audio Intelligence

SilenceArc is a high-performance, real-time noise suppression and voice enhancement application designed specifically for **Intel Arc GPUs**. By bypassing high-level runtimes like OpenVINO and communicating directly with the hardware via **SYCL** and **oneDNN**, SilenceArc achieves single-digit millisecond latency and extreme resource efficiency.

## 🚀 Key Features

-   **Native Intel Arc Acceleration:** Leverages Xe Matrix eXtensions (XMX) for lightning-fast AI inference.
-   **Direct oneAPI Integration:** Built using pure SYCL and oneDNN primitives—no Python, no heavy wrappers.
-   **Zero-Copy Memory:** Utilizes Unified Shared Memory (USM) for maximum throughput between CPU and GPU.
-   **Real-Time Perceptual Quality:** Integrates the state-of-the-art **DeepFilterNet3** model for superior voice clarity.
-   **Ultra-Low Latency:** Optimized for live streaming, gaming, and professional vocal monitoring.
-   **Minimalist GUI:** Lightweight interface with real-time telemetry, signal levels, and system tray integration.

---

## 🛠️ Tech Stack

-   **Core Language:** C++ (C++20)
-   **GPU Backend:** SYCL / Intel oneAPI
-   **Neural Primitives:** oneDNN (oneAPI Deep Neural Network Library)
-   **DSP Math:** oneMKL (oneAPI Math Kernel Library)
-   **Audio Pipeline:** miniaudio (WASAPI / ASIO)
-   **Model Logic:** DeepFilterNet3 (Rust-based core with C++ Adapter)
-   **UI Framework:** Dear ImGui (DX11/DX12 backend)
-   **Build System:** CMake + Ninja

---

## 📋 Prerequisites

To build and run SilenceArc, you need the following:

1.  **Hardware:** An Intel Arc GPU (B-Series/Battlemage or A-Series/Alchemist).
2.  **Compiler:** `icx` (Intel LLVM C++ Compiler) from the **Intel oneAPI Base Toolkit** (2024.0+).
3.  **Libraries:** oneDNN and oneMKL (included in oneAPI Base Toolkit).
4.  **CMake:** Version 3.20 or newer.
5.  **Rust:** (Optional) Only required if you need to recompile the `df.dll` core.

---

## 🏁 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/FellypeMelo/SilenceArc.git
cd SilenceArc
```

### 2. Configure Environment
Initialize the oneAPI environment variables (required for the compiler and libraries):
```powershell
.\setup_intel.bat
```

### 3. Build the Application
We recommend using the Ninja generator for high-speed builds:
```bash
mkdir build
cd build
cmake -G "Ninja" -DCMAKE_CXX_COMPILER=icx -DCMAKE_C_COMPILER=icx ..
cmake --build . --config Release
```

### 4. Run SilenceArc
```bash
cd ..
.\run.bat
```

---

## 🏗️ Architecture Overview

SilenceArc follows a **Clean Architecture** approach, separating hardware-specific acceleration from high-level application logic.

### Directory Structure
```
├── docs/               # In-depth technical documentation
├── include/            # C++ Header files
│   └── silence_arc/
│       ├── domain/     # Core interfaces (Audio, GPU, NN)
│       └── infrastructure/ # Implementations (SYCL, oneDNN, miniaudio)
├── src/                # Implementation files
├── models/             # DeepFilterNet3 weights and metadata
├── scripts/            # Utility scripts (Weight export, etc.)
├── tests/              # SYCL and NN unit tests
└── DeepFilterNet/      # Submodule for the Rust perceptual core
```

### Data Flow
1.  **Audio Capture:** `miniaudio` captures raw buffers via WASAPI/ASIO.
2.  **Analysis:** Signal is windowed and converted to frequency domain via **oneMKL DFT**.
3.  **Inference:** The **OneDNNInferenceEngine** executes the 133-tensor pipeline on the **Arc GPU**.
4.  **Permutation:** Custom SYCL kernels handle layout transitions between sequential (TNC) and spatial (NCHW) memory.
5.  **Synthesis:** ISTFT and Overlap-Add reconstruction via SYCL kernels.
6.  **Playback:** Processed audio is pushed back to the output device.

---

## 🧠 The Engine: Why Native SYCL?

Most noise suppression tools use generic runtimes like OpenVINO. SilenceArc chooses a harder, more powerful path:

-   **Layout Mastery:** We wrote custom kernels to handle sequential GRU outputs that oneDNN's standard reorder couldn't process efficiently on GPUs.
-   **Weight Mapping:** Every tensor from the DeepFilterNet3 PyTorch model is mapped bit-exactly to a oneDNN primitive.
-   **Total Control:** By owning the SYCL queue, we can interleave custom DSP logic with neural layers without pipeline stalls.

For more details, see the [Whitepaper](./WHITE_PAPER.md) or the [Engine Deep-Dive](./docs/ENGINE.md).

---

## 🎮 Usage Guide

### Interface Basics
-   **Input/Output:** Select your microphone and speakers. Note that the app uses **Exclusive Mode** for ultra-low latency.
-   **Attenuation Limit:** Sets the noise floor. 20dB sounds natural; 100dB provides absolute silence.
-   **Telemetry:** Monitor your real-time **GPU Utilization** and **Processing Latency**.

### System Tray
Minimize the application to the tray to keep it running in the background while you focus on your work or game.

---

## 🧪 Verification & Testing

To ensure your hardware is fully compatible, run the neural layer verification test:
```bash
.\build\test_nn_layers.exe
```
This test initializes the SYCL engine, loads 133 tensors, and executes a full inference cycle on your GPU.

---

## 📄 License

SilenceArc is licensed under the **Apache License 2.0**. See the [LICENSE](./LICENSE) file for details.

---

**SilenceArc** — Silence the noise, amplify the voice. Built for the Intel Arc era.
