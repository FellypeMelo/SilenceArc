# SilenceArc (DirectML Edition)

SilenceArc is a high-performance, real-time noise suppression and voice enhancement application designed specifically for **Intel Arc GPUs**. By utilizing **DirectML** and **ONNX Runtime**, SilenceArc achieves single-digit millisecond latency and extreme resource efficiency on Windows 11.

## 🚀 Key Features

-   **DeepFilterNet3 Integration:** State-of-the-art neural noise suppression with Deep Filtering.
-   **DirectML Acceleration:** Optimized for Intel Arc B580 (Battlemage) and other modern GPUs via Microsoft's ML API.
-   **Ultra-Low Latency:** Optimized C++ implementation with native feature extraction and optimized DSP.
-   **No Heavy Wrappers:** Direct interaction with the GPU via ONNX Runtime native C++ API.

## 🏛️ Architecture

SilenceArc follows a hybrid CPU/GPU architecture:

1.  **Audio Pipeline (CPU):** Powered by `miniaudio`, handles low-latency capture and playback.
2.  **DSP Core (CPU):** Performs STFT (Short-Time Fourier Transform) using an optimized DFT basis and Vorbis windowing.
3.  **Neural Backbone (GPU):** Executes the DeepFilterNet3 models (Encoder, ERB/DF Decoders) using **DirectML**.
4.  **Feature Extraction:** Native C++ port of DeepFilterNet's ERB filterbank and adaptive normalization.

## 🛠️ Requirements

-   **OS:** Windows 10/11 (64-bit)
-   **GPU:** Intel Arc B-Series (recommended) or any DirectX 12 compatible GPU.
-   **Compiler:** Intel oneAPI ICX or MSVC 2022.
-   **Libraries:** ONNX Runtime with DirectML support.

## 📦 Project Structure

```text
G:/Programas/SilenceArc/
├── include/            # C++ Header files
├── src/                # Implementation files
│   ├── domain/         # Business logic & interfaces
│   └── infrastructure/ # DirectML, ONNX, and Audio implementations
├── models/onnx/        # DeepFilterNet3 ONNX models
├── tests/              # Unit and integration tests
└── third_party/        # External dependencies (onnxruntime, imgui, miniaudio)
```

## 🏗️ Building from Source

1.  Initialize the environment: `setup_environment.bat`
2.  Build the project: `build_project.bat`
3.  Run the application: `build\silence_arc.exe`

## 🧠 Why DirectML?

DirectML provides a stable, vendor-agnostic high-performance path for machine learning on Windows. It ensures that SilenceArc remains stable across driver updates and provides native access to the Intel Arc AI hardware without the complexity of low-level kernel management.

---
**License:** MIT
**Version:** 2.0.0
