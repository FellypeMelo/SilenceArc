# SilenceArc: DirectML/ONNX Inference Engine

The SilenceArc engine manages the real-time execution of the DeepFilterNet3 neural models using **DirectML** via **ONNX Runtime**.

## 🚀 Engine Components

### 1. OnnxAdapter
A robust wrapper for the ONNX Runtime C++ API. It handles:
-   **Execution Provider Management:** Configures `DmlExecutionProvider` for GPU acceleration.
-   **Dynamic Buffer Management:** Pre-allocates and manages GPU-resident tensors.
-   **Multi-Session Coordination:** Orchestrates the Encoder, ERB Decoder, and DF Decoder sessions.

### 2. FeatureExtractor (Native C++)
A high-performance port of the DeepFilterNet feature extraction logic:
-   **ERB Filterbank:** Efficiently calculates energy across 32 Equivalent Rectangular Bandwidth bands.
-   **Adaptive Normalization:** Implements exponential moving average normalization for both ERB and complex features, preserving voice transients.

### 3. DirectMLAudioEngine
The top-level orchestrator that implements the `IAudioProcessor` interface:
-   **Lookahead Handling:** Manages a 5-frame history buffer to support the 2-frame lookahead required by DeepFilterNet3.
-   **Spectral Fusion:** Combines the results of ERB Masking and Deep Filtering based on the frequency bin index.

## 🛠️ Performance Tuning

### DirectML XMX Utilization
The engine is configured to use the `ORT_SEQUENTIAL` execution mode and disables memory pattern optimization to ensure maximum stability on Intel Arc hardware. It utilizes the XMX (Matrix Extensions) for tensor operations, resulting in sub-4ms inference times.

### CPU/GPU Sync
Data capture and DSP analysis occur on the CPU thread, while the heavy neural computation is offloaded to the GPU. This parallel pipeline ensures that the audio thread is never blocked by GPU stalls.
