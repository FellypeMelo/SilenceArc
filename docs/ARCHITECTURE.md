# SilenceArc Architecture: DirectML Edition

> **Version:** 2.0.0 (DirectML Native)  
> **Target Hardware:** Intel Arc GPUs (B-Series / Battlemage)  
> **Platform:** Windows 11 (DirectX 12)

## 🏛️ High-Level Design

SilenceArc follows a decoupled, hybrid architecture designed for ultra-low latency audio processing.

### 1. Audio Backbone (CPU)
-   **Host:** C++ Process.
-   **I/O:** `miniaudio` manages the capture/playback loop with 480-sample hop size.
-   **Buffering:** `AudioStreamBuffer` provides a thread-safe ring buffer for jitter compensation.

### 2. DSP Engine (CPU)
-   **Analysis:** Optimized STFT using a pre-computed DFT basis and Vorbis window.
-   **Bins:** Full 481-bin complex spectrum processing (0 to 24kHz).
-   **Synthesis:** ISTFT with overlap-add reconstruction.

### 3. Neural Engine (GPU via DirectML)
-   **Runtime:** ONNX Runtime with `DmlExecutionProvider`.
-   **Acceleration:** Utilizes Intel Arc XMX (Matrix Extensions) for tensor operations.
-   **Components:**
    -   `OnnxAdapter`: Manages sessions and GPU memory.
    -   `FeatureExtractor`: Native C++ port of ERB filterbank and normalization.
    -   `DirectMLAudioEngine`: Orchestrates the inference pipeline.

## 🔄 Data Flow

1.  **Capture:** 480 samples captured via `miniaudio`.
2.  **DSP Analysis:** Windowing and DFT conversion to 481 complex bins.
3.  **Features:** `FeatureExtractor` computes 32 log-ERB bands and 96 complex features.
4.  **Inference:**
    -   **Encoder:** Processes features, generates 512-dim embedding.
    -   **ERB Decoder:** Predicts 32-band spectral mask.
    -   **DF Decoder:** Predicts 96 complex FIR coefficients.
5.  **Enhancement:**
    -   Apply mask to all frequency bins.
    -   Apply Deep Filtering to the first 96 bins (low-frequency speech recovery).
6.  **Synthesis:** ISTFT and Overlap-Add to restore time-domain audio.

## 🛡️ Reliability & Performance

-   **Deterministic Latency:** Real-time processing budget < 10ms (current: ~4ms).
-   **Driver Stability:** Leverages the stable Windows DirectML driver stack.
-   **Resource Efficiency:** Minimal VRAM footprint (< 128MB).
