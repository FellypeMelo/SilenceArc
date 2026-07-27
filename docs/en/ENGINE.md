# SilenceArc: SYCL/oneDNN Inference Engine

The heart of SilenceArc is its native inference engine, which bypasses high-level runtimes to achieve maximum performance on Intel Arc hardware.

## Weight Mapping & Topology
DeepFilterNet3 consists of 133 distinct weight tensors. The `OneDNNInferenceEngine` maps these PyTorch-exported tensors to native oneDNN primitives.

### Layer Stages:
1.  **Encoder:** 34 primitives including depthwise-separable convolutions and a GRU layer for embedding extraction.
2.  **ERB Decoder:** Reconstructs the Equivalent Rectangular Bandwidth (ERB) mask using transposed convolutions and skip connections.
3.  **DF Decoder:** Calculates the Deep Filtering (DF) coefficients for fine-grained noise removal.

## Memory Layouts & Permutations
A critical challenge in GPU inference is the discrepancy between sequential and spatial memory layouts.

-   **oneDNN Standard:** Prefers **NCHW** [Batch, Channels, Time, Freq] for spatial convolutions.
-   **GRU Standard:** Prefers **TNC** [Time, Batch, Channels] for sequential processing.

Because oneDNN's GPU `reorder` primitive has limitations with complex strides, SilenceArc implements **Custom SYCL Kernels** for these permutations.

### Custom Reorder Kernel (TNC -> NCHW):
The kernel maps sequential GRU output back to spatial dimensions using the formula:
`out[n*(C*T) + c*T + t] = in[t*(N*C) + n*C + c]`

## Unified Shared Memory (USM)
SilenceArc uses **Device USM** for all internal buffers. This allows:
-   **Zero-Copy:** Data is processed in-place on the GPU without intermediate host-side staging.
-   **Direct Access:** SYCL kernels and oneDNN primitives share the same memory pointers, reducing management complexity.

## Inference Flow
1.  **STFT Analysis:** Input audio is windowed and converted to the frequency domain using **oneMKL DFT**.
2.  **Feature Extraction:** Power spectrum and ERB features are calculated via SYCL kernels.
3.  **Engine Execution:** The `OneDNNInferenceEngine` executes the sequential primitive chain (Encoder -> Decoders).
4.  **Coefficient Application:** DF coefficients are applied to the complex frequency bins.
5.  **ISTFT Synthesis:** The filtered signal is converted back to time-domain using an inverse FFT and overlap-add synthesis.

## Hardware Synchronization
The engine uses **In-Order SYCL Queues** to ensure that primitives execute in the correct topological order without the overhead of manual event tracking.
