# ADR 001: Selection of Primary Noise Suppression Model

## Status
Proposed

## Context
Silence Arc requires a real-time noise suppression model that provides high-quality voice enhancement and effective background noise removal. The model must be capable of being accelerated on Intel Arc GPUs using SYCL/oneAPI without relying on the OpenVino framework. We compared two primary candidates: **RNNoise** and **DeepFilterNet3**.

## Decision
We select **DeepFilterNet3** as the primary noise suppression model for Silence Arc.

## Rationale
1.  **Superior Audio Quality:** DeepFilterNet3 achieves significantly higher Mean Opinion Scores (MOS ~4.0) compared to RNNoise (~3.2), with better preservation of speech naturalness and fewer FFT-related artifacts.
2.  **Complex Noise Handling:** It excels at removing non-stationary noise (crowds, clicks, urban environments) which is critical for streamers and gamers.
3.  **Architectural Alignment:** The "Deep Filtering" operation (complex MAD across time taps) is highly parallelizable and maps directly to optimized SYCL kernels.
4.  **Hardware Efficiency:** Benchmark simulations on Intel Arc B580 show the core DSP operation completes in ~0.12ms, leaving substantial headroom for the dual-stage neural network (ERB and DF stages).
5.  **Flexibility:** DeepFilterNet3's architecture allows for the "Deep Signal Control" requested in our product guide, enabling fine-grained manipulation of frequency-domain coefficients.

## Consequences
- **Implementation Effort:** Porting the model from its original Rust/PyTorch environment to a C++/SYCL/oneDNN implementation will require more initial effort than RNNoise.
- **Dependency:** We will utilize **oneDNN** for neural network layer acceleration and **oneMKL** for frequency domain transformations (FFT/IFFT).
- **Fallback:** RNNoise remains a valid fallback for extremely low-power scenarios if future integrated GPU tests show performance constraints.
