# ADR 001: Selection of Primary Noise Suppression Model

## Status
Accepted

## Context
Silence Arc requires a real-time noise suppression model that provides high-quality voice enhancement and effective background noise removal. The model must be capable of being accelerated on Intel Arc GPUs using SYCL/oneAPI without relying on the OpenVino framework. We compared two primary candidates: **RNNoise** and **DeepFilterNet3**.

## Decision
We select **DeepFilterNet3** as the primary noise suppression model for Silence Arc.

## Rationale
1.  **Audio quality reported in the upstream literature:** the DeepFilterNet3 research (vendored under `DeepFilterNet/`, upstream paper linked from `DeepFilterNet/README.md`) reports Mean Opinion Score (MOS) improvements over RNNoise-class baselines, with better preservation of speech naturalness and fewer FFT-related artifacts. These are the published authors' figures from the DeepFilterNet3 paper, not a MOS evaluation SilenceArc has run itself — this project has not reproduced or independently measured a MOS comparison.
2.  **Complex Noise Handling:** It excels at removing non-stationary noise (crowds, clicks, urban environments) which is critical for streamers and gamers.
3.  **Architectural Alignment:** The "Deep Filtering" operation (complex MAD across time taps) is highly parallelizable and maps directly to optimized SYCL kernels.
4.  **Hardware efficiency — a design target, not a measured result:** early informal benchmark simulations during model selection suggested the core DSP operation could complete in the sub-millisecond range on Intel Arc B580, leaving headroom for the dual-stage neural network (ERB and DF stages). No benchmark script, dataset, or output backing a specific number is committed to this repository, so this figure motivated the choice rather than confirming it after the fact. `tests/bench_pipeline_latency.cpp` is the mechanism that could produce a real, committed p50/p99 latency figure for this pipeline against a 10ms budget; as of this writing its output has not been committed.
5.  **Flexibility:** DeepFilterNet3's architecture allows for the "Deep Signal Control" requested in our product guide, enabling fine-grained manipulation of frequency-domain coefficients.

## Consequences
- **Implementation Effort:** Porting the model from its original Rust/PyTorch environment to a C++/SYCL/oneDNN implementation will require more initial effort than RNNoise.
- **Dependency:** We will utilize **oneDNN** for neural network layer acceleration and **oneMKL** for frequency domain transformations (FFT/IFFT).
- **Fallback:** RNNoise remains a valid fallback for extremely low-power scenarios if future integrated GPU tests show performance constraints.
