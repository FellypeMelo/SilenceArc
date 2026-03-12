# Specification: Deep Filter Integration (DfIntegration)

## Overview
Implement and stabilize the DeepFilterNet3 integration in Silence Arc using the SYCL/oneDNN inference engine. The track focuses on resolving initialization crashes and eliminating "robotic" audio artifacts by ensuring architectural parity with the original model and implementing robust, artifact-free signal processing on Intel Arc GPUs.

## Functional Requirements
- **Stable Initialization**: Resolve the "could not create a primitive descriptor for the reorder primitive" error by replacing problematic oneDNN reorders with custom SYCL kernels or memory views.
- **Hybrid Reshaping**: Implement a hybrid approach for tensor reshaping:
    - Use oneDNN memory handle sharing (zero-copy) for simple flattening/reshaping.
    - Implement custom SYCL kernels for complex layout transitions (e.g., TNC to NCHW) that oneDNN primitives fail to handle.
- **Signal Parity**: Correctly implement the Complex and ERB pathways in the DF decoder, ensuring proper summation of coefficients to avoid phase distortions or robotic artifacts.
- **Weight Mapping**: Validate the mapping of all 133 DeepFilterNet3 weight tensors to oneDNN primitives.

## Non-Functional Requirements
- **Audio Quality**: Achieve natural-sounding voice processing with zero audible FFT artifacts.
- **Latency**: Maintain inference latency suitable for real-time 48kHz audio processing on Intel Arc hardware.
- **Parity**: Aim for bit-exact or near-bit-exact results compared to the Rust `libDF` implementation.

## Acceptance Criteria
- `test_df_integration.exe` initializes and passes all integration tests on Intel Arc GPU.
- SNR (Signal-to-Noise Ratio) improvement >= 15dB in standard noise suppression benchmarks.
- Natural speech test: Processed clean speech remains perceptually natural and free of metallic/robotic artifacts.
- Architectural mapping documentation between PyTorch weights and oneDNN layers is completed.

## Out of Scope
- Optimization for non-Intel hardware.
- Integration with OpenVino or other alternative inference backends.
- UI development for DF toggles (focus is on the engine implementation).
