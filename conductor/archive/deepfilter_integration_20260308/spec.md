# Specification - DeepFilterNet Integration and E2E Testing

## Overview
This track focuses on the full integration of the DeepFilterNet noise suppression engine into the Silence Arc C++ application and the establishment of a robust End-to-End (E2E) testing suite to verify performance, quality, and UI responsiveness.

## Functional Requirements
- **Core Integration:** Finalize the connection between the C++ infrastructure and the Rust-based DeepFilterNet using the existing `deep_filter_adapter`.
- **Asynchronous Audio Pipeline:** Implement audio processing within a high-priority asynchronous callback thread to ensure real-time performance without blocking the GUI.
- **E2E Test Suite:**
    - **Noise Reduction Verification:** Quantify the reduction in noise floor between input and output buffers.
    - **Latency Benchmarking:** Measure and report the round-trip latency from capture to render.
    - **GUI Signal Feedback:** Ensure the `ui_manager` correctly reflects real-time signal levels and filtering status during active suppression.
    - **Audio Loopback:** Implement an automated loopback test using WASAPI/ASIO to simulate real-world capture and playback.

## Non-Functional Requirements
- **Real-time Performance:** Processing must stay within the audio buffer deadline (e.g., < 10ms for a 48kHz/480 sample buffer).
- **Stability:** The asynchronous callback must be thread-safe and resilient to buffer overflows or underflows.
- **Resource Efficiency:** GPU utilization for inference should be optimized for Intel Arc architecture.

## Acceptance Criteria
- [ ] DeepFilterNet successfully processes raw audio buffers provided by the C++ adapter.
- [ ] Automated E2E tests pass for both noise reduction levels and latency thresholds.
- [ ] GUI displays real-time telemetry (latency, dB reduction) during the filtering process.
- [ ] Custom WAV files are correctly loaded and used as the source for automated testing.

## Out of Scope
- Integration of other noise suppression models (e.g., RNNoise).
- Advanced voice enhancement (EQ, compression) beyond standard DeepFilterNet capabilities.
- Final production installer or system tray implementation (to be handled in a separate track).
