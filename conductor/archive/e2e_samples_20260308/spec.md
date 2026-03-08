# Specification - E2E Sample-Based Audio Verification

## Overview
This track implements a comprehensive End-to-End (E2E) testing suite for Silence Arc, using both pre-recorded audio samples and programmatically generated synthetic audio. The goal is to verify the noise suppression quality, signal integrity, and real-time performance specifically on Intel Arc hardware.

## Functional Requirements
- **Sample Processing:** The test suite must load existing .wav samples from `tests/samples/` and process them through the full DeepFilterNet pipeline.
- **Synthetic Generation:** Implement a mechanism to generate synthetic voice-plus-noise signals for controlled testing of specific noise profiles (e.g., white noise, fan hum).
- **Metric Calculation:**
    - **dB Reduction:** Measure the noise floor reduction in the non-voice sections of the processed samples.
    - **Signal Integrity:** Use RMSE (Root Mean Square Error) or a similar spectral comparison to ensure the voice component is not distorted.
    - **Latency Monitoring:** Accurately measure the round-trip time from buffer input to processed output.
- **CTest Integration:** Integrate the E2E tests into the existing CMake/CTest environment for automated execution.

## Non-Functional Requirements
- **Hardware Target:** Tests are optimized for and must run on Intel Arc GPUs using SYCL/oneAPI.
- **Performance Budget:** Total processing latency for a 10ms frame must remain < 10ms to qualify as a pass.
- **Determinism:** Synthetic tests must use seeded randomness to ensure repeatable results across runs.

## Acceptance Criteria
- [ ] Automated execution via `ctest` returns a pass/fail status based on defined thresholds.
- [ ] Processed output for `High-Noise.wav` shows at least 20dB reduction in stationary noise.
- [ ] RMSE between processed synthetic voice and original clean reference remains below a 0.05 threshold.
- [ ] Average callback latency measured during E2E runs is < 5ms on Intel Arc hardware.

## Out of Scope
- Support for non-Intel GPU architectures in this specific test suite.
- Integration with live audio hardware (physical loopback); these tests are file/buffer-based simulations of the E2E path.
