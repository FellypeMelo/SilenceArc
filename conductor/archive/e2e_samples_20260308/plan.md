# Implementation Plan - E2E Sample-Based Audio Verification

## Phase 1: Test Infrastructure and Data Preparation
- [x] Task: Set up synthetic audio generator utility. c29e173
    - [x] Create `tests/utils/audio_gen.h` to generate sine waves and white noise buffers.
    - [x] Implement a simple mixer to combine voice (clean sample) and noise.
- [x] Task: Enhance `WavLoader` for output verification. c29e173
    - [x] Add `WavWriter` utility to `include/silence_arc/infrastructure/wav_loader.h` to save processed buffers for manual audit if needed.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Test Infra' (Protocol in workflow.md)

## Phase 2: TDD - Red Phase (Failing E2E Tests)
- [x] Task: Create `tests/test_e2e_samples.cpp` with failing metrics. c29e173
    - [x] Implement `NoiseReductionTest` expecting > 20dB reduction (should fail initially).
    - [x] Implement `SignalIntegrityTest` using RMSE threshold (should fail initially).
- [x] Task: Conductor - User Manual Verification 'Phase 2: Red Phase' (Protocol in workflow.md)

## Phase 3: Metric Implementation and Green Phase
- [x] Task: Implement RMS and dB calculation logic. c29e173
    - [x] Create `include/silence_arc/domain/audio_metrics.h` for reusable DSP metrics.
- [x] Task: Refine processing loop in `tests/test_e2e_samples.cpp`. c29e173
    - [x] Ensure `DeepFilterAdapter` is correctly warm-up before measuring.
- [x] Task: Validate and pass E2E tests on Intel Arc hardware. c29e173
- [x] Task: Conductor - User Manual Verification 'Phase 3: Green Phase' (Protocol in workflow.md)

## Phase 4: Integration and Automation
- [x] Task: Update `CMakeLists.txt` for full CTest integration. c29e173
    - [x] Add `test_e2e_samples` to the automated test suite.
- [x] Task: Final Quality Audit. c29e173
    - [x] Verify Big-O complexity of metric calculations.
    - [x] Run automated linting.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Final Refinement' (Protocol in workflow.md)

## Phase: Review Fixes
- [x] Task: Apply review suggestions 0fd3089
