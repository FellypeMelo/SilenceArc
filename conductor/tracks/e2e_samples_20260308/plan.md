# Implementation Plan - E2E Sample-Based Audio Verification

## Phase 1: Test Infrastructure and Data Preparation
- [ ] Task: Set up synthetic audio generator utility.
    - [ ] Create `tests/utils/audio_gen.h` to generate sine waves and white noise buffers.
    - [ ] Implement a simple mixer to combine voice (clean sample) and noise.
- [ ] Task: Enhance `WavLoader` for output verification.
    - [ ] Add `WavWriter` utility to `include/silence_arc/infrastructure/wav_loader.h` to save processed buffers for manual audit if needed.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Test Infra' (Protocol in workflow.md)

## Phase 2: TDD - Red Phase (Failing E2E Tests)
- [ ] Task: Create `tests/test_e2e_samples.cpp` with failing metrics.
    - [ ] Implement `NoiseReductionTest` expecting > 20dB reduction (should fail initially).
    - [ ] Implement `SignalIntegrityTest` using RMSE threshold (should fail initially).
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Red Phase' (Protocol in workflow.md)

## Phase 3: Metric Implementation and Green Phase
- [ ] Task: Implement RMS and dB calculation logic.
    - [ ] Create `include/silence_arc/domain/audio_metrics.h` for reusable DSP metrics.
- [ ] Task: Refine processing loop in `tests/test_e2e_samples.cpp`.
    - [ ] Ensure `DeepFilterAdapter` is correctly warm-up before measuring.
- [ ] Task: Validate and pass E2E tests on Intel Arc hardware.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Green Phase' (Protocol in workflow.md)

## Phase 4: Integration and Automation
- [ ] Task: Update `CMakeLists.txt` for full CTest integration.
    - [ ] Add `test_e2e_samples` to the automated test suite.
- [ ] Task: Final Quality Audit.
    - [ ] Verify Big-O complexity of metric calculations.
    - [ ] Run automated linting.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Refinement' (Protocol in workflow.md)
