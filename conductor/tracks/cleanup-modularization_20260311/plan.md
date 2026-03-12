# Implementation Plan: Project Cleanup and Modularization

## Phase 1: Infrastructure Isolation & Decoupling
This phase focuses on separating the core SYCL/oneDNN infrastructure from the high-level audio processing logic.

- [ ] Task: Analysis - Identify tight coupling in `SYCLAccelerator` and `OneDNNInferenceEngine`.
    - [ ] Map all SYCL-specific types used in public interfaces.
    - [ ] Define abstract interfaces for `AudioProcessor` and `InferenceEngine`.
- [ ] Task: TDD - Create regression benchmarks for latency and SNR.
    - [ ] Ensure `test_df_integration` is robust and captures current baseline.
- [ ] Task: Refactor - Implement abstract interfaces and move SYCL/oneDNN logic to implementation files.
    - [ ] Update `SYCLAccelerator` to implement a generic `IAudioProcessor`.
    - [ ] Update `OneDNNInferenceEngine` to implement a generic `INeuralEngine`.
- [ ] Task: Conductor - User Manual Verification 'Infrastructure Isolation' (Protocol in workflow.md)

## Phase 2: Domain Refactoring & SRP
This phase focuses on breaking down large functions and improving code density in the domain layer.

- [ ] Task: Refactor - Modularize `SYCLAccelerator::process_frame`.
    - [ ] Extract feature extraction, inference coordination, and synthesis into private helper methods.
- [ ] Task: Refactor - Clean up `OneDNNInferenceEngine` weight loading and layer setup.
    - [ ] Simplify layer creation logic using the "Command" or "Builder" pattern where appropriate.
- [ ] Task: TDD - Verify each refactored method with unit tests.
    - [ ] Add unit tests for extracted DSP helpers in `test_kernel_correctness.cpp`.
- [ ] Task: Conductor - User Manual Verification 'Domain Refactoring' (Protocol in workflow.md)

## Phase 3: UI & Utility Modularization (KISS & YAGNI)
This phase addresses the UI and general project structure, removing redundant code.

- [ ] Task: Refactor - Modularize ImGui management.
    - [ ] Separate telemetry readout logic from UI rendering logic.
- [ ] Task: Cleanup - Apply YAGNI cleanup across the project.
    - [ ] Remove unused variables, deprecated warnings, and speculative comments.
    - [ ] Simplify header includes to reduce build times.
- [ ] Task: TDD - Run full regression suite.
    - [ ] Verify UI remains responsive and telemetry data is accurate.
- [ ] Task: Conductor - User Manual Verification 'UI & Utility Modularization' (Protocol in workflow.md)

## Phase 4: Final Validation & Standards Alignment
Final sweep to ensure all AI-XP standards are met.

- [ ] Task: Audit - Review entire codebase for SOLID compliance.
    - [ ] Verify no infrastructure leaks into the domain layer.
- [ ] Task: Documentation - Update `ARCHITECTURE.md` to reflect new modular structure.
- [ ] Task: Final Check - Ensure clean build with zero warnings.
- [ ] Task: Conductor - User Manual Verification 'Final Validation' (Protocol in workflow.md)
