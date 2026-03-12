# Implementation Plan: Deep Filter Integration (DfIntegration)

## Phase 1: Stabilization & Infrastructure
- [x] **Task: Resolve oneDNN Initialization Blockers**
    - [x] Implement `sycl_reorder_tnc_to_nchw` kernel to replace failing oneDNN reorder primitives.
    - [x] Refactor `add_grouped_linear` to use custom SYCL reorders for layout transitions.
    - [x] Implement zero-copy memory handle sharing in `add_flatten_to_nchw` for dimension changes.
    - [x] Verify engine initializes without "could not create primitive descriptor" errors.
- [x] **Task: Conductor - User Manual Verification 'Phase 1: Stabilization & Infrastructure' (Protocol in workflow.md)**

## Phase 2: Architectural Parity & Mapping
- [x] **Task: Complete Weight Tensor Mapping**
    - [x] Verify all 133 weight tensors are correctly loaded and mapped to corresponding oneDNN primitives.
    - [x] Implement validation for Complex pathway vs ERB pathway coefficient summation logic.
- [x] **Task: Parity Verification (Rust vs SYCL)**
    - [x] Create a parity test case comparing intermediate complex coefficients between Rust and C++/SYCL implementations.
    - [x] Resolve any mismatches causing "robotic" artifacts.
- [x] **Task: Conductor - User Manual Verification 'Phase 2: Architectural Parity & Mapping' (Protocol in workflow.md)**

## Phase 4: Audio Quality Refinement
- [x] **Task: Resolve Robotic Audio Artifacts**
    - [x] Implement 2-frame lookahead compensation in filtering kernel.
    - [x] Correct ERB normalization divisor to 20.0f.
    - [x] Align FIR window taps with model expectations [t-2, t-1, t, t+1, t+2].
    - [x] Verify SNR and RMSE metrics in integration tests.
- [x] **Task: Conductor - User Manual Verification 'Phase 4: Audio Quality Refinement' (Protocol in workflow.md)**
