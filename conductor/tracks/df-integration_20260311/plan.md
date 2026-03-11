# Implementation Plan: Deep Filter Integration (DfIntegration)

## Phase 1: Stabilization & Infrastructure
- [ ] **Task: Resolve oneDNN Initialization Blockers**
    - [ ] Implement `sycl_reorder_tnc_to_nchw` kernel to replace failing oneDNN reorder primitives.
    - [ ] Refactor `add_grouped_linear` to use custom SYCL reorders for layout transitions.
    - [ ] Implement zero-copy memory handle sharing in `add_flatten_to_nchw` for dimension changes.
    - [ ] Verify engine initializes without "could not create primitive descriptor" errors.
- [ ] **Task: Conductor - User Manual Verification 'Phase 1: Stabilization & Infrastructure' (Protocol in workflow.md)**

## Phase 2: Architectural Parity & Mapping
- [ ] **Task: Complete Weight Tensor Mapping**
    - [ ] Verify all 133 weight tensors are correctly loaded and mapped to corresponding oneDNN primitives.
    - [ ] Implement validation for Complex pathway vs ERB pathway coefficient summation logic.
- [ ] **Task: Parity Verification (Rust vs SYCL)**
    - [ ] Create a parity test case comparing intermediate complex coefficients between Rust and C++/SYCL implementations.
    - [ ] Resolve any mismatches causing "robotic" artifacts.
- [ ] **Task: Conductor - User Manual Verification 'Phase 2: Architectural Parity & Mapping' (Protocol in workflow.md)**

## Phase 3: Validation & Quality Assurance
- [ ] **Task: Integration Testing & Metrics**
    - [ ] Run `test_df_integration.exe` and verify SNR improvement >= 15dB.
    - [ ] Validate natural sound preservation in `ErbOnlyProvidesNaturalSound` test case.
- [ ] **Task: Latency Benchmarking**
    - [ ] Measure end-to-end inference latency on Intel Arc GPU.
    - [ ] Ensure latency is within the real-time budget for 48kHz audio.
- [ ] **Task: Conductor - User Manual Verification 'Phase 3: Validation & Quality Assurance' (Protocol in workflow.md)**
