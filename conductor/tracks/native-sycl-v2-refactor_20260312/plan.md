# Implementation Plan: Native SYCL Engine v2 & Project Refactor

## Phase 1: Foundation & Modularization (DONE)
- [x] Create directory structure `src/infrastructure/sycl/`.
- [x] Implement `SyclMemoryManager` using USM (Unified Shared Memory).
- [x] Define the `Tensor` and `Layer` abstractions.
- [x] Refactor core interfaces to `sa::` namespace and `IAudioProcessor`.
- **Checkpoint:** Unit test for USM allocation and basic SYCL kernel execution PASSED.

## Phase 2: Optimized DSP Engine (DONE)
- [x] Integrate manual stable SYCL DFT (Full signal integrity verified).
- [x] Implement the **Truncation Kernel** (481 -> 480 bins).
- [x] Implement **Overlap-Add (OLA)** logic with correct synthesis scaling.
- [x] Refactor `AudioProcessor` to work with the new DSP engine.
- **Checkpoint:** Signal loopback test (Input -> STFT -> ISTFT -> Output) with 100% integrity PASSED.

## Phase 3: Data-Driven Layer Implementation (DONE)
- [x] Implement `SyclGraphBuilder` to auto-map weights from `models/df3_weights`.
- [x] Implement oneDNN wrappers for:
    - [x] `Linear`
    - [x] `GroupedLinear`
    - [x] `Conv2d` / `ConvTranspose2d`
    - [x] `BinaryAdd`
- [x] Implement `NormalizationLayer` (ERB log-scale).
- **Checkpoint:** Verify single-layer output against Rust reference PASSED.

## Phase 4: Full Path Assembly (IN PROGRESS)
- [ ] Assemble **Full Encoder** path.
- [ ] Implement **GRU Cell** state management in SYCL.
- [ ] Assemble **ERB Decoder** and **DF Decoder** paths.
- [ ] Handle complex number alignment in the DF path.
- **Checkpoint:** Full inference pass on a static buffer with mask verification.

## Phase 5: Integration & Global Refactor (HARD)
- [ ] Switch `main.cpp` to `NativeSyclEngine`.
- [ ] Implement "Lookahead Compensation" in the SYCL queue.
- [ ] Global "Clean Code" sweep: remove dead code, unify naming conventions.
- [ ] Verify telemetry still accurately reports Arc B580 load during native inference.
- **Checkpoint:** Application running fully native with telemetry and real-time audio.
