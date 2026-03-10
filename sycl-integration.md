# SYCL Integration & Rust Bridge Plan

## Goal
Establish a reliable SYCL build environment, replace GTest with a lightweight harness for GPU tests, and architect the Rust-to-C++ bridge for GPU acceleration.

## Tasks
- [x] Task 1: Create `tests/sycl_test_harness.h` (header-only, zero dependencies) → Verify: File exists with basic assertion macros.
- [x] Task 2: Migrate `tests/test_sycl_discovery.cpp` to use the new harness → Verify: Compiles with `icx -fsycl`.
- [x] Task 3: Patch `CMakeLists.txt` to bypass broken `find_package(IntelSYCL)` and link SYCL manually → Verify: `cmake` configuration completes.
- [x] Task 4: Implement `GPUAccelerator` domain interface and SYCL implementation → Verify: Clean Architecture separation (Domain vs Infra).
- [x] Task 5: Design FFI Bridge in `DeepFilterNet/libDF/src/gpu_bridge.rs` → Verify: `extern "C"` signatures match C++ implementation.
- [x] Task 6: [TDD RED] Run `test_sycl_discovery` → Verify: Fails if Arc GPU is not found or environment is misconfigured.
- [x] Task 7: [TDD GREEN] Fix environment/code until `test_sycl_discovery` passes → Verify: Console output shows Intel Arc GPU detected.

## Phase 2: Core Kernel Porting & oneDNN Integration
- [x] Task 8: Integrate oneDNN (DNNL) into the build system.
- [x] Task 9: Port `STFT` (Analysis) logic to SYCL kernels.
- [x] Task 10: Port `ISTFT` (Synthesis) logic to SYCL kernels.
- [x] Task 11: Implement `Deep Filtering` frequency-domain convolution to SYCL.
- [x] Task 12: [TDD RED] Implement `test_kernel_correctness` using synthetic signal → Verify: SYCL output matches baseline (MSE < 1e-13).
- [x] Task 13: [TDD GREEN] Fix kernel logic until `test_kernel_correctness` passes.

## Phase 3: Neural Network Porting (oneDNN)
- [ ] Task 14: Map `DeepFilterNet3` layers (Convolutions, GRU/Linear) to oneDNN primitives.
- [ ] Task 15: Implement weights loading from `tract` exported tensors to oneDNN buffers.
- [ ] Task 16: Port the Encoder inference to GPU.
- [ ] Task 17: Port the Decoders (ERB and DF) to GPU.
- [ ] Task 18: [TDD RED] Verify full inference on GPU against Rust/Tract baseline.
- [ ] Task 19: [TDD GREEN] Optimize data flow and batching.

## Done When
- [x] `test_sycl_discovery` runs successfully without GTest conflicts.
- [x] `GPUAccelerator` interface is defined in `domain`.
- [x] Build system is stable using `icx -fsycl`.
- [x] STFT/ISTFT and Deep Filtering kernels are functional on GPU.

## Notes
- AI-XP: TDD is mandatory. Phase RED must be proven.
- Clean Architecture: `GPUAccelerator` (Domain) -> `SYCLAccelerator` (Infra).
- USM (Unified Shared Memory) is used for zero-copy between CPU and GPU.
- High-performance scratch buffers are pre-allocated to avoid host-device overhead in the hot path.
