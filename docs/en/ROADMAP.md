# Roadmap

SilenceArc's SYCL/oneDNN GPU port was built in three phases. This document tracks what's done, what's verified, and what's still open. It replaces an earlier root-level task tracker (`sycl-integration.md`) with the same phase history, corrected where it had drifted from the actual codebase (see the note under Phase 1, Task 5).

## Phase 1 — SYCL build environment & test harness (complete)

- [x] **Task 1:** Create `tests/sycl_test_harness.h` — a header-only, zero-dependency assertion harness for GPU-device-level tests that don't fit GoogleTest's process model.
- [x] **Task 2:** Migrate `tests/test_sycl_discovery.cpp` onto the new harness. Verified: compiles with `icx -fsycl`.
- [x] **Task 3:** Patch `CMakeLists.txt` to bypass a broken `find_package(IntelSYCL)` path and link SYCL manually. Verified: `cmake` configure step completes.
- [x] **Task 4:** Define the `GPUAccelerator` abstraction and its SYCL implementation. Verified: Clean Architecture separation held at the time (later revised — see the note below).
- [x] **Task 5:** Establish the GPU FFI bridge boundary. Verified: `extern "C"` signatures match the calling code.

  > **Correction from the original tracker.** This task was previously recorded as "Design FFI Bridge in `DeepFilterNet/libDF/src/gpu_bridge.rs`". No such file exists anywhere under `DeepFilterNet/libDF/src/` in this repository, and it never did — that path was wrong from the start. The actual GPU FFI boundary is the `extern "C"` block (`sycl_init`, `sycl_process`, `sycl_get_device_name`, `sycl_set_df_enabled`, `sycl_reset`) in `src/infrastructure/sycl_accelerator.cpp`, exercised by `tests/test_gpu_bridge.cpp` (`GPUBridgeTest` in the CTest suite). The task itself was genuinely completed; only the file path recorded for it was stale.

- [x] **Task 6 (TDD RED):** `test_sycl_discovery` fails when no Arc GPU is present or the environment is misconfigured.
- [x] **Task 7 (TDD GREEN):** Environment and code fixed until `test_sycl_discovery` passes on real hardware.

## Phase 2 — Core kernel porting & oneDNN integration (complete)

- [x] **Task 8:** Integrate oneDNN (DNNL) into the build system.
- [x] **Task 9:** Port STFT (analysis) to SYCL kernels.
- [x] **Task 10:** Port ISTFT (synthesis) to SYCL kernels.
- [x] **Task 11:** Implement Deep Filtering (frequency-domain convolution) in SYCL.
- [x] **Task 12 (TDD RED):** `test_kernel_correctness` written against a synthetic signal, checked against a numerical baseline.
- [x] **Task 13 (TDD GREEN):** Kernel logic fixed until `test_kernel_correctness` passes (MSE < 1e-13 against the baseline).

## Phase 3 — Neural-network porting to GPU (open)

- [ ] **Task 14:** Map the remaining DeepFilterNet3 layers (convolutions, GRU/linear) to oneDNN primitives.
- [ ] **Task 15:** Load weights from the exported tensors (`models/df3_weights/`, 133 files, see `scripts/export_df3_weights.py`) into oneDNN buffers.
- [ ] **Task 16:** Port the encoder inference path to GPU.
- [ ] **Task 17:** Port the ERB and DF decoders to GPU.
- [ ] **Task 18 (TDD RED):** Verify full GPU inference against the Rust/`tract` CPU baseline (`test_backend_parity`).
- [ ] **Task 19 (TDD GREEN):** Optimize data flow and batching once correctness is established.

## Phase 1–2 done-when criteria

- [x] `test_sycl_discovery` runs without GTest conflicts.
- [x] The GPU-backend abstraction is defined and isolated from the domain layer.
- [x] Build is stable using `icx -fsycl`.
- [x] STFT/ISTFT and Deep Filtering kernels are functional on GPU.

## Notes

- Every phase followed a TDD RED→GREEN discipline: a failing test proven first, then the minimal implementation to pass it.
- The Clean Architecture layering referenced by Task 4 above has since been revised: `GPUAccelerator`/`NeuralNetworkModel` no longer live under `domain/` in an `sa::` namespace. [ADR-002](./adr/002-two-tier-noise-suppression-abstraction.md) moved them into `include/silence_arc/infrastructure/` as a Bridge private to `SyclNoiseSuppressor`, collapsing `sa::` into `silence_arc::infrastructure`. Treat [ARCHITECTURE.md](./ARCHITECTURE.md) and ADR-002 as the current source of truth for where these types live, not this roadmap's Phase 1 wording.
- Unified Shared Memory (USM) is used for zero-copy transfer between host and device.
- Hot-path scratch buffers are pre-allocated to avoid host-device allocation overhead in the audio callback path — see [ENGINE.md](./ENGINE.md).
