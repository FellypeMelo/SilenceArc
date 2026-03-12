# Plan: Revert DeepFiltering to Rust Adapter

## Objective
Remove the failed native SYCL/oneDNN DeepFiltering implementation and restore the stable "old way" using the Rust-based `DeepFilterAdapter`.

## Key Files & Context
- **`src/main.cpp`**: Currently chooses between `SyclNoiseSuppressor` and `DeepFilterAdapter`. Needs to be forced to `DeepFilterAdapter`.
- **`CMakeLists.txt`**: Contains build instructions for the native SYCL implementation and its tests.
- **`src/infrastructure/`**: Home to the files being removed.

## Implementation Steps

### 1. Update `src/main.cpp`
- Remove `#include "silence_arc/infrastructure/sycl_noise_suppressor.h"`.
- Remove the `if (sycl_available)` logic for suppressor creation.
- Always instantiate `silence_arc::infrastructure::DeepFilterAdapter`.
- Ensure model path points to the correct `.tar.gz` or `.onnx` model expected by the Rust library.

### 2. Clean up `CMakeLists.txt`
- Remove the following source files from `silence_arc_infra`:
    - `src/infrastructure/sycl_noise_suppressor.cpp`
    - `src/infrastructure/sycl_accelerator.cpp`
    - `src/infrastructure/sycl_dsp_coordinator.cpp`
    - `src/infrastructure/onednn_inference_engine.cpp`
- Remove the corresponding test executables and `add_test` entries:
    - `test_sycl_inference`
    - `test_df_integration`
    - `test_kernel_correctness`
    - `test_nn_layers`
    - `test_gpu_bridge`

### 3. Update Status and Tracks
- Update `deepfilter3_status.md` to reflect the reversion.
- Add an entry to `conductor/tracks.md`.

## Verification & Testing
- **Build**: Ensure the project builds successfully with `build_project.bat`.
- **Unit Test**: Run `test_noise_suppression.exe` (from `build/` directory to ensure `df.dll` is found).
- **Runtime**: Run `silence_arc.exe` and verify noise suppression can be enabled/disabled without "robotic" artifacts.
- **Telemetry**: Verify GPU load and VRAM usage still display (as they use Level Zero directly via `SyclTelemetryProvider`).
