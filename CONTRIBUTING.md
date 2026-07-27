# Contributing to SilenceArc

SilenceArc is a native C++20/SYCL project with a hard hardware dependency: most of the code you'd touch only proves itself correct on a physical Intel Arc GPU. This document sets expectations up front so you don't build for an hour and then discover you can't run the part of the test suite that matters.

## Before you start

Read [docs/en/ARCHITECTURE.md](./docs/en/ARCHITECTURE.md) for the Clean Architecture layering and [docs/en/ENGINE.md](./docs/en/ENGINE.md) for the SYCL/oneDNN internals. Both ADRs in [docs/en/adr/](./docs/en/adr/) explain *why* the project is shaped the way it is (model choice, the two-tier backend abstraction) — read them before proposing to restructure either. (Português: [docs/pt-BR/](./docs/pt-BR/), mesma estrutura.)

## Prerequisites

- **Hardware:** an Intel Arc GPU (Alchemist/A-series or Battlemage/B-series) if you intend to build or test the GPU backend. CPU-only contributions (UI, the `DeepFilterAdapter` CPU path, non-SYCL tests) don't require one, but most of the test suite does.
- **Intel oneAPI Base Toolkit** 2024.0 or newer, for the `icx` compiler, oneDNN, and oneMKL.
- **CMake** 3.20+ and **Ninja**.
- **Level Zero SDK** — `CMakeLists.txt` auto-detects it under `C:/Program Files/LevelZeroSDK/*` and fails the configure step outright if it isn't found. Set `SILENCE_ARC_LEVEL_ZERO_ROOT` if yours lives elsewhere.
- **Rust**, only if you're modifying the vendored `DeepFilterNet/` workspace and need to rebuild `df.dll`.

See [docs/en/SETUP.md](./docs/en/SETUP.md) for the full walkthrough and [README.md](./README.md#quickstart) for the condensed version.

## Building and testing

```bash
.\setup_intel.bat
mkdir build && cd build
cmake -G "Ninja" -DCMAKE_CXX_COMPILER=icx -DCMAKE_C_COMPILER=icx ..
cmake --build . --config Release
ctest --output-on-failure
```

`CMakeLists.txt` registers 14 tests via `add_test()`. Seven run on GoogleTest; the other seven are small custom executables with no test framework, three of which (`SYCLDiscoveryTest`, `GPUBridgeTest`, `KernelCorrectnessTest`) share a header-only assertion harness, `tests/sycl_test_harness.h`. If you don't have an Arc GPU attached, expect the SYCL-dependent tests (`SYCLDiscoveryTest`, `GPUBridgeTest`, `KernelCorrectnessTest`, `NNLayersTest`, `PipelineLatencyBench`, `BackendParityTest`) to fail or be meaningless — this is a hardware limitation of the project, not something you need to work around in your PR.

There is no CI for this project. Nothing runs automatically on push or on your pull request; running the relevant tests locally and describing what you ran (and on what hardware) in your PR description is the review signal maintainers have to go on.

## Making changes

- Keep the domain layer (`include/silence_arc/domain/`) free of SYCL, oneDNN, or WASAPI headers — that separation is the point of the architecture (see ADR-002).
- If you touch the SYCL/oneDNN engine, run `test_kernel_correctness` (checks kernel output against a numerical baseline, MSE < 1e-13) and `test_backend_parity` (GPU vs. CPU/`tract` output on real speech samples) before opening a PR.
- If you touch the async audio path, run the E2E tests (`test_e2e_loopback`, `test_e2e_samples`) — they exercise the real capture→process→playback path, not mocks.
- Match the existing commit style (`type(scope): summary`, e.g. `fix(onednn): ...`, `feat(infra): ...`) — see `git log` for real examples.
- Do not commit new binaries. `df.dll`/`df.dll.lib` at the repository root are a known, already-flagged exception (see [README.md](./README.md#known-limitations)); adding more prebuilt artifacts makes the problem worse, not better.

## Pull requests

- Describe what you tested and on what hardware (Arc GPU model, or "CPU-only, GPU tests not run").
- Keep unrelated changes out of the PR — this makes hardware-gated review tractable for a maintainer without your exact GPU.
- Update the relevant doc under `docs/en/` (and its `docs/pt-BR/` mirror, same filename) if you change behavior it documents (`docs/en/USAGE.md` for user-facing behavior, `docs/en/ARCHITECTURE.md` or `docs/en/ENGINE.md` for structural changes, a new ADR under `docs/en/adr/` for a real architectural decision).

## Reporting bugs and requesting features

Use the issue templates under `.github/ISSUE_TEMPLATE/`. For security-relevant issues (crashes on malformed input, memory-safety problems in the SYCL/oneDNN/miniaudio paths, anything exploitable), see [SECURITY.md](./SECURITY.md) instead of opening a public issue.
