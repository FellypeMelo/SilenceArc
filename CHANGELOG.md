# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project has no tagged releases yet (`git tag` is empty and no version number is set in `CMakeLists.txt`), so all history below is grouped under a single [Unreleased] section, derived from the project's real commit history rather than invented version numbers.

## [Unreleased]

### Added

- Initial SilenceArc application: Dear ImGui UI, `miniaudio` audio pipeline, DeepFilterNet-based noise suppression, and SYCL acceleration (`e26ce8d`).
- Native C++/SYCL inference engine for DeepFilterNet3 (`9a8c55a`), later reimplemented as a full native SYCL/oneDNN integration (`167e5cb`).
- DeepFilterNet integration via `tract` ONNX inference, a C API, and GPU bridging (`3c14ee6`).
- `DeepFilterNet/` vendored into the repository to track local modifications (`50279a9`).
- `miniaudio`-based audio pipeline and device management components (`b5dd871`), with dynamic audio device selection and VB-Cable output routing (`2091fc3`).
- Asynchronous audio callback / async processing pipeline to keep heavy work off the real-time audio thread (`4bf87cb`), later finalized with UI telemetry feedback (`c7a121d`, `581c6ca`).
- End-to-end test coverage: automated audio loopback (`6f9cd13`) and sample-file-based verification (`c29e173`), plus supporting audio buffer tests (`2f86ddd`).
- SYCL benchmark harness used for model-selection candidate simulations (`b469479`).
- `SyclTelemetryProvider` test coverage (`b476b73`).
- Apache License 2.0 (`aa6dd2d`).

### Changed

- Collapsed hot-path GPU synchronization points, offloaded processing to the async worker thread, and unified the backend architecture around the two-tier `INoiseSuppressor` abstraction (`2129d83`, see [ADR-002](./docs/en/adr/002-two-tier-noise-suppression-abstraction.md)).
- Improved DeepFilterNet integration and async pipeline robustness (`10217b0`).

### Fixed

- Resolved a oneDNN initialization crash caused by an incorrect groups count and unnecessary memory copies during tensor flattening, now using memory views instead (`43a8606`).
- Used stable hex-string device IDs for robust audio output device switching (`0369261`).
- Supported stereo WAV files by downmixing to mono in `WavLoader`, and fixed state bleeding between test samples (`e765614`).

[Unreleased]: https://github.com/FellypeMelo/SilenceArc/commits/master
