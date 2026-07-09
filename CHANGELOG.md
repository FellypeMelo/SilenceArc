# Changelog — SilenceArc

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project has no released
versions yet, so changes live under Unreleased.

## [Unreleased]

### Audit fix sweep (2026-07, branch `fix/audit-sweep-2026-07`)

#### Added
- GPU `SetAttenuationLimit` now works: the SYCL suppressor mixes the dry (noisy)
  signal back in after inference using libDF's `10^(-|db|/20)` mapping, giving the
  "Suppression Strength" slider real effect on the GPU path (was a no-op).
- Real signal metering: the UI input/output meters and dB-reduction readout now
  show actual RMS-derived values instead of a hardcoded `0.5 / 0.5 / 10 dB`.
- `FramesDropped()` from the async pipeline is surfaced in the Telemetry panel.
- Verbose logging gate: `SILENCEARC_VERBOSE` env var re-enables the per-layer NN
  build dumps and engine progress; stdout is clean by default.
- Tests: `test_audio_metrics` (RMS / dB-reduction), `test_attenuation_limit`
  (dry/wet mix math + a GPU bypass-identity integration case), and a pipeline test
  asserting `FramesDropped()` reaches the UI state.
- CI: GitHub Actions workflow (`.github/workflows/ci.yml`) targeting a self-hosted
  Windows + oneAPI + Arc runner to run build + ctest.

#### Changed
- Build: `-Wall -Wextra` enabled for icx across the infra lib and all executables;
  every resulting warning in our code fixed; third-party include dirs marked
  SYSTEM and vendored imgui warnings silenced so the build is warning-clean.
- Signal-level math routed through the unit-tested `domain::AudioMetrics` helper.
- Attenuation-limit math extracted to `domain::attenuation_limit` so the GPU
  suppressor and the tests share one implementation.

#### Removed
- Dead code: `MiniaudioPipeline::Impl::HexToDeviceId`, the unused
  `df_p_flat` / `df_p_view_md` descriptors in the DF decoder, and the empty
  `test_conv2d/batchnorm/gru/linear_mapping()` stubs (+ their call site).

#### Fixed
- `.gitignore` no longer blanket-ignores `*.txt` (it was silently ignoring
  `CMakeLists.txt`); replaced with specific generated-output patterns.
- `main.cpp` metering used `std::min`, which the `<windows.h>` `min()` macro broke;
  replaced with the `AudioMetrics` helpers plus an explicit clamp.
