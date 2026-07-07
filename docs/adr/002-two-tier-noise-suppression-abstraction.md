# ADR 002: Two-Tier Noise Suppression Abstraction

## Status
Accepted

## Context
SilenceArc ships two independent inference stacks:

- a native **SYCL/oneDNN** engine that runs DeepFilterNet3 on Intel Arc GPUs, and
- a **Rust `libDF`** (`tract`/ONNX) CPU path used as a fallback when no SYCL device is present.

Early in the codebase these were entangled with the internal decomposition of the GPU
engine. Two abstractions named `GPUAccelerator` and `NeuralNetworkModel` lived under a
`domain/` directory in a separate `sa::` namespace, distinct from the
`silence_arc::domain`/`silence_arc::infrastructure` namespace used by the rest of the
application. This created three problems:

1. **Two things called "domain" that are not peers.** `GPUAccelerator`/`NeuralNetworkModel`
   are not a backend-selection seam — they are the *internal* DSP/NN split of one backend.
   Placing them in `domain/` implied they were an application-level contract like
   `INoiseSuppressor`, which they are not.
2. **A third namespace root (`sa::`)** existed only for these two types, so a reader had to
   track `sa::` vs `silence_arc::` with no semantic reason for the split.
3. It obscured where the real backend-selection boundary is.

## Decision
Adopt an explicit **two-tier** abstraction and collapse the `sa::` namespace into
`silence_arc::`.

**Tier 1 — the backend-selection seam (domain):**
`silence_arc::domain::INoiseSuppressor` is the *only* abstraction over "which noise
suppression engine runs." Its two implementations are
`silence_arc::infrastructure::SyclNoiseSuppressor` (GPU) and
`silence_arc::infrastructure::DeepFilterAdapter` (CPU/Rust). `main.cpp` selects one at
runtime based on `sycl_init()` success.

**Tier 2 — the GPU engine's internal Bridge (infrastructure-private):**
`GPUAccelerator` (DSP: STFT, features, filtering, ISTFT) and `NeuralNetworkModel`
(DeepFilterNet3 layers) are a **Bridge/SRP split private to `SyclNoiseSuppressor`**. They
are *not* a second backend seam. They now live in
`include/silence_arc/infrastructure/` under `silence_arc::infrastructure`, alongside their
only concrete implementations, `SYCLAccelerator` and `OneDNNInferenceEngine`.

The `sa::` namespace is removed entirely; the internal test harness moves from `sa::test`
to `silence_arc::test`.

## Consequences
- **Clearer layering.** `domain/` now contains only technology-agnostic contracts
  (`INoiseSuppressor`, `IAudioPipeline`, `ITelemetryProvider`) and value types. No SYCL or
  oneDNN header is reachable from the domain layer.
- **One namespace root.** Everything is `silence_arc::{domain,infrastructure,test}`.
- **The Bridge stays substitutable but private.** Separating DSP from NN behind
  `GPUAccelerator`/`NeuralNetworkModel` still lets each half evolve independently (e.g. a
  future non-oneDNN NN runner) without leaking into the domain seam.
- **Pure move/rename**, verified by a full rebuild and the complete CTest suite (14/14)
  staying green, including the backend-parity test that exercises both `INoiseSuppressor`
  implementations against the same fixtures.

## Related
- ADR 001 — Selection of DeepFilterNet3 as the primary model.
