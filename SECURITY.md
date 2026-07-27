# Security Policy

## Scope

SilenceArc is a native C++20 desktop application that:

- captures and processes raw audio buffers from WASAPI/ASIO devices via `miniaudio`,
- loads DeepFilterNet3 model weights (`models/df3_weights/*.bin`) into SYCL/oneDNN structures,
- links against a prebuilt Rust shared library (`df.dll`, built from the vendored `DeepFilterNet/` workspace) through a C ABI,
- runs GPU kernels directly against Intel Arc hardware via SYCL and Level Zero.

Any of these are plausible attack surfaces for memory-safety issues: a malformed audio device response, a corrupted or adversarial model-weight file, a mismatched C ABI call across the `df.dll` boundary, or an out-of-bounds SYCL kernel access. Reports touching any of these areas are in scope.

The vendored `DeepFilterNet/` workspace (Rikorose/DeepFilterNet) is a separate upstream project with its own codebase, license, and maintainers. If you find a vulnerability in the upstream code itself (not in how SilenceArc calls into it), please also consider reporting it upstream at [github.com/Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet).

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Use GitHub's private vulnerability reporting for this repository: go to the [Security tab](https://github.com/FellypeMelo/SilenceArc/security/advisories/new) and open a new draft security advisory. This reaches the maintainer privately and lets a fix land before the details are public.

If private vulnerability reporting is not enabled on the repository at the time you look (this is a repository setting the maintainer controls, not something that can be guaranteed from outside), open a regular issue asking that reporting be enabled, without describing the vulnerability itself, and wait for a response before disclosing details.

## What to include

- The affected component (audio pipeline, SYCL engine, oneDNN inference, `df.dll` C ABI, UI) and, if applicable, the input that triggers it (a specific WAV file, device configuration, or model file).
- Whether reproduction requires an Intel Arc GPU or reproduces on the CPU fallback path too.
- A rough severity assessment (crash / memory corruption / information disclosure / other) if you're able to make one.

## Response

This is a single-maintainer, alpha-stage open source project with no service-level agreement. There is no guaranteed response time. Reports will be read and triaged as time allows.
