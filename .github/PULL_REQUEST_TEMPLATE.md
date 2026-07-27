## Summary

<!-- What does this PR change, and why? -->

## Testing

<!--
There is no CI for this project (see CONTRIBUTING.md / README.md#testing). This section is
the primary review signal a maintainer has, so be specific.
-->

- [ ] I ran `ctest --output-on-failure` in `build/` after this change.
- **Hardware used:** <!-- e.g. "Intel Arc B580" or "CPU-only, GPU-path tests not run" -->
- **Tests affected / relevant:** <!-- e.g. test_kernel_correctness, test_backend_parity -->
- **Result:** <!-- pass/fail, and any test skipped with a reason -->

## Checklist

- [ ] Changes to `include/silence_arc/domain/` (if any) introduce no SYCL/oneDNN/WASAPI dependency.
- [ ] Relevant docs updated (`docs/en/USAGE.md`, `docs/en/ARCHITECTURE.md`, `docs/en/ENGINE.md`, or a new ADR under `docs/en/adr/`, plus the matching `docs/pt-BR/` mirror), if this PR changes behavior they document.
- [ ] No new binaries committed (`df.dll`/`df.dll.lib` at the repo root are a known, already-flagged exception — see README.md#known-limitations).
- [ ] Commit messages follow the existing `type(scope): summary` convention.

## Related issues

<!-- Closes #... / Relates to #... -->
