# SilenceArc Documentation

This directory holds SilenceArc's internal documentation in two mirrored trees: **[`en/`](./en/)** (English, source of truth) and **[`pt-BR/`](./pt-BR/)** (Português do Brasil, a technical translation). Both trees use the same filenames, the same directory layout, and the same section order, so a link that works in one works in the other by swapping `en` for `pt-BR`.

| Document | English | Português (Brasil) |
|---|---|---|
| System architecture & data flow | [en/ARCHITECTURE.md](./en/ARCHITECTURE.md) | [pt-BR/ARCHITECTURE.md](./pt-BR/ARCHITECTURE.md) |
| SYCL/oneDNN inference engine internals | [en/ENGINE.md](./en/ENGINE.md) | [pt-BR/ENGINE.md](./pt-BR/ENGINE.md) |
| Design philosophy / rationale | [en/PHILOSOPHY.md](./en/PHILOSOPHY.md) | [pt-BR/PHILOSOPHY.md](./pt-BR/PHILOSOPHY.md) |
| Build & environment setup | [en/SETUP.md](./en/SETUP.md) | [pt-BR/SETUP.md](./pt-BR/SETUP.md) |
| End-user guide | [en/USAGE.md](./en/USAGE.md) | [pt-BR/USAGE.md](./pt-BR/USAGE.md) |
| Whitepaper | [en/WHITEPAPER.md](./en/WHITEPAPER.md) | [pt-BR/WHITEPAPER.md](./pt-BR/WHITEPAPER.md) |
| Roadmap | [en/ROADMAP.md](./en/ROADMAP.md) | [pt-BR/ROADMAP.md](./pt-BR/ROADMAP.md) |
| ADR 001 — model selection | [en/adr/001-model-selection.md](./en/adr/001-model-selection.md) | [pt-BR/adr/001-model-selection.md](./pt-BR/adr/001-model-selection.md) |
| ADR 002 — two-tier backend abstraction | [en/adr/002-two-tier-noise-suppression-abstraction.md](./en/adr/002-two-tier-noise-suppression-abstraction.md) | [pt-BR/adr/002-two-tier-noise-suppression-abstraction.md](./pt-BR/adr/002-two-tier-noise-suppression-abstraction.md) |

For the top-level project README, see [../README.md](../README.md) (English) or [../README.pt-BR.md](../README.pt-BR.md) (Português).

## Notes

- English is authored first; the `pt-BR/` tree is a faithful technical translation of it, not an independent source. If the two ever disagree, the English document and the source code govern.
- Code blocks, commands, identifiers, file paths, and error strings are identical in both trees — only prose is translated.
- Neither tree currently holds language-neutral binary assets (diagrams are Mermaid text embedded directly in the Markdown); if that changes, such assets belong under `en/` only and get linked from both trees rather than duplicated.
- `conductor/` and `gemini.md` at the repository root are AI-agent operating/process artifacts from how this project was built, not user- or contributor-facing documentation, and are out of scope for this tree.
