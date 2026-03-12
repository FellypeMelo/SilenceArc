# Specification: Project Cleanup and Modularization

## Overview
This track focuses on improving the code quality, maintainability, and modularity of the entire Silence Arc project. The primary goal is to align the codebase with Clean Code, KISS, and YAGNI principles, specifically targeting high-density code areas and ensuring strict decoupling between domain logic and infrastructure.

## Functional Requirements (Refactor)
- **Infrastructure Decoupling:** Refactor `sycl_accelerator` and `onednn_inference_engine` to ensure that domain logic (noise suppression algorithms, ERB logic) is not tightly coupled to SYCL or oneDNN types where possible.
- **SRP Enforcement:** Identify and break down large functions (especially in the audio pipeline and UI management) into smaller, single-responsibility units.
- **YAGNI Cleanup:** Remove any speculative features, unused variables, or redundant abstractions that do not serve current project goals.
- **Modularization:** Restructure the project to have clear boundaries between `domain`, `infrastructure`, and `application` layers.

## Non-Functional Requirements
- **Performance Parity:** The refactored code must maintain current latency metrics (<4ms for inference).
- **Readability:** Code must adhere to the project's C++ style guide, with a focus on self-documenting naming and low cyclomatic complexity.

## Acceptance Criteria
- **TDD Regression:** All existing integration and unit tests (e.g., `test_df_integration`, `test_audio_pipeline`) must pass with zero behavioral regressions.
- **Architectural Audit:** A manual review confirms that infrastructure dependencies are properly isolated.
- **Clean Build:** The project builds without warnings under strict compiler flags.

## Out of Scope
- Implementation of new features or new AI models.
- Changes to the core DeepFilterNet3 weight values or network topology.
