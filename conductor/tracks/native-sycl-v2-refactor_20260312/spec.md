# Specification: Native SYCL Engine v2 & Project Refactor

## Objective
Implement a high-performance, stable native SYCL/oneDNN inference engine for DeepFilterNet3 using a data-driven approach, while refactoring the codebase for better modularity and adherence to Clean Code principles.

## Success Criteria
- [ ] **Performance:** Real-time processing with < 10ms latency on Intel Arc B580.
- [ ] **Stability:** No crashes during initialization or runtime.
- [ ] **Audio Quality:** Output matches the Rust Adapter reference (no artifacts, no volume loss).
- [ ] **Architecture:** Modular design with clear separation between Domain, Infrastructure, and UI.
- [ ] **Data-Driven:** Weights are mapped to layers automatically via metadata/naming conventions.

## Constraints
- **Hardware:** Optimization specifically targeted at Intel Arc GPUs.
- **Libraries:** Mandatory use of oneMKL (FFT) and oneDNN (convolutions/linear).
- **Memory:** Use SYCL Unified Shared Memory (USM) for zero-copy transfers.
- **Alignment:** Handle the 481-to-480 bin frequency misalignment using padding/truncation.

## Key Modules
1. **SyclMemoryManager:** Handles USM allocations and life-cycle.
2. **SyclGraphBuilder:** Dynamically builds the network graph from weight files.
3. **SyclDspEngine:** Optimized STFT/ISTFT using oneMKL.
4. **NativeSyclEngine:** The main implementation of `NeuralEngine` interface.
