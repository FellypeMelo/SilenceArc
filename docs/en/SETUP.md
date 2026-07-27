# SilenceArc: Setup & Build Guide

Follow these steps to configure your environment and build SilenceArc for Intel Arc GPUs.

## Prerequisites

### 1. Intel oneAPI Base Toolkit
You must have the **Intel oneAPI Base Toolkit** installed (version 2024.0 or newer).
-   **Compiler:** `icx` (Intel LLVM C++ Compiler) is required for `-fsycl` support.
-   **Libraries:** oneDNN and oneMKL must be included in your installation.

### 2. CMake
Version 3.20 or newer is required.

### 3. Rust (For DeepFilterNet Core)
If you plan to modify the model logic, you will need the Rust toolchain installed. The project uses a pre-compiled `df.dll` for standard builds.

## Environment Configuration

Before building or running the application, you must initialize the oneAPI environment variables. This project provides a helper script:

```powershell
# In a PowerShell or CMD window:
.\setup_intel.bat
```

This script invokes Intel's `setvars.bat` and configures the environment for the `icx` compiler and required libraries.

## Building the Project

SilenceArc uses CMake for project management. We recommend using the **Ninja** generator for faster builds.

```bash
# 1. Create a build directory
mkdir build
cd build

# 2. Configure with Intel LLVM Compiler
cmake -G "Ninja" -DCMAKE_CXX_COMPILER=icx -DCMAKE_C_COMPILER=icx ..

# 3. Build the application
cmake --build . --config Release
```

## Runtime Dependencies
Ensure that `df.dll` (from the DeepFilterNet target directory) is located in the same folder as `silence_arc.exe` or available in your system path.

## Verification
Run the following command to verify your GPU is correctly detected and the kernels are functional:
```bash
.\build\test_nn_layers.exe
```
You should see a message: `[INFO] SYCL Initialized on: Intel(R) Arc(TM) ...`
