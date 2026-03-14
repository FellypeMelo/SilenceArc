# SilenceArc Setup Guide (DirectML)

Follow these steps to set up the SilenceArc development environment on Windows 11.

## 🛠️ Prerequisites

-   **GPU:** Intel Arc B-Series or any DX12 compatible GPU.
-   **OS:** Windows 10/11 64-bit.
-   **Compiler:** Intel oneAPI ICX or MSVC 2022.
-   **CMake:** Version 3.20 or higher.

## 📥 Environment Setup

1.  **Download ONNX Runtime:**
    The project uses ONNX Runtime with DirectML. Run the provided script to download the necessary binaries:
    ```powershell
    .\download_onnxruntime.ps1
    ```

2.  **Initialize Environment:**
    Run the setup script to configure path variables and verify dependencies:
    ```cmd
    setup_environment.bat
    ```

## 🏗️ Building the Project

Run the build script to compile the application and tests:
```cmd
build_project.bat
```

## ✅ Verification

To verify that DirectML is working correctly on your GPU, run the neural path test:
```cmd
build\test_neural_path.exe
```
You should see: `[SUCCESS] DirectML Engine active on GPU.`

## 🚀 Running the App

Execute the final binary:
```cmd
build\silence_arc.exe
```
You can then select your input/output devices in the UI and toggle noise suppression.
