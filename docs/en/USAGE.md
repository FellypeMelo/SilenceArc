# SilenceArc: User Guide

SilenceArc provides a minimalist, high-performance interface for real-time noise suppression.

## Getting Started

1.  **Launch the Application:** Run `run.bat` or execute `build/silence_arc.exe`.
2.  **Verify GPU Acceleration:** Check the console window or the Telemetry section in the GUI to ensure your Intel Arc GPU is active.

## Interface Controls

### 1. Audio Device Selection
-   **Input Device:** Select your microphone from the dropdown menu. SilenceArc supports WASAPI and ASIO devices.
-   **Output Device:** Select your speakers or monitoring headphones.
-   **Note:** SilenceArc uses Exclusive Mode where available to ensure the lowest possible latency.

### 2. Suppression Settings
-   **Enable Suppression:** Toggle the noise suppression engine on or off.
-   **Attenuation Limit (dB):** Adjust how aggressively the noise is removed. 
    -   **20dB:** More natural sounding, preserves vocal nuances.
    -   **100dB:** Maximum silence, ideal for very noisy environments.

### 3. Signal Monitoring
-   **Input Level:** Real-time visual readout of your raw microphone signal.
-   **Output Level:** The signal after noise suppression has been applied.
-   **Noise Floor:** Estimates the current level of background noise being suppressed.

### 4. Hardware Telemetry
SilenceArc provides real-time insights into your hardware performance:
-   **GPU Utilization:** How much of your Intel Arc GPU's compute power is being used by the inference engine.
-   **Processing Latency:** The round-trip time (in milliseconds) for a frame to be processed on the GPU.
-   **Memory Footprint:** The VRAM usage of the oneDNN primitives and STFT buffers.

## System Tray Integration
SilenceArc can be minimized to the system tray, where it will continue to process your audio in the background with minimal CPU impact.
