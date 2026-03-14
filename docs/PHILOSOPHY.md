# SilenceArc Philosophy: Stability & Performance via DirectML

SilenceArc is built on the principle that real-time audio enhancement requires both **extreme stability** and **hardware-native performance**.

## The Choice of DirectML

After investigating several backends (including SYCL and OpenVINO), we selected **DirectML** as our primary inference engine for the following reasons:

1.  **Platform Stability:** As part of the DirectX ecosystem, DirectML offers superior stability on Windows 11, with dedicated support from Intel for the Arc driver stack.
2.  **Hardware Native:** DirectML leverages the XMX (Matrix Extensions) on Intel Arc GPUs, providing the same performance level as low-level kernels with less maintenance overhead.
3.  **Future-Proof:** Using the ONNX Runtime with DirectML ensures compatibility with future GPU generations without needing to rewrite custom kernels.

## Native C++ Over Python

Real-time audio processing has no room for the nondeterministic latency of a garbage collector or the overhead of Python-to-C++ bridging. SilenceArc is written in **Pure C++** to ensure:
-   **Predictable Latency:** Guaranteed sub-10ms processing windows.
-   **Direct Memory Access:** Zero-copy transfers between audio buffers and neural tensors.
-   **Binary Compactness:** A lightweight application that runs without heavy runtime dependencies.

## No-Compromise Audio Quality

By implementing the full **DeepFilterNet3** architecture natively, we ensure that SilenceArc provides studio-quality noise suppression that outperforms generic, simpler algorithms like RNNoise.
