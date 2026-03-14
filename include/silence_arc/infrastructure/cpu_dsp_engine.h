#pragma once

#include <complex>
#include <memory>
#include <vector>

namespace sa::infrastructure::directml_impl {

/**
 * @brief Stable CPU-based DSP Engine for STFT/ISTFT.
 * Optimized for low-latency processing on Intel CPUs.
 */
class CpuDspEngine {
public:
    CpuDspEngine(size_t fft_size = 960, size_t hop_size = 480);
    ~CpuDspEngine();

    void initialize();
    void analyze(const float* input_hop, std::complex<float>* freq_out_480);
    void synthesize(const std::complex<float>* freq_in_480, float* output_hop);

private:
    size_t m_fft_size;
    size_t m_hop_size;
    size_t m_freq_size_truncated; // 480

    // CPU Stability Buffers
    float* m_window_cpu = nullptr;
    float* m_analysis_buf = nullptr;
    float* m_synthesis_buf = nullptr;
    
    // DFT Basis (Optimization)
    float* m_dft_re = nullptr;
    float* m_dft_im = nullptr;
};

} // namespace sa::infrastructure::directml_impl
