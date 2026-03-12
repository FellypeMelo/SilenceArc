#pragma once

#include "silence_arc/infrastructure/sycl/sycl_memory_manager.h"
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/dft.hpp>
#include <complex>
#include <memory>

namespace sa::infrastructure::sycl_impl {

/**
 * @brief Optimized DSP Engine using oneMKL for STFT/ISTFT.
 * Handles the 481-to-480 bin mapping and proper scaling for DeepFilterNet3.
 */
class SyclDspEngine {
public:
    SyclDspEngine(sycl::queue& queue, SyclMemoryManager& mem_manager, size_t fft_size = 960, size_t hop_size = 480);
    ~SyclDspEngine();

    void initialize();

    /**
     * @brief Performs STFT on input frame.
     * @param input_hop Input samples (size = hop_size).
     * @param freq_out Output frequency bins (size = 480, truncated).
     */
    void analyze(const float* input_hop, std::complex<float>* freq_out_480);

    /**
     * @brief Performs ISTFT and Overlap-Add.
     * @param freq_in_480 Input frequency bins (size = 480).
     * @param output_hop Output samples (size = hop_size).
     */
    void synthesize(const std::complex<float>* freq_in_480, float* output_hop);

    // Getters for internal buffers (for testing/feature extraction)
    float* get_analysis_mem() { return m_analysis_mem; }
    std::complex<float>* get_freq_full() { return m_freq_full; }

private:
    sycl::queue& m_queue;
    SyclMemoryManager& m_mem_manager;

    size_t m_fft_size;
    size_t m_hop_size;
    size_t m_freq_size_full; // 481
    size_t m_freq_size_truncated; // 480

    // oneMKL DFT descriptors
    std::unique_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> m_fft_desc;
    std::unique_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> m_ifft_desc;

    // Device Buffers
    float* m_window;
    float* m_analysis_mem;
    float* m_synthesis_mem;
    float* m_fft_input_scratch;
    float* m_ifft_output_scratch;
    std::complex<float>* m_freq_full; // 481 bins
};

} // namespace sa::infrastructure::sycl_impl
