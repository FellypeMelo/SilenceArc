#pragma once

#include "silence_arc/domain/gpu_accelerator.h"
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <dnnl.hpp>
#include <optional>
#include <vector>
#include <complex>

namespace sa::infrastructure {

/**
 * @brief SYCL implementation of GPUAccelerator optimized for Intel Arc (oneAPI).
 * Uses Unified Shared Memory (USM) for zero-copy performance.
 */
class SYCLAccelerator : public domain::GPUAccelerator {
public:
    SYCLAccelerator();
    ~SYCLAccelerator() override;

    bool initialize() override;
    std::string get_device_name() const override;
    void process_frame(const float* input, float* output, size_t size) override;

    // Getters for internal SYCL/oneDNN objects (needed by inference engine)
    sycl::queue& get_queue() { return *m_queue; }
    dnnl::engine& get_dnnl_engine() { return *m_dnnl_engine; }
    dnnl::stream& get_dnnl_stream() { return *m_dnnl_stream; }

private:
    std::optional<sycl::queue> m_queue;
    std::string m_device_name;

    // oneMKL DFT descriptors
    std::unique_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> m_fft_config;
    std::unique_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> m_ifft_config;

    // USM Buffers for intermediate processing
    float* m_window_buffer = nullptr;
    float* m_analysis_mem = nullptr;
    float* m_synthesis_mem = nullptr;
    std::complex<float>* m_freq_buffer = nullptr;
    std::complex<float>* m_freq_history = nullptr; // [df_order * freq_size]
    float* m_reconstructed_frame = nullptr;
    
    // Feature Extraction Buffers
    float* m_power_spectrum = nullptr;    // [freq_size]
    float* m_erb_buffer = nullptr;        // [nb_erb]
    float* m_erb_fb_matrix = nullptr;     // [freq_size * nb_erb]
    float* m_erb_norm_state = nullptr;    // [nb_erb]
    float* m_spec_norm_state = nullptr;   // [nb_df]

    // oneDNN Engine and Stream
    std::unique_ptr<dnnl::engine> m_dnnl_engine;
    std::unique_ptr<dnnl::stream> m_dnnl_stream;

    // Scratch buffers (reused to avoid allocation in hot path)
    float* m_fft_input_scratch = nullptr;
    std::complex<float>* m_filtered_freq_scratch = nullptr;

    // Constants for DeepFilterNet
    const size_t m_fft_size = 960;
    const size_t m_hop_size = 480;
    const size_t m_freq_size = m_fft_size / 2 + 1;
    const size_t m_df_order = 5;
    const size_t m_nb_erb = 32;
    const size_t m_nb_df = 960 / 2 + 1;

    void setup_kernels();
    void cleanup();
};

} // namespace sa::infrastructure

extern "C" {
    bool sycl_init();
    void sycl_process(const float* input, float* output, size_t size);
    void sycl_get_device_name(char* buffer, size_t max_size);
}
