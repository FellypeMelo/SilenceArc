#pragma once

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/dft.hpp>
#include <memory>
#include <complex>
#include <vector>

namespace sa::infrastructure {

/**
 * @brief Internal SYCL coordinator for DSP operations (STFT, ISTFT, ERB).
 * Isolated from high-level application logic.
 */
class SYCLDSPCoordinator {
public:
    SYCLDSPCoordinator(sycl::queue& queue, size_t fft_size, size_t hop_size);
    ~SYCLDSPCoordinator();

    void initialize();
    
    // STFT Analysis
    void analyze(const float* input, const float* window, float* analysis_mem, std::complex<float>* freq_out);
    
    // Feature Extraction
    void extract_power_spectrum(const std::complex<float>* freq, float* power_spectrum);
    void apply_erb_filterbank(const float* power_spectrum, const float* fb_matrix, float* erb_out);
    
    // ISTFT Synthesis
    void apply_erb_mask(const std::complex<float>* freq_in, const float* mask, const float* inv_fb_matrix, std::complex<float>* freq_out);
    void apply_df_coefficients(const std::complex<float>* history, const float* coefs, std::complex<float>* freq_out);
    
    void synthesize(const std::complex<float>* freq, const float* window, float* reconstructed, float* synthesis_mem, float* output);

    void shift_analysis_buffer(float* analysis_mem);

    // USM Buffer Accessors
    float* get_window_buffer() { return m_window_buffer; }
    float* get_analysis_mem() { return m_analysis_mem; }
    float* get_synthesis_mem() { return m_synthesis_mem; }
    std::complex<float>* get_freq_buffer() { return m_freq_buffer; }
    std::complex<float>* get_freq_history() { return m_freq_history; }
    float* get_reconstructed_frame() { return m_reconstructed_frame; }
    float* get_power_spectrum() { return m_power_spectrum; }
    float* get_erb_buffer() { return m_erb_buffer; }
    std::complex<float>* get_filtered_freq_scratch() { return m_filtered_freq_scratch; }
    float* get_erb_fb_matrix() { return m_erb_fb_matrix; }
    float* get_erb_inv_fb_matrix() { return m_erb_inv_fb_matrix; }
    float* get_df_coefs() { return m_df_coefs; }

private:
    sycl::queue& m_queue;
    size_t m_fft_size;
    size_t m_hop_size;
    size_t m_freq_size;
    size_t m_nb_erb = 32;
    size_t m_nb_df = 96;
    size_t m_df_order = 5;

    // USM Buffers
    float* m_window_buffer = nullptr;
    float* m_analysis_mem = nullptr;
    float* m_synthesis_mem = nullptr;
    std::complex<float>* m_freq_buffer = nullptr;
    std::complex<float>* m_freq_history = nullptr;
    float* m_reconstructed_frame = nullptr;
    float* m_power_spectrum = nullptr;
    float* m_erb_buffer = nullptr;
    std::complex<float>* m_filtered_freq_scratch = nullptr;
    float* m_erb_fb_matrix = nullptr;
    float* m_erb_inv_fb_matrix = nullptr;
    float* m_df_coefs = nullptr;
    float* m_fft_input_scratch = nullptr;

    std::unique_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> m_fft_config;
    std::unique_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> m_ifft_config;
};

} // namespace sa::infrastructure
