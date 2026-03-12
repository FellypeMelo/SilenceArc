#include "silence_arc/infrastructure/sycl_dsp_coordinator.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace sa::infrastructure {

SYCLDSPCoordinator::SYCLDSPCoordinator(sycl::queue& queue, size_t fft_size, size_t hop_size)
    : m_queue(queue), m_fft_size(fft_size), m_hop_size(hop_size) {
    m_freq_size = m_fft_size / 2 + 1;
}

SYCLDSPCoordinator::~SYCLDSPCoordinator() {
    m_fft_config.reset();
    m_ifft_config.reset();
    
    if (m_window_buffer) sycl::free(m_window_buffer, m_queue);
    if (m_analysis_mem) sycl::free(m_analysis_mem, m_queue);
    if (m_synthesis_mem) sycl::free(m_synthesis_mem, m_queue);
    if (m_freq_buffer) sycl::free(m_freq_buffer, m_queue);
    if (m_freq_history) sycl::free(m_freq_history, m_queue);
    if (m_reconstructed_frame) sycl::free(m_reconstructed_frame, m_queue);
    if (m_power_spectrum) sycl::free(m_power_spectrum, m_queue);
    if (m_erb_buffer) sycl::free(m_erb_buffer, m_queue);
    if (m_filtered_freq_scratch) sycl::free(m_filtered_freq_scratch, m_queue);
    if (m_erb_fb_matrix) sycl::free(m_erb_fb_matrix, m_queue);
    if (m_erb_inv_fb_matrix) sycl::free(m_erb_inv_fb_matrix, m_queue);
    if (m_df_coefs) sycl::free(m_df_coefs, m_queue);
    if (m_fft_input_scratch) sycl::free(m_fft_input_scratch, m_queue);
}

void SYCLDSPCoordinator::initialize() {
    try {
        m_fft_config = std::make_unique<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(static_cast<std::int64_t>(m_fft_size));
        m_fft_config->set_value(oneapi::mkl::dft::config_param::PLACEMENT, oneapi::mkl::dft::config_value::NOT_INPLACE);
        m_fft_config->set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, oneapi::mkl::dft::config_value::COMPLEX_COMPLEX);
        m_fft_config->set_value(oneapi::mkl::dft::config_param::PACKED_FORMAT, oneapi::mkl::dft::config_value::CCE_FORMAT);
        m_fft_config->set_value(oneapi::mkl::dft::config_param::FORWARD_SCALE, 1.0f);
        m_fft_config->commit(m_queue);

        m_ifft_config = std::make_unique<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(static_cast<std::int64_t>(m_fft_size));
        m_ifft_config->set_value(oneapi::mkl::dft::config_param::PLACEMENT, oneapi::mkl::dft::config_value::NOT_INPLACE);
        m_ifft_config->set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, oneapi::mkl::dft::config_value::COMPLEX_COMPLEX);
        m_ifft_config->set_value(oneapi::mkl::dft::config_param::PACKED_FORMAT, oneapi::mkl::dft::config_value::CCE_FORMAT);
        m_ifft_config->set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, 1.0f);
        m_ifft_config->commit(m_queue);

        // Allocate USM Buffers
        m_window_buffer = sycl::malloc_device<float>(m_fft_size, m_queue);
        m_analysis_mem = sycl::malloc_device<float>(m_fft_size, m_queue);
        m_synthesis_mem = sycl::malloc_device<float>(m_fft_size - m_hop_size, m_queue);
        m_freq_buffer = sycl::malloc_device<std::complex<float>>(m_freq_size, m_queue);
        m_freq_history = sycl::malloc_device<std::complex<float>>(m_df_order * m_freq_size, m_queue);
        m_reconstructed_frame = sycl::malloc_device<float>(m_fft_size, m_queue);
        m_power_spectrum = sycl::malloc_device<float>(m_freq_size, m_queue);
        m_erb_buffer = sycl::malloc_device<float>(m_nb_erb, m_queue);
        m_filtered_freq_scratch = sycl::malloc_device<std::complex<float>>(m_freq_size, m_queue);
        m_erb_fb_matrix = sycl::malloc_device<float>(m_freq_size * m_nb_erb, m_queue);
        m_erb_inv_fb_matrix = sycl::malloc_device<float>(m_nb_erb * m_freq_size, m_queue);
        m_df_coefs = sycl::malloc_device<float>(m_nb_df * m_df_order * 2, m_queue);
        m_fft_input_scratch = sycl::malloc_device<float>(m_fft_size, m_queue);

        m_queue.fill(m_analysis_mem, 0.0f, m_fft_size);
        m_queue.fill(m_synthesis_mem, 0.0f, m_fft_size - m_hop_size);
        m_queue.fill(reinterpret_cast<float*>(m_freq_history), 0.0f, m_df_order * m_freq_size * 2);
        m_queue.fill(m_erb_buffer, 0.0f, m_nb_erb);
        m_queue.fill(m_df_coefs, 0.0f, m_nb_df * m_df_order * 2);
        m_queue.wait();

        const double pi = 3.14159265358979323846;
        std::vector<float> host_window(m_fft_size);
        for (size_t i = 0; i < m_fft_size; ++i) {
            double sin_val = std::sin(0.5 * pi * (static_cast<double>(i) + 0.5) / (m_fft_size / 2.0));
            host_window[i] = static_cast<float>(std::sin(0.5 * pi * sin_val * sin_val));
        }
        m_queue.memcpy(m_window_buffer, host_window.data(), m_fft_size * sizeof(float)).wait();

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] SYCLDSPCoordinator Initialization failed: " << e.what() << std::endl;
        throw;
    }
}

void SYCLDSPCoordinator::analyze(const float* input, const float* window, float* analysis_mem, std::complex<float>* freq_out) {
    const size_t overlap_size = m_fft_size - m_hop_size;
    const size_t fft_size = m_fft_size;
    
    m_queue.memcpy(analysis_mem + overlap_size, input, m_hop_size * sizeof(float)).wait();
    
    // Windowing using dedicated scratch buffer
    float* scratch = m_fft_input_scratch;
    m_queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(fft_size), [=](sycl::id<1> idx) {
            scratch[idx] = analysis_mem[idx] * window[idx];
        });
    }).wait();
    
    oneapi::mkl::dft::compute_forward(*m_fft_config, scratch, freq_out).wait();
}

void SYCLDSPCoordinator::extract_power_spectrum(const std::complex<float>* freq, float* power_spectrum) {
    const size_t freq_size = m_freq_size;
    m_queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(freq_size), [=](sycl::id<1> idx) {
            float re = freq[idx].real();
            float im = freq[idx].imag();
            power_spectrum[idx] = re * re + im * im;
        });
    }).wait();
}

void SYCLDSPCoordinator::apply_erb_filterbank(const float* power_spectrum, const float* fb_matrix, float* erb_out) {
    const size_t nb_erb = m_nb_erb;
    const size_t freq_size = m_freq_size;
    m_queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nb_erb), [=](sycl::id<1> erb_idx) {
            float sum = 0.0f;
            for (size_t f = 0; f < freq_size; ++f) {
                sum += power_spectrum[f] * fb_matrix[f * nb_erb + erb_idx];
            }
            erb_out[erb_idx] = sum;
        });
    }).wait();
}

void SYCLDSPCoordinator::apply_erb_mask(const std::complex<float>* freq_in, const float* mask, const float* inv_fb_matrix, std::complex<float>* freq_out) {
    const size_t nb_erb = m_nb_erb;
    const size_t freq_size = m_freq_size;
    m_queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(freq_size), [=](sycl::id<1> f_idx) {
            float m = 0.0f;
            for (size_t e = 0; e < nb_erb; ++e) {
                m += mask[e] * inv_fb_matrix[e * freq_size + f_idx];
            }
            m = std::clamp(m, 0.0f, 1.0f);
            freq_out[f_idx] = freq_in[f_idx] * m;
        });
    }).wait();
}

void SYCLDSPCoordinator::apply_df_coefficients(const std::complex<float>* history, const float* coefs, std::complex<float>* freq_out) {
    const size_t nb_df = m_nb_df;
    const size_t df_order = m_df_order;
    const size_t freq_size = m_freq_size;
    m_queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nb_df), [=](sycl::id<1> f_idx) {
            std::complex<float> df(0.0f, 0.0f);
            for (size_t i = 0; i < df_order; ++i) {
                std::complex<float> tap_coef(coefs[(f_idx * df_order + i) * 2 + 0], 
                                           coefs[(f_idx * df_order + i) * 2 + 1]);
                df += tap_coef * history[(df_order - 1 - i) * freq_size + f_idx];
            }
            freq_out[f_idx] = df;
        });
    }).wait();
}

void SYCLDSPCoordinator::synthesize(const std::complex<float>* freq, const float* window, float* reconstructed, float* synthesis_mem, float* output) {
    const size_t fft_size = m_fft_size;
    const size_t hop_size = m_hop_size;
    
    oneapi::mkl::dft::compute_backward(*m_ifft_config, const_cast<std::complex<float>*>(freq), reconstructed).wait();
    
    m_queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(hop_size), [=](sycl::id<1> idx) {
            output[idx] = ((reconstructed[idx] * window[idx]) + synthesis_mem[idx]) / static_cast<float>(fft_size);
            synthesis_mem[idx] = reconstructed[idx + hop_size] * window[idx + hop_size];
        });
    }).wait();
}

void SYCLDSPCoordinator::shift_analysis_buffer(float* analysis_mem) {
    const size_t overlap_size = m_fft_size - m_hop_size;
    const size_t hop_size = m_hop_size;
    m_queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(overlap_size), [=](sycl::id<1> idx) {
            analysis_mem[idx] = analysis_mem[idx + hop_size];
        });
    }).wait();
}

} // namespace sa::infrastructure
