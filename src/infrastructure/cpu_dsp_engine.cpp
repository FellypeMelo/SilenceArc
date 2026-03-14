#include "silence_arc/infrastructure/cpu_dsp_engine.h"
#include <cmath>
#include <iostream>
#include <complex>
#include <algorithm>
#include <vector>

namespace sa::infrastructure::directml_impl {

CpuDspEngine::CpuDspEngine(size_t fft_size, size_t hop_size) 
    : m_fft_size(fft_size), m_hop_size(hop_size) {
    m_freq_size_truncated = fft_size / 2 + 1; // 481
}

CpuDspEngine::~CpuDspEngine() {
    if (m_window_cpu) _aligned_free(m_window_cpu);
    if (m_analysis_buf) _aligned_free(m_analysis_buf);
    if (m_synthesis_buf) _aligned_free(m_synthesis_buf);
    if (m_dft_re) _aligned_free(m_dft_re);
    if (m_dft_im) _aligned_free(m_dft_im);
}

void CpuDspEngine::initialize() {
    std::cout << "[INFO] Initializing CPU DSP Engine (Optimized DFT)..." << std::endl;
    
    m_window_cpu = (float*)_aligned_malloc(m_fft_size * sizeof(float), 64);
    m_analysis_buf = (float*)_aligned_malloc(m_fft_size * sizeof(float), 64);
    m_synthesis_buf = (float*)_aligned_malloc(m_fft_size * sizeof(float), 64);
    
    // Allocate basis: Size is [481 * 960]
    m_dft_re = (float*)_aligned_malloc(m_freq_size_truncated * m_fft_size * sizeof(float), 64);
    m_dft_im = (float*)_aligned_malloc(m_freq_size_truncated * m_fft_size * sizeof(float), 64);

    std::fill(m_analysis_buf, m_analysis_buf + m_fft_size, 0.0f);
    std::fill(m_synthesis_buf, m_synthesis_buf + m_fft_size, 0.0f);

    const float pi = std::acos(-1.0f);
    // Vorbis window
    for (size_t i = 0; i < m_fft_size; ++i) {
        float sin_val = std::sin(0.5f * pi * (i + 0.5f) / (float)(m_fft_size / 2));
        m_window_cpu[i] = std::sin(0.5f * pi * sin_val * sin_val);
    }

    // Pre-compute DFT Basis for 481 bins
    for (size_t k = 0; k < m_freq_size_truncated; ++k) {
        for (size_t n = 0; n < m_fft_size; ++n) {
            float angle = -2.0f * pi * k * n / (float)m_fft_size;
            m_dft_re[k * m_fft_size + n] = std::cos(angle);
            m_dft_im[k * m_fft_size + n] = std::sin(angle);
        }
    }
}

void CpuDspEngine::analyze(const float* input_hop, std::complex<float>* freq_out) {
    std::move(m_analysis_buf + m_hop_size, m_analysis_buf + m_fft_size, m_analysis_buf);
    std::copy(input_hop, input_hop + m_hop_size, m_analysis_buf + (m_fft_size - m_hop_size));

    // Optimized DFT using basis
    for (size_t k = 0; k < m_freq_size_truncated; ++k) {
        float re = 0.0f;
        float im = 0.0f;
        const float* b_re = &m_dft_re[k * m_fft_size];
        const float* b_im = &m_dft_im[k * m_fft_size];
        
        for (size_t n = 0; n < m_fft_size; ++n) {
            float sample = m_analysis_buf[n] * m_window_cpu[n];
            re += sample * b_re[n];
            im += sample * b_im[n];
        }
        freq_out[k] = std::complex<float>(re, im);
    }
}

void CpuDspEngine::synthesize(const std::complex<float>* freq_in, float* output_hop) {
    const float pi = std::acos(-1.0f);
    std::vector<float> reconstructed(m_fft_size, 0.0f);
    const float scale = 1.0f / (float)m_fft_size;

    for (size_t n = 0; n < m_fft_size; ++n) {
        float val = 0.0f;
        for (size_t k = 0; k < m_freq_size_truncated; ++k) {
            const auto& f = freq_in[k];
            float factor = (k == 0 || k == m_freq_size_truncated - 1) ? 1.0f : 2.0f;
            
            // Use basis (conjugate for IDFT)
            float b_re = m_dft_re[k * m_fft_size + n];
            float b_im = -m_dft_im[k * m_fft_size + n]; // Conjugate
            
            val += factor * (f.real() * b_re - f.imag() * b_im);
        }
        reconstructed[n] = val * scale;
    }

    for (size_t i = 0; i < m_hop_size; ++i) {
        float val = (reconstructed[i] * m_window_cpu[i]) + m_synthesis_buf[i];
        output_hop[i] = std::clamp(val, -1.0f, 1.0f);
        m_synthesis_buf[i] = reconstructed[i + m_hop_size] * m_window_cpu[i + m_hop_size];
    }
}

} // namespace sa::infrastructure::directml_impl
