#include "silence_arc/infrastructure/sycl/sycl_dsp_engine.h"
#include <cmath>
#include <iostream>
#include <complex>

namespace sa::infrastructure::sycl_impl {

// Helper kernel for bit-reversal and initial copy
static void fft_init_kernel(float* input, std::complex<float>* output, size_t n, size_t fft_size) {
    // Basic bit reversal could be added here, but for 960 we'll use a simpler approach or just pad to 1024
}

SyclDspEngine::SyclDspEngine(sycl::queue& queue, SyclMemoryManager& mem_manager, size_t fft_size, size_t hop_size)
    : m_queue(queue), m_mem_manager(mem_manager), m_fft_size(fft_size), m_hop_size(hop_size) {
    m_freq_size_full = m_fft_size / 2 + 1; // 481
    m_freq_size_truncated = m_hop_size;    // 480
}

SyclDspEngine::~SyclDspEngine() {}

void SyclDspEngine::initialize() {
    std::cout << "[DEBUG] Initializing SyclDspEngine (Manual SYCL FFT)..." << std::endl;
    
    // Allocate Buffers
    m_window = m_mem_manager.allocate_device<float>(m_fft_size);
    m_analysis_mem = m_mem_manager.allocate_device<float>(m_fft_size);
    m_synthesis_mem = m_mem_manager.allocate_device<float>(m_fft_size - m_hop_size);
    m_fft_input_scratch = m_mem_manager.allocate_device<float>(m_fft_size);
    m_ifft_output_scratch = m_mem_manager.allocate_device<float>(m_fft_size);
    m_freq_full = m_mem_manager.allocate_device<std::complex<float>>(m_freq_size_full);

    // Initialize
    m_queue.fill(m_analysis_mem, 0.0f, m_fft_size);
    m_queue.fill(m_synthesis_mem, 0.0f, m_fft_size - m_hop_size);
    m_queue.wait();

    // Generate Window
    const double pi = 3.14159265358979323846;
    std::vector<float> host_window(m_fft_size);
    for (size_t i = 0; i < m_fft_size; ++i) {
        double inner_sin = std::sin(0.5 * pi * (static_cast<double>(i) + 0.5) / (m_fft_size / 2.0));
        host_window[i] = static_cast<float>(std::sin(0.5 * pi * inner_sin * inner_sin));
    }
    m_queue.memcpy(m_window, host_window.data(), m_fft_size * sizeof(float)).wait();
    
    std::cout << "[DEBUG] Infrastructure ready." << std::endl;
}

// Since native MKL is failing on Windows, we'll use a very stable DFT-based approach 
// for the 960 size or a simple Radix-2 if we pad to 1024.
// For now, let's use a Direct DFT kernel (O(N^2)) just to PROVE stability, 
// then optimize to FFT (O(N log N)).
void SyclDspEngine::analyze(const float* input_hop, std::complex<float>* freq_out_480) {
    if (!input_hop || !freq_out_480 || !m_analysis_mem) return;

    const size_t overlap_size = m_fft_size - m_hop_size;
    const size_t fft_size = m_fft_size;
    const size_t freq_size_truncated = m_freq_size_truncated;
    const size_t hop_size = m_hop_size;

    // 1. Shift and Load Input
    float* am_shift = m_analysis_mem;
    m_queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(overlap_size), [=](sycl::id<1> idx) {
            am_shift[idx] = am_shift[idx + hop_size];
        });
    });
    m_queue.memcpy(m_analysis_mem + overlap_size, input_hop, m_hop_size * sizeof(float)).wait();

    // 2. Windowing
    float* am_win = m_analysis_mem;
    float* win_ptr = m_window;
    float* scratch_win = m_fft_input_scratch;
    m_queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(fft_size), [=](sycl::id<1> idx) {
            scratch_win[idx] = am_win[idx] * win_ptr[idx];
        });
    }).wait();

    // 3. Stable DFT (Proof of Concept for Windows Stability)
    std::complex<float>* ff = m_freq_full;
    m_queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(m_freq_size_full), [=](sycl::id<1> k_idx) {
            size_t k = k_idx[0];
            std::complex<float> sum(0.0f, 0.0f);
            for (size_t n = 0; n < fft_size; ++n) {
                float angle = -2.0f * 3.1415926535f * k * n / fft_size;
                sum += std::complex<float>(scratch_win[n] * std::cos(angle), scratch_win[n] * std::sin(angle));
            }
            ff[k] = sum;
        });
    }).wait();

    // 4. Truncate 481 -> 480
    m_queue.memcpy(freq_out_480, m_freq_full, freq_size_truncated * sizeof(std::complex<float>)).wait();
}

void SyclDspEngine::synthesize(const std::complex<float>* freq_in_480, float* output_hop) {
    if (!freq_in_480 || !output_hop) return;
    const size_t freq_size_truncated = m_freq_size_truncated;
    const size_t fft_size = m_fft_size;
    const size_t hop_size = m_hop_size;

    // 1. Pad 480 -> 481
    m_queue.memcpy(m_freq_full, freq_in_480, freq_size_truncated * sizeof(std::complex<float>));
    m_queue.fill(m_freq_full + freq_size_truncated, std::complex<float>(0.0f, 0.0f), 1).wait();

    // 2. Stable IDFT
    float* rec = m_ifft_output_scratch;
    std::complex<float>* ff = m_freq_full;
    m_queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(fft_size), [=](sycl::id<1> n_idx) {
            size_t n = n_idx[0];
            float sum = 0.0f;
            // Only need to sum up to N/2 due to conjugate symmetry of real input
            for (size_t k = 0; k < fft_size / 2 + 1; ++k) {
                float angle = 2.0f * 3.1415926535f * k * n / fft_size;
                float phase_real = std::cos(angle);
                float phase_imag = std::sin(angle);
                
                // Real part of (Complex * Complex_Phase)
                float val = (ff[k].real() * phase_real - ff[k].imag() * phase_imag);
                
                // Weighting for symmetry (DC and Nyquist are unique, others appear twice)
                if (k == 0 || k == fft_size / 2) {
                    sum += val;
                } else {
                    sum += 2.0f * val;
                }
            }
            rec[n] = sum / static_cast<float>(fft_size);
        });
    }).wait();

    // 3. Synthesis OLA
    float* reconstructed = m_ifft_output_scratch;
    float* win_synth = m_window;
    float* sm = m_synthesis_mem;
    m_queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(hop_size), [=](sycl::id<1> idx) {
            float val = (reconstructed[idx] * win_synth[idx]) + sm[idx];
            output_hop[idx] = val;
            sm[idx] = reconstructed[idx + hop_size] * win_synth[idx + hop_size];
        });
    }).wait();
}

} // namespace sa::infrastructure::sycl_impl
