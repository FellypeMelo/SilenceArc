#include "silence_arc/infrastructure/sycl_accelerator.h"
#include <dnnl_sycl.hpp>
#include <iostream>
#include <fstream>
#include <mutex>
#include <cmath>

namespace sa::infrastructure {

static std::unique_ptr<SYCLAccelerator> g_accelerator = nullptr;
static std::mutex g_accel_mutex;

SYCLAccelerator::SYCLAccelerator() : m_device_name("Not Initialized") {}

SYCLAccelerator::~SYCLAccelerator() {
    cleanup();
}

void SYCLAccelerator::cleanup() {
    if (m_queue) {
        if (m_window_buffer) sycl::free(m_window_buffer, *m_queue);
        if (m_analysis_mem) sycl::free(m_analysis_mem, *m_queue);
        if (m_synthesis_mem) sycl::free(m_synthesis_mem, *m_queue);
        if (m_freq_buffer) sycl::free(m_freq_buffer, *m_queue);
        if (m_freq_history) sycl::free(m_freq_history, *m_queue);
        if (m_reconstructed_frame) sycl::free(m_reconstructed_frame, *m_queue);
        if (m_fft_input_scratch) sycl::free(m_fft_input_scratch, *m_queue);
        if (m_filtered_freq_scratch) sycl::free(m_filtered_freq_scratch, *m_queue);
        if (m_power_spectrum) sycl::free(m_power_spectrum, *m_queue);
        if (m_erb_buffer) sycl::free(m_erb_buffer, *m_queue);
        if (m_erb_fb_matrix) sycl::free(m_erb_fb_matrix, *m_queue);
        if (m_erb_norm_state) sycl::free(m_erb_norm_state, *m_queue);
        if (m_spec_norm_state) sycl::free(m_spec_norm_state, *m_queue);
        
        m_window_buffer = nullptr;
        m_analysis_mem = nullptr;
        m_synthesis_mem = nullptr;
        m_freq_buffer = nullptr;
        m_freq_history = nullptr;
        m_reconstructed_frame = nullptr;
        m_fft_input_scratch = nullptr;
        m_filtered_freq_scratch = nullptr;
        m_power_spectrum = nullptr;
        m_erb_buffer = nullptr;
        m_erb_fb_matrix = nullptr;
        m_erb_norm_state = nullptr;
        m_spec_norm_state = nullptr;
    }
}

bool SYCLAccelerator::initialize() {
    try {
        sycl::device device;
        bool found = false;

        auto platforms = sycl::platform::get_platforms();
        for (auto& platform : platforms) {
            auto devices = platform.get_devices();
            for (auto& dev : devices) {
                std::string name = dev.get_info<sycl::info::device::name>();
                if (dev.is_gpu() && name.find("Arc") != std::string::npos) {
                    device = dev;
                    found = true;
                    break;
                }
            }
            if (found) break;
        }

        if (!found) {
            std::cerr << "[ERROR] Intel Arc GPU not found. Falling back to default selector." << std::endl;
            device = sycl::device(sycl::default_selector_v);
        }

        m_queue = sycl::queue(device, sycl::property::queue::in_order());
        m_device_name = device.get_info<sycl::info::device::name>();
        
        std::cout << "[INFO] SYCL Initialized on: " << m_device_name << std::endl;
        
        // Initialize oneDNN Engine and Stream using SYCL Interop
        try {
            m_dnnl_engine = std::make_unique<dnnl::engine>(
                dnnl::sycl_interop::make_engine(m_queue->get_device(), m_queue->get_context()));
            m_dnnl_stream = std::make_unique<dnnl::stream>(
                dnnl::sycl_interop::make_stream(*m_dnnl_engine, *m_queue));
            std::cout << "[INFO] oneDNN Engine (GPU) initialized successfully with SYCL Interop." << std::endl;
        } catch (const dnnl::error& e) {
            std::cerr << "[FATAL] oneDNN Engine initialization failed: " << e.message << std::endl;
            return false;
        }

        setup_kernels();

        return true;
    } catch (const oneapi::mkl::exception& e) {
        std::cerr << "[FATAL] oneMKL Exception during initialization: " << e.what() << std::endl;
        return false;
    } catch (const dnnl::error& e) {
        std::cerr << "[FATAL] oneDNN Exception during initialization: " << e.message << std::endl;
        return false;
    } catch (const sycl::exception& e) {
        std::cerr << "[FATAL] SYCL Initialization failed: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Standard Exception during initialization: " << e.what() << std::endl;
        return false;
    }
}

void SYCLAccelerator::setup_kernels() {
    if (!m_queue) return;

    // Allocate USM device memory
    m_window_buffer = sycl::malloc_device<float>(m_fft_size, *m_queue);
    m_analysis_mem = sycl::malloc_device<float>(m_fft_size, *m_queue);
    m_synthesis_mem = sycl::malloc_device<float>(m_fft_size - m_hop_size, *m_queue);
    m_freq_buffer = sycl::malloc_device<std::complex<float>>(m_freq_size, *m_queue);
    m_freq_history = sycl::malloc_device<std::complex<float>>(m_df_order * m_freq_size, *m_queue);
    m_reconstructed_frame = sycl::malloc_device<float>(m_fft_size, *m_queue);
    m_fft_input_scratch = sycl::malloc_device<float>(m_fft_size, *m_queue);
    m_filtered_freq_scratch = sycl::malloc_device<std::complex<float>>(m_freq_size, *m_queue);

    // Feature Extraction Buffers
    m_power_spectrum = sycl::malloc_device<float>(m_freq_size, *m_queue);
    m_erb_buffer = sycl::malloc_device<float>(m_nb_erb, *m_queue);
    m_erb_fb_matrix = sycl::malloc_device<float>(m_freq_size * m_nb_erb, *m_queue);
    m_erb_norm_state = sycl::malloc_device<float>(m_nb_erb, *m_queue);
    m_spec_norm_state = sycl::malloc_device<float>(m_nb_df, *m_queue);

    // Initialize Memories to zero
    m_queue->fill(m_analysis_mem, 0.0f, m_fft_size);
    m_queue->fill(m_synthesis_mem, 0.0f, m_fft_size - m_hop_size);
    m_queue->fill(m_freq_history, std::complex<float>(0.0f, 0.0f), m_df_order * m_freq_size);
    m_queue->fill(m_erb_norm_state, 1e-10f, m_nb_erb); // Small epsilon to avoid div by zero
    m_queue->fill(m_spec_norm_state, 1e-10f, m_nb_df);
    m_queue->wait();

    // Pre-calculate Vorbis Window
    const double pi = 3.14159265358979323846;
    std::vector<float> host_window(m_fft_size);
    for (size_t i = 0; i < m_fft_size; ++i) {
        double sin_val = std::sin(0.5 * pi * (static_cast<double>(i) + 0.5) / (m_fft_size / 2.0));
        host_window[i] = static_cast<float>(std::sin(0.5 * pi * sin_val * sin_val));
    }
    m_queue->memcpy(m_window_buffer, host_window.data(), m_fft_size * sizeof(float)).wait();

    // Load ERB Filterbank Weights
    std::filesystem::path path = std::filesystem::current_path();
    if (path.filename() == "build") {
        path = path.parent_path();
    }
    auto fb_path = path / "models" / "df3_weights" / "erb_fb.bin";

    std::ifstream fb_file(fb_path, std::ios::binary);
    if (fb_file) {
        std::vector<float> fb_weights(m_freq_size * m_nb_erb);
        fb_file.read(reinterpret_cast<char*>(fb_weights.data()), fb_weights.size() * sizeof(float));
        m_queue->memcpy(m_erb_fb_matrix, fb_weights.data(), fb_weights.size() * sizeof(float)).wait();
        std::cout << "[INFO] ERB Filterbank loaded from: " << fb_path.string() << std::endl;
    } else {
        std::cerr << "[WARN] Could not load ERB filterbank (" << fb_path.string() << "). Model might fail." << std::endl;
    }

    // Configure oneMKL DFT
    std::cout << "[DEBUG] Configuring Forward FFT..." << std::endl;
    m_fft_config = std::make_unique<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(static_cast<std::int64_t>(m_fft_size));
    m_fft_config->set_value(oneapi::mkl::dft::config_param::PLACEMENT, oneapi::mkl::dft::config_value::NOT_INPLACE);
    std::cout << "[DEBUG] Committing Forward FFT..." << std::endl;
    m_fft_config->commit(*m_queue);

    std::cout << "[DEBUG] Configuring Backward FFT..." << std::endl;
    m_ifft_config = std::make_unique<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(static_cast<std::int64_t>(m_fft_size));
    m_ifft_config->set_value(oneapi::mkl::dft::config_param::PLACEMENT, oneapi::mkl::dft::config_value::NOT_INPLACE);
    std::cout << "[DEBUG] Committing Backward FFT..." << std::endl;
    m_ifft_config->commit(*m_queue);

    std::cout << "[INFO] GPU Kernels and FFT configured (FFT Size: " << m_fft_size << ", DF Order: " << m_df_order << ")" << std::endl;
}

std::string SYCLAccelerator::get_device_name() const {
    return m_device_name;
}

void SYCLAccelerator::process_frame(const float* input, float* output, size_t size) {
    if (!m_queue || size != m_hop_size) return;

    auto q = *m_queue;
    float* window = m_window_buffer;
    float* analysis = m_analysis_mem;
    float* synthesis = m_synthesis_mem;
    std::complex<float>* freq = m_freq_buffer;
    std::complex<float>* history = m_freq_history;
    float* reconstructed = m_reconstructed_frame;
    float* device_out = m_fft_input_scratch; // reuse this scratch for output
    std::complex<float>* filtered_freq = m_filtered_freq_scratch;
    
    const size_t fft_size = m_fft_size;
    const size_t hop_size = m_hop_size;
    const size_t freq_size = m_freq_size;
    const size_t overlap_size = m_fft_size - m_hop_size;
    const size_t df_order = m_df_order;

    // --- STFT Analysis ---
    
    // 1. Copy host input to device and shift
    q.memcpy(analysis + overlap_size, input, hop_size * sizeof(float)).wait();

    // 2. Apply Windowing (using Reconstruction buffer as temp input for FFT)
    float* fft_in = reconstructed; // reuse
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(fft_size), [=](sycl::id<1> idx) {
            fft_in[idx] = analysis[idx] * window[idx];
        });
    }).wait();

    // 3. Forward FFT
    oneapi::mkl::dft::compute_forward(*m_fft_config, fft_in, freq).wait();
    
    // Scaling
    const float wnorm = 1.0f / (static_cast<float>(fft_size * fft_size) / (2.0f * hop_size));
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(freq_size), [=](sycl::id<1> idx) {
            freq[idx] *= wnorm;
        });
    }).wait();

    // --- Deep Filtering Convolution ---

    // Update History
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((df_order - 1) * freq_size), [=](sycl::id<1> idx) {
            history[idx] = history[idx + freq_size];
        });
    }).wait();
    q.memcpy(history + (df_order - 1) * freq_size, freq, freq_size * sizeof(std::complex<float>)).wait();

    // Convolution Kernel
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(freq_size), [=](sycl::id<1> f_idx) {
            std::complex<float> sum(0.0f, 0.0f);
            for (size_t i = 0; i < df_order; ++i) {
                float coef_re = (i == (df_order - 1)) ? 1.0f : 0.0f;
                std::complex<float> c(coef_re, 0.0f);
                sum += history[i * freq_size + f_idx] * c;
            }
            filtered_freq[f_idx] = sum;
        });
    }).wait();

    // --- ISTFT Synthesis ---

    // 4. Backward FFT
    oneapi::mkl::dft::compute_backward(*m_ifft_config, filtered_freq, reconstructed).wait();

    // 5. Apply Window and Overlap-Add to DEVICE output buffer
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(hop_size), [=](sycl::id<1> idx) {
            device_out[idx] = (reconstructed[idx] * window[idx]) + synthesis[idx];
            synthesis[idx] = reconstructed[idx + hop_size] * window[idx + hop_size];
        });
    }).wait();

    // 6. Copy device output back to host
    q.memcpy(output, device_out, hop_size * sizeof(float)).wait();

    // 7. Final shift for Analysis Overlap
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(overlap_size), [=](sycl::id<1> idx) {
            analysis[idx] = analysis[idx + hop_size];
        });
    }).wait();
}

} // namespace sa::infrastructure

extern "C" {

bool sycl_init() {
    std::lock_guard<std::mutex> lock(sa::infrastructure::g_accel_mutex);
    if (!sa::infrastructure::g_accelerator) {
        sa::infrastructure::g_accelerator = std::make_unique<sa::infrastructure::SYCLAccelerator>();
    }
    return sa::infrastructure::g_accelerator->initialize();
}

void sycl_process(const float* input, float* output, size_t size) {
    std::lock_guard<std::mutex> lock(sa::infrastructure::g_accel_mutex);
    if (sa::infrastructure::g_accelerator) {
        sa::infrastructure::g_accelerator->process_frame(input, output, size);
    }
}

void sycl_get_device_name(char* buffer, size_t max_size) {
    std::lock_guard<std::mutex> lock(sa::infrastructure::g_accel_mutex);
    if (sa::infrastructure::g_accelerator) {
        std::string name = sa::infrastructure::g_accelerator->get_device_name();
        strncpy_s(buffer, max_size, name.c_str(), _TRUNCATE);
    }
}

}
