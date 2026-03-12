#include "silence_arc/infrastructure/sycl_accelerator.h"
#include "silence_arc/infrastructure/onednn_inference_engine.h"
#include <dnnl_sycl.hpp>
#include <iostream>
#include <fstream>
#include <mutex>
#include <cmath>
#include <filesystem>
#include <algorithm>

namespace sa::infrastructure {

static std::unique_ptr<SYCLAccelerator> g_accelerator = nullptr;
static std::mutex g_accel_mutex;

SYCLAccelerator::SYCLAccelerator() 
    : m_device_name("Not Initialized")
{
    // Initialize tracking states with reference defaults
    m_erb_mean.resize(32);
    float start = -60.0f;
    float end = -90.0f;
    for (int i = 0; i < 32; ++i) {
        m_erb_mean[i] = start + i * (end - start) / 31.0f;
    }
    
    m_erb_var.resize(96);
    start = 0.001f;
    end = 0.0001f;
    for (int i = 0; i < 96; ++i) {
        m_erb_var[i] = start + i * (end - start) / 95.0f;
    }
}

SYCLAccelerator::~SYCLAccelerator() {
    cleanup();
}

void SYCLAccelerator::cleanup() {
    if (m_queue) {
        auto q = *m_queue;
        if (m_window_buffer) sycl::free(m_window_buffer, q);
        if (m_analysis_mem) sycl::free(m_analysis_mem, q);
        if (m_synthesis_mem) sycl::free(m_synthesis_mem, q);
        if (m_freq_buffer) sycl::free(m_freq_buffer, q);
        if (m_freq_history) sycl::free(m_freq_history, q);
        if (m_reconstructed_frame) sycl::free(m_reconstructed_frame, q);
        if (m_fft_input_scratch) sycl::free(m_fft_input_scratch, q);
        if (m_filtered_freq_scratch) sycl::free(m_filtered_freq_scratch, q);
        if (m_power_spectrum) sycl::free(m_power_spectrum, q);
        if (m_erb_buffer) sycl::free(m_erb_buffer, q);
        if (m_erb_fb_matrix) sycl::free(m_erb_fb_matrix, q);
        if (m_erb_inv_fb_matrix) sycl::free(m_erb_inv_fb_matrix, q);
        if (m_erb_norm_state) sycl::free(m_erb_norm_state, q);
        if (m_spec_norm_state) sycl::free(m_spec_norm_state, q);
        if (m_df_coefs) sycl::free(m_df_coefs, q);

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
        m_erb_inv_fb_matrix = nullptr;
        m_erb_norm_state = nullptr;
        m_spec_norm_state = nullptr;
        m_df_coefs = nullptr;
    }
    m_fft_config.reset();
    m_ifft_config.reset();
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

        if (!found) device = sycl::device(sycl::default_selector_v);

        m_queue = sycl::queue(device, sycl::property::queue::in_order());
        m_device_name = device.get_info<sycl::info::device::name>();

        std::cout << "[INFO] SYCL Initialized on: " << m_device_name << std::endl;

        m_dnnl_engine = std::make_unique<dnnl::engine>(dnnl::sycl_interop::make_engine(m_queue->get_device(), m_queue->get_context()));
        m_dnnl_stream = std::make_unique<dnnl::stream>(dnnl::sycl_interop::make_stream(*m_dnnl_engine, *m_queue));
        m_engine = std::make_unique<OneDNNInferenceEngine>(*m_queue, *m_dnnl_engine, *m_dnnl_stream);

        std::filesystem::path weights_path = std::filesystem::current_path();
        if (weights_path.filename() == "build") weights_path = weights_path.parent_path();
        weights_path = weights_path / "models" / "df3_weights";

        if (!m_engine->load_weights(weights_path.string())) return false;
        std::cout << "[SUCCESS] Neural Engine ready." << std::endl;

        setup_kernels();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] SYCL Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void SYCLAccelerator::setup_kernels() {
    if (!m_queue) return;

    try {
        m_fft_config = std::make_unique<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(static_cast<std::int64_t>(m_fft_size));
        m_fft_config->set_value(oneapi::mkl::dft::config_param::PLACEMENT, oneapi::mkl::dft::config_value::NOT_INPLACE);
        m_fft_config->commit(*m_queue);

        m_ifft_config = std::make_unique<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(static_cast<std::int64_t>(m_fft_size));
        m_ifft_config->set_value(oneapi::mkl::dft::config_param::PLACEMENT, oneapi::mkl::dft::config_value::NOT_INPLACE);
        m_ifft_config->commit(*m_queue);
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] FFT configuration failed: " << e.what() << std::endl;
        return;
    }

    auto q = *m_queue;
    m_window_buffer = sycl::malloc_device<float>(m_fft_size, q);
    m_analysis_mem = sycl::malloc_device<float>(m_fft_size, q);
    m_synthesis_mem = sycl::malloc_device<float>(m_fft_size - m_hop_size, q);
    m_freq_buffer = sycl::malloc_device<std::complex<float>>(m_freq_size, q);
    m_freq_history = sycl::malloc_device<std::complex<float>>(m_df_order * m_freq_size, q);
    m_reconstructed_frame = sycl::malloc_device<float>(m_fft_size, q);
    m_fft_input_scratch = sycl::malloc_device<float>(m_fft_size, q);
    m_filtered_freq_scratch = sycl::malloc_device<std::complex<float>>(m_freq_size, q);
    m_power_spectrum = sycl::malloc_device<float>(m_freq_size, q);
    m_erb_buffer = sycl::malloc_device<float>(m_nb_erb, q);
    m_erb_fb_matrix = sycl::malloc_device<float>(m_freq_size * m_nb_erb, q);
    m_erb_inv_fb_matrix = sycl::malloc_device<float>(m_nb_erb * m_freq_size, q);
    m_erb_norm_state = sycl::malloc_device<float>(m_nb_erb, q);
    m_spec_norm_state = sycl::malloc_device<float>(m_nb_df, q);
    m_df_coefs = sycl::malloc_device<float>(m_nb_df * m_df_order * 2, q);

    q.fill(m_analysis_mem, 0.0f, m_fft_size);
    q.fill(m_synthesis_mem, 0.0f, m_fft_size - m_hop_size);
    q.fill(reinterpret_cast<float*>(m_freq_history), 0.0f, m_df_order * m_freq_size * 2);
    q.fill(m_erb_norm_state, 0.0f, m_nb_erb);
    q.fill(m_spec_norm_state, 0.0f, m_nb_df);
    q.fill(m_df_coefs, 0.0f, m_nb_df * m_df_order * 2);
    q.wait();

    const double pi = 3.14159265358979323846;
    std::vector<float> host_window(m_fft_size);
    for (size_t i = 0; i < m_fft_size; ++i) {
        double sin_val = std::sin(0.5 * pi * (static_cast<double>(i) + 0.5) / (m_fft_size / 2.0));
        host_window[i] = static_cast<float>(std::sin(0.5 * pi * sin_val * sin_val));
    }
    q.memcpy(m_window_buffer, host_window.data(), m_fft_size * sizeof(float)).wait();

    std::filesystem::path path = std::filesystem::current_path();
    if (path.filename() == "build") path = path.parent_path();
    auto fb_path = path / "models" / "df3_weights" / "erb_fb.bin";
    std::ifstream fb_file(fb_path, std::ios::binary);
    if (fb_file) {
        std::vector<float> fb_weights(m_freq_size * m_nb_erb);
        fb_file.read(reinterpret_cast<char*>(fb_weights.data()), fb_weights.size() * sizeof(float));
        q.memcpy(m_erb_fb_matrix, fb_weights.data(), fb_weights.size() * sizeof(float)).wait();
    }
    auto inv_fb_path = path / "models" / "df3_weights" / "mask_erb_inv_fb.bin";
    std::ifstream inv_fb_file(inv_fb_path, std::ios::binary);
    if (inv_fb_file) {
        std::vector<float> inv_fb_weights(m_nb_erb * m_freq_size);
        inv_fb_file.read(reinterpret_cast<char*>(inv_fb_weights.data()), inv_fb_weights.size() * sizeof(float));
        q.memcpy(m_erb_inv_fb_matrix, inv_fb_weights.data(), inv_fb_weights.size() * sizeof(float)).wait();
        std::cout << "[INFO] Inverse Filterbank loaded." << std::endl;
    }
}

std::string SYCLAccelerator::get_device_name() const {
    return m_device_name;
}

void SYCLAccelerator::reset() {
    if (m_engine) {
        m_engine->reset();
    }
    
    // Reset ERB tracking means to reference defaults
    float start = -60.0f;
    float end = -90.0f;
    for (int i = 0; i < 32; ++i) {
        m_erb_mean[i] = start + i * (end - start) / 31.0f;
    }
    
    // Reset unit norm states
    start = 0.001f;
    end = 0.0001f;
    for (int i = 0; i < 96; ++i) {
        m_erb_var[i] = start + i * (end - start) / 95.0f;
    }

    if (m_queue) {
        m_queue->fill(reinterpret_cast<float*>(m_freq_history), 0.0f, m_df_order * m_freq_size * 2).wait();
        m_queue->fill(m_analysis_mem, 0.0f, m_fft_size).wait();
        m_queue->fill(m_synthesis_mem, 0.0f, m_fft_size - m_hop_size).wait();
    }
}

void SYCLAccelerator::process_frame(const float* input, float* output, size_t size) {
    if (!m_queue || size != m_hop_size) return;

    try {
        auto q = *m_queue;
        float* window = m_window_buffer;
        float* analysis = m_analysis_mem;
        float* synthesis = m_synthesis_mem;
        std::complex<float>* freq = m_freq_buffer;
        std::complex<float>* history = m_freq_history;
        float* reconstructed = m_reconstructed_frame;
        float* device_out = m_fft_input_scratch;
        std::complex<float>* filtered_freq = m_filtered_freq_scratch;

        // Local constants to avoid illegal 'this' capture
        const size_t fft_size_val = m_fft_size;
        const size_t hop_size_val = m_hop_size;
        const size_t freq_size_val = m_freq_size;
        const size_t overlap_size_val = m_fft_size - m_hop_size;
        const size_t df_order_val = m_df_order;
        const size_t nb_erb_val = m_nb_erb;
        const size_t nb_df_val = m_nb_df;
        const bool df_enabled_val = m_df_enabled;

        // 1. Analysis
        q.memcpy(analysis + overlap_size_val, input, hop_size_val * sizeof(float)).wait();
        float* fft_in = reconstructed;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(fft_size_val), [=](sycl::id<1> idx) {
                fft_in[idx] = analysis[idx] * window[idx];
            });
        }).wait();
        oneapi::mkl::dft::compute_forward(*m_fft_config, fft_in, freq).wait();

        const float wnorm = 1.0f / (static_cast<float>(fft_size_val * fft_size_val) / (2.0f * hop_size_val));
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(freq_size_val), [=](sycl::id<1> idx) {
                freq[idx] *= wnorm;
            });
        }).wait();

        // 2. Feature Extraction
        float* ps_ptr = m_power_spectrum;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(freq_size_val), [=](sycl::id<1> idx) {
                float re = freq[idx].real();
                float im = freq[idx].imag();
                ps_ptr[idx] = re * re + im * im;
            });
        }).wait();

        float* fb_ptr = m_erb_fb_matrix;
        float* erb_ptr = m_erb_buffer;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(nb_erb_val), [=](sycl::id<1> erb_idx) {
                float sum = 0.0f;
                for (size_t f = 0; f < freq_size_val; ++f) {
                    sum += ps_ptr[f] * fb_ptr[f * nb_erb_val + erb_idx];
                }
                erb_ptr[erb_idx] = sum;
            });
        }).wait();

        // 3. Normalization (Reference: 10 * log10 and band_mean_norm_erb)
        std::vector<float> host_erb(nb_erb_val);
        q.memcpy(host_erb.data(), m_erb_buffer, nb_erb_val * sizeof(float)).wait();
        const float alpha = 0.1f;
        for (size_t i = 0; i < nb_erb_val; ++i) {
            float lp = std::log10(host_erb[i] + 1e-10f) * 10.0f;
            m_erb_mean[i] = lp * (1.0f - alpha) + m_erb_mean[i] * alpha;
            host_erb[i] = (lp - m_erb_mean[i]) / 40.0f;
        }

        // 4. Complex Unit Normalization for DF path
        std::vector<std::complex<float>> host_freq_in(nb_df_val);
        std::vector<float> host_df_features(nb_df_val * 2);
        q.memcpy(host_freq_in.data(), freq, nb_df_val * sizeof(std::complex<float>)).wait();
        for (size_t i = 0; i < nb_df_val; ++i) {
            float n = std::sqrt(host_freq_in[i].real()*host_freq_in[i].real() + host_freq_in[i].imag()*host_freq_in[i].imag());
            m_erb_var[i] = n * (1.0f - alpha) + m_erb_var[i] * alpha;
            float s = std::sqrt(m_erb_var[i] + 1e-10f);
            host_df_features[i * 2 + 0] = host_freq_in[i].real() / s;
            host_df_features[i * 2 + 1] = host_freq_in[i].imag() / s;
        }

        // 5. Inference
        std::vector<float> host_mask(nb_erb_val);
        std::vector<float> host_df_coefs(nb_df_val * df_order_val * 2);
        if (m_engine) {
            m_engine->infer(host_erb.data(), host_df_features.data(), host_mask.data(), host_df_coefs.data());
        } else {
            std::fill(host_mask.begin(), host_mask.end(), 1.0f);
            std::fill(host_df_coefs.begin(), host_df_coefs.end(), 0.0f);
        }

        // 6. Apply Filtering & Synthesis
        q.memcpy(m_df_coefs, host_df_coefs.data(), host_df_coefs.size() * sizeof(float));
        q.memcpy(m_erb_buffer, host_mask.data(), nb_erb_val * sizeof(float)).wait();

        q.submit([&](sycl::handler& h) {
            float* m_ptr = m_erb_buffer;
            float* inv_fb_ptr = m_erb_inv_fb_matrix;
            float* c_ptr = m_df_coefs;
            std::complex<float>* h_ptr = history;
            h.parallel_for(sycl::range<1>(freq_size_val), [=](sycl::id<1> f_idx) {
                float m = 0.0f;
                for (size_t e = 0; e < nb_erb_val; ++e) {
                    m += m_ptr[e] * inv_fb_ptr[e * freq_size_val + f_idx];
                }
                std::complex<float> res = freq[f_idx] * std::clamp(m, 0.0f, 1.0f);

                if (df_enabled_val && f_idx < nb_df_val) {
                    std::complex<float> df(0.0f, 0.0f);
                    for (size_t i = 0; i < df_order_val; ++i) {
                        std::complex<float> c(c_ptr[(f_idx * df_order_val + i) * 2 + 0], c_ptr[(f_idx * df_order_val + i) * 2 + 1]);
                        df += c * h_ptr[i * freq_size_val + f_idx];
                    }
                    filtered_freq[f_idx] = df;
                } else {
                    filtered_freq[f_idx] = res;
                }

                // Update History (Frame Delay for next iteration)
                for (int i = static_cast<int>(df_order_val) - 1; i > 0; --i) {
                    h_ptr[i * freq_size_val + f_idx] = h_ptr[(i - 1) * freq_size_val + f_idx];
                }
                h_ptr[f_idx] = freq[f_idx];
            });
        }).wait();

        oneapi::mkl::dft::compute_backward(*m_ifft_config, filtered_freq, reconstructed).wait();

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(hop_size_val), [=](sycl::id<1> idx) {
                device_out[idx] = (reconstructed[idx] * window[idx]) + synthesis[idx];
                synthesis[idx] = reconstructed[idx + hop_size_val] * window[idx + hop_size_val];
            });
        }).wait();

        q.memcpy(output, device_out, hop_size_val * sizeof(float)).wait();
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(overlap_size_val), [=](sycl::id<1> idx) {
                analysis[idx] = analysis[idx + hop_size_val];
            });
        }).wait();
    } catch (const sycl::exception& e) {
        std::cerr << "[ERROR] SYCL Runtime Exception: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during process_frame: " << e.what() << std::endl;
    }
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
void sycl_set_df_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(sa::infrastructure::g_accel_mutex);
    if (sa::infrastructure::g_accelerator) {
        sa::infrastructure::g_accelerator->set_deep_filtering_enabled(enabled);
    }
}
void sycl_reset() {
    std::lock_guard<std::mutex> lock(sa::infrastructure::g_accel_mutex);
    if (sa::infrastructure::g_accelerator) {
        sa::infrastructure::g_accelerator->reset();
    }
}
}
