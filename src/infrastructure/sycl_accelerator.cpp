#include "silence_arc/infrastructure/sycl_accelerator.h"
#include "silence_arc/domain/neural_engine.h"
#include "silence_arc/infrastructure/onednn_inference_engine.h"
#include "silence_arc/infrastructure/sycl_dsp_coordinator.h"
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
    m_erb_mean.resize(32);
    m_erb_var.resize(96);
    reset_stats();
}

SYCLAccelerator::~SYCLAccelerator() {
    cleanup();
}

void SYCLAccelerator::reset_stats() {
    float start = -60.0f;
    float end = -90.0f;
    for (int i = 0; i < 32; ++i) {
        m_erb_mean[i] = start + i * (end - start) / 31.0f;
    }
    start = 0.001f;
    end = 0.0001f;
    for (int i = 0; i < 96; ++i) {
        m_erb_var[i] = start + i * (end - start) / 95.0f;
    }
}

void SYCLAccelerator::cleanup() {
    m_dsp.reset();
    m_engine.reset();
    m_dnnl_stream.reset();
    m_dnnl_engine.reset();
}

bool SYCLAccelerator::initialize() {
    if (m_dsp && m_engine) return true; // Already initialized

    try {
        if (!m_queue) {
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
        }

        if (!m_dnnl_engine) {
            m_dnnl_engine = std::make_unique<dnnl::engine>(dnnl::sycl_interop::make_engine(m_queue->get_device(), m_queue->get_context()));
            m_dnnl_stream = std::make_unique<dnnl::stream>(dnnl::sycl_interop::make_stream(*m_dnnl_engine, *m_queue));
        }

        /* 
         * [REVERTED] Native Inference Engine disabled to restore stability via Rust Adapter.
         * Telemetry remains functional via Level Zero discovery above.
         *
        if (!m_engine) {
            m_engine = std::make_unique<OneDNNInferenceEngine>(*m_queue, *m_dnnl_engine, *m_dnnl_stream);
            std::filesystem::path weights_path = std::filesystem::current_path();
            if (weights_path.filename() == "build") weights_path = weights_path.parent_path();
            weights_path = weights_path / "models" / "df3_weights";

            if (!m_engine->load_weights(weights_path.string())) return false;
        }
        
        if (!m_dsp) {
            m_dsp = std::make_unique<SYCLDSPCoordinator>(*m_queue, m_fft_size, m_hop_size);
            m_dsp->initialize();
            setup_kernels();
        }
        */

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] SYCL Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void SYCLAccelerator::setup_kernels() {
    if (!m_queue || !m_dsp) return;
    auto q = *m_queue;

    std::filesystem::path path = std::filesystem::current_path();
    if (path.filename() == "build") path = path.parent_path();
    
    auto fb_path = path / "models" / "df3_weights" / "erb_fb.bin";
    std::ifstream fb_file(fb_path, std::ios::binary);
    if (fb_file) {
        std::vector<float> fb_weights(m_freq_size * m_nb_erb);
        fb_file.read(reinterpret_cast<char*>(fb_weights.data()), fb_weights.size() * sizeof(float));
        q.memcpy(m_dsp->get_erb_fb_matrix(), fb_weights.data(), fb_weights.size() * sizeof(float)).wait();
    }
    
    auto inv_fb_path = path / "models" / "df3_weights" / "mask_erb_inv_fb.bin";
    std::ifstream inv_fb_file(inv_fb_path, std::ios::binary);
    if (inv_fb_file) {
        std::vector<float> inv_fb_weights(m_nb_erb * m_freq_size);
        inv_fb_file.read(reinterpret_cast<char*>(inv_fb_weights.data()), inv_fb_weights.size() * sizeof(float));
        q.memcpy(m_dsp->get_erb_inv_fb_matrix(), inv_fb_weights.data(), inv_fb_weights.size() * sizeof(float)).wait();
        std::cout << "[INFO] Inverse Filterbank loaded." << std::endl;
    }
}

std::string SYCLAccelerator::get_device_name() const {
    return m_device_name;
}

void SYCLAccelerator::reset() {
    if (m_engine) m_engine->reset();
    reset_stats();
    if (m_queue && m_dsp) {
        m_queue->fill(reinterpret_cast<float*>(m_dsp->get_freq_history()), 0.0f, m_df_order * m_freq_size * 2).wait();
        m_queue->fill(m_dsp->get_analysis_mem(), 0.0f, m_fft_size).wait();
        m_queue->fill(m_dsp->get_synthesis_mem(), 0.0f, m_fft_size - m_hop_size).wait();
    }
}

void SYCLAccelerator::process_frame(const float* input, float* output, size_t size) {
    if (!m_queue || !m_dsp || size != m_hop_size) return;

    try {
        auto q = *m_queue;
        
        // 1. Analysis
        m_dsp->analyze(input, m_dsp->get_window_buffer(), m_dsp->get_analysis_mem(), m_dsp->get_freq_buffer());

        // 2. Feature Extraction
        m_dsp->extract_power_spectrum(m_dsp->get_freq_buffer(), m_dsp->get_power_spectrum());
        m_dsp->apply_erb_filterbank(m_dsp->get_power_spectrum(), m_dsp->get_erb_fb_matrix(), m_dsp->get_erb_buffer());

        // 3. Normalization (ERB log-scale)
        std::vector<float> host_erb(m_nb_erb);
        q.memcpy(host_erb.data(), m_dsp->get_erb_buffer(), m_nb_erb * sizeof(float)).wait();
        
        const float alpha = 0.9f; 
        for (size_t i = 0; i < m_nb_erb; ++i) {
            float lp = std::log10(host_erb[i] + 1e-10f) * 10.0f;
            m_erb_mean[i] = lp * (1.0f - alpha) + m_erb_mean[i] * alpha;
            host_erb[i] = (lp - m_erb_mean[i]) / 20.0f;
        }

        // 4. Complex Unit Normalization for DF path
        std::vector<std::complex<float>> host_freq_in(m_nb_df);
        std::vector<float> host_df_features(m_nb_df * 2);
        q.memcpy(host_freq_in.data(), m_dsp->get_freq_buffer(), m_nb_df * sizeof(std::complex<float>)).wait();
        for (size_t i = 0; i < m_nb_df; ++i) {
            float n = std::sqrt(host_freq_in[i].real()*host_freq_in[i].real() + host_freq_in[i].imag()*host_freq_in[i].imag());
            m_erb_var[i] = n * (1.0f - alpha) + m_erb_var[i] * alpha;
            float s = std::sqrt(m_erb_var[i] + 1e-10f);
            host_df_features[i * 2 + 0] = host_freq_in[i].real() / s;
            host_df_features[i * 2 + 1] = host_freq_in[i].imag() / s;
        }

        // 5. Inference
        std::vector<float> host_mask(m_nb_erb);
        std::vector<float> host_df_coefs(m_nb_df * m_df_order * 2);
        if (m_engine) {
            m_engine->infer(host_erb.data(), host_df_features.data(), host_mask.data(), host_df_coefs.data());
            
            static int debug_count = 0;
            if (debug_count++ % 100 == 0) {
                float mask_sum = 0;
                for(float m : host_mask) mask_sum += m;
                float coef_sum = 0;
                for(float c : host_df_coefs) coef_sum += std::abs(c);
                std::cout << "[DEBUG] Frame " << debug_count << " Mask Sum: " << mask_sum << " Coef Sum: " << coef_sum << std::endl;
            }
        } else {
            std::fill(host_mask.begin(), host_mask.end(), 1.0f);
            std::fill(host_df_coefs.begin(), host_df_coefs.end(), 0.0f);
        }

        // 6. Shift and update history with RAW signal
        std::complex<float>* h_ptr = m_dsp->get_freq_history();
        auto freq_buf = m_dsp->get_freq_buffer();
        const size_t freq_size = m_freq_size;
        const size_t df_order = m_df_order;

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(freq_size), [=](sycl::id<1> f_idx) {
                for (int i = static_cast<int>(df_order) - 1; i > 0; --i) {
                    h_ptr[i * freq_size + f_idx] = h_ptr[(i - 1) * freq_size + f_idx];
                }
                h_ptr[f_idx] = freq_buf[f_idx]; 
            });
        }).wait();

        // 7. Apply Filtering
        q.memcpy(m_dsp->get_df_coefs(), host_df_coefs.data(), host_df_coefs.size() * sizeof(float));
        q.memcpy(m_dsp->get_erb_buffer(), host_mask.data(), m_nb_erb * sizeof(float)).wait();

        if (m_df_enabled) {
            m_dsp->apply_df_coefficients(m_dsp->get_freq_history(), m_dsp->get_df_coefs(), m_dsp->get_filtered_freq_scratch());
            q.wait(); // Synchronize before next step using the scratch buffer

            // Copy high frequency bins from ERB path (target frame t-2)
            const size_t nb_df = m_nb_df;
            const size_t nb_erb = m_nb_erb;
            q.submit([&](sycl::handler& h) {
                auto filtered = m_dsp->get_filtered_freq_scratch();
                auto history = m_dsp->get_freq_history();
                auto mask_ptr = m_dsp->get_erb_buffer();
                auto inv_fb = m_dsp->get_erb_inv_fb_matrix();
                h.parallel_for(sycl::range<1>(freq_size - nb_df), [=](sycl::id<1> idx) {
                    size_t f_idx = idx[0] + nb_df;
                    float m = 0.0f;
                    for (size_t e = 0; e < nb_erb; ++e) {
                        m += mask_ptr[e] * inv_fb[e * freq_size + f_idx];
                    }
                    // Lookahead 2: history[2] is frame t-2
                    filtered[f_idx] = history[2 * freq_size + f_idx] * std::clamp(m, 0.0f, 1.0f);
                });
            }).wait();
        } else {
            m_dsp->apply_erb_mask(m_dsp->get_freq_buffer(), m_dsp->get_erb_buffer(), m_dsp->get_erb_inv_fb_matrix(), m_dsp->get_filtered_freq_scratch());
            q.wait();
        }
        // 8. Synthesis
        m_dsp->synthesize(m_dsp->get_filtered_freq_scratch(), m_dsp->get_window_buffer(), m_dsp->get_reconstructed_frame(), m_dsp->get_synthesis_mem(), output);

        // 9. Shift analysis buffer
        m_dsp->shift_analysis_buffer(m_dsp->get_analysis_mem());

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during SYCLAccelerator::process_frame: " << e.what() << std::endl;
    }
}

} // namespace sa::infrastructure

extern "C" {
bool sycl_init() {
    std::lock_guard<std::mutex> lock(sa::infrastructure::g_accel_mutex);
    if (!sa::infrastructure::g_accelerator) {
        sa::infrastructure::g_accelerator = std::make_unique<sa::infrastructure::SYCLAccelerator>();
        if (!sa::infrastructure::g_accelerator->initialize()) {
            sa::infrastructure::g_accelerator.reset();
            return false;
        }
    }
    return true;
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
