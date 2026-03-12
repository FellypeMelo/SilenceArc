#pragma once

#include "silence_arc/domain/audio_processor.h"
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/dft.hpp>
#include <dnnl.hpp>
#include <optional>
#include <vector>
#include <complex>
#include <memory>

namespace sa::domain {
    class INeuralEngine;
}

namespace sa::infrastructure {

class SYCLDSPCoordinator;

/**
 * @brief SYCL implementation of IAudioProcessor optimized for Intel Arc (oneAPI).
 * Uses Unified Shared Memory (USM) for zero-copy performance.
 */
class alignas(64) SYCLAccelerator : public domain::IAudioProcessor {
public:
    SYCLAccelerator();
    ~SYCLAccelerator() override;

    bool initialize() override;
    std::string get_device_name() const override;
    void process_frame(const float* input, float* output, size_t size) override;
    void set_deep_filtering_enabled(bool enabled) override { m_df_enabled = enabled; }
    void reset();

    // Internal SYCL/oneDNN objects (decoupled from domain)
    sycl::queue& get_queue() { return *m_queue; }
    dnnl::engine& get_dnnl_engine() { return *m_dnnl_engine; }
    dnnl::stream& get_dnnl_stream() { return *m_dnnl_stream; }

private:
    std::optional<sycl::queue> m_queue;
    std::string m_device_name;

    // oneDNN Engine and Stream
    std::unique_ptr<dnnl::engine> m_dnnl_engine;
    std::unique_ptr<dnnl::stream> m_dnnl_stream;

    std::unique_ptr<domain::INeuralEngine> m_engine;
    std::unique_ptr<SYCLDSPCoordinator> m_dsp;

    // Constants for DeepFilterNet

    const size_t m_fft_size = 960;
    const size_t m_hop_size = 480;
    const size_t m_freq_size = m_fft_size / 2 + 1;
    const size_t m_df_order = 5;
    const size_t m_nb_erb = 32;
    const size_t m_nb_df = 96;

    bool m_df_enabled = true;

    // Tracking Statistics for Normalization
    std::vector<float> m_erb_mean;
    std::vector<float> m_erb_var;

    void reset_stats();
    void setup_kernels();
    void cleanup();
};

} // namespace sa::infrastructure

extern "C" {
    bool sycl_init();
    void sycl_process(const float* input, float* output, size_t size);
    void sycl_get_device_name(char* buffer, size_t max_size);
    void sycl_set_df_enabled(bool enabled);
    void sycl_reset();
}
