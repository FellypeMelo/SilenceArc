#pragma once

#include "silence_arc/domain/audio_processor.h"
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/dft.hpp>
#include <dnnl.hpp>
#include <optional>
#include <vector>
#include <complex>
#include <mutex>

namespace sa::infrastructure {

/**
 * @brief Legacy SYCL Accelerator (Native Engine v1).
 * Now being refactored to a telemetry wrapper and pass-through.
 * NativeSyclEngine (v2) handles full inference.
 */
class SYCLAccelerator : public domain::IAudioProcessor {
public:
    SYCLAccelerator();
    ~SYCLAccelerator() override;

    // IAudioProcessor Implementation
    bool initialize() override;
    std::string get_device_name() const override;
    void process_frame(const float* input, float* output, size_t size) override;
    size_t get_frame_size() const override;
    size_t get_latency() const override;
    void set_deep_filtering_enabled(bool enabled) override;
    void set_attenuation_limit(float limit_db) override;
    void reset() override;

    // Telemetry access
    float get_gpu_load();
    float get_vram_usage();

private:
    void setup_kernels();

    size_t m_fft_size = 960;
    size_t m_hop_size = 480;
    bool m_initialized = false;
    bool m_df_enabled = true;
    float m_attenuation_limit = 40.0f;

    std::unique_ptr<sycl::queue> m_queue;
    std::unique_ptr<dnnl::engine> m_dnnl_engine;
    std::unique_ptr<dnnl::stream> m_dnnl_stream;
};

// Global singleton access for C API
extern std::unique_ptr<SYCLAccelerator> g_accelerator;
extern std::mutex g_accel_mutex;

} // namespace sa::infrastructure

extern "C" {
    bool sycl_init();
    void sycl_process(const float* input, float* output, size_t size);
    void sycl_get_device_name(char* buffer, size_t max_size);
    void sycl_set_df_enabled(bool enabled);
    void sycl_reset();
}
