#include "silence_arc/infrastructure/sycl_accelerator.h"
#include <dnnl_sycl.hpp>
#include <iostream>
#include <fstream>
#include <mutex>
#include <cmath>
#include <filesystem>
#include <algorithm>

namespace sa::infrastructure {

std::unique_ptr<SYCLAccelerator> g_accelerator = nullptr;
std::mutex g_accel_mutex;

SYCLAccelerator::SYCLAccelerator() {}

SYCLAccelerator::~SYCLAccelerator() {
    m_dnnl_stream.reset();
    m_dnnl_engine.reset();
    m_queue.reset();
}

bool SYCLAccelerator::initialize() {
    if (m_initialized) return true;

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

            m_queue = std::make_unique<sycl::queue>(device, sycl::property::queue::in_order());
            std::cout << "[INFO] SYCL Initialized on: " << device.get_info<sycl::info::device::name>() << std::endl;
        }

        if (!m_dnnl_engine) {
            m_dnnl_engine = std::make_unique<dnnl::engine>(dnnl::sycl_interop::make_engine(m_queue->get_device(), m_queue->get_context()));
            m_dnnl_stream = std::make_unique<dnnl::stream>(dnnl::sycl_interop::make_stream(*m_dnnl_engine, *m_queue));
        }

        m_initialized = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] SYCL Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

std::string SYCLAccelerator::get_device_name() const {
    if (!m_queue) return "Not Initialized";
    return m_queue->get_device().get_info<sycl::info::device::name>();
}

void SYCLAccelerator::process_frame(const float* input, float* output, size_t size) {
    // SYCLAccelerator is currently a telemetry wrapper and pass-through.
    // Full inference is handled by NativeSyclEngine.
    std::copy(input, input + size, output);
}

size_t SYCLAccelerator::get_frame_size() const {
    return m_hop_size;
}

size_t SYCLAccelerator::get_latency() const {
    return 0;
}

void SYCLAccelerator::set_deep_filtering_enabled(bool enabled) {
    m_df_enabled = enabled;
}

void SYCLAccelerator::set_attenuation_limit(float limit_db) {
    m_attenuation_limit = limit_db;
}

void SYCLAccelerator::reset() {}

float SYCLAccelerator::get_gpu_load() {
    return 0.0f;
}

float SYCLAccelerator::get_vram_usage() {
    return 0.0f;
}

void SYCLAccelerator::setup_kernels() {}

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
