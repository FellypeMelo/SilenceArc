#include "silence_arc/infrastructure/deep_filter_adapter.h"
#include "deep_filter.h"
#include <stdexcept>
#include <filesystem>
#include <iostream>

namespace sa::infrastructure {

struct DeepFilterAdapter::Impl {
    DFState* state = nullptr;
    size_t frame_length = 0;
};

DeepFilterAdapter::DeepFilterAdapter(std::string model_path) 
    : m_impl(std::make_unique<Impl>()), m_model_path(std::move(model_path)) {}

DeepFilterAdapter::~DeepFilterAdapter() {
    if (m_impl->state) {
        df_free(m_impl->state);
    }
}

bool DeepFilterAdapter::initialize() {
    if (m_impl->state) {
        df_free(m_impl->state);
        m_impl->state = nullptr;
    }

    if (!std::filesystem::exists(m_model_path)) {
        std::cerr << "[ERROR] DeepFilter model not found at: " << m_model_path << std::endl;
        return false;
    }

    // Initialize with default attenuation limit 40.0 dB
    m_impl->state = df_create(m_model_path.c_str(), 40.0f, nullptr);
    if (!m_impl->state) {
        std::cerr << "[ERROR] Failed to create DeepFilter state from model." << std::endl;
        return false;
    }

    m_impl->frame_length = df_get_frame_length(m_impl->state);
    return true;
}

std::string DeepFilterAdapter::get_device_name() const {
    return "CPU (DeepFilterNet Rust Runtime)";
}

void DeepFilterAdapter::process_frame(const float* input, float* output, size_t size) {
    if (!m_impl->state || size != m_impl->frame_length) {
        return;
    }
    df_process_frame(m_impl->state, const_cast<float*>(input), output);
}

size_t DeepFilterAdapter::get_frame_size() const {
    return m_impl->frame_length;
}

size_t DeepFilterAdapter::get_latency() const {
    // DeepFilterNet3 usually has a latency of 2 frames (lookahead)
    return m_impl->frame_length * 2; 
}

void DeepFilterAdapter::set_deep_filtering_enabled(bool /*enabled*/) {
    // Rust adapter handles this internally
}

void DeepFilterAdapter::set_attenuation_limit(float limit_db) {
    if (m_impl->state) {
        df_set_atten_lim(m_impl->state, limit_db);
    }
}

void DeepFilterAdapter::reset() {
    // DLL doesn't expose a reset, so we re-initialize the state
    initialize();
}

} // namespace sa::infrastructure
