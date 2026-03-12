#pragma once

#include "silence_arc/domain/audio_processor.h"
#include <string>
#include <memory>

namespace sa::infrastructure {

/**
 * @brief Stable Rust-based adapter for DeepFilterNet3.
 */
class DeepFilterAdapter : public domain::IAudioProcessor {
public:
    DeepFilterAdapter(std::string model_path);
    ~DeepFilterAdapter() override;

    // IAudioProcessor Implementation
    bool initialize() override;
    std::string get_device_name() const override;
    void process_frame(const float* input, float* output, size_t size) override;
    size_t get_frame_size() const override;
    size_t get_latency() const override;
    void set_deep_filtering_enabled(bool enabled) override;
    void set_attenuation_limit(float limit_db) override;
    void reset() override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    std::string m_model_path;
};

} // namespace sa::infrastructure
