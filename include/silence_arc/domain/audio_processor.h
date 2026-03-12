#pragma once

#include <string>
#include <cstddef>

namespace sa::domain {

/**
 * @brief Unified interface for all audio processing components.
 * Combines general processor needs with noise suppression specifics.
 */
class IAudioProcessor {
public:
    virtual ~IAudioProcessor() = default;

    /**
     * @brief Initialize the processor (loads models, initializes hardware).
     */
    virtual bool initialize() = 0;

    /**
     * @brief Get the name of the active processing device.
     */
    virtual std::string get_device_name() const = 0;

    /**
     * @brief Process a single frame of audio data.
     * @param input  Pointer to the input time-domain buffer.
     * @param output Pointer to the output time-domain buffer.
     * @param size   Number of samples in the frame.
     */
    virtual void process_frame(const float* input, float* output, size_t size) = 0;

    /**
     * @brief Returns the frame length expected by this processor.
     */
    virtual size_t get_frame_size() const = 0;

    /**
     * @brief Returns the processing latency in samples.
     */
    virtual size_t get_latency() const = 0;

    /**
     * @brief Toggle Deep Filtering features.
     */
    virtual void set_deep_filtering_enabled(bool enabled) = 0;

    /**
     * @brief Set noise attenuation limit in dB.
     */
    virtual void set_attenuation_limit(float limit_db) = 0;

    /**
     * @brief Resets internal recurrent states and buffers.
     */
    virtual void reset() = 0;
};

} // namespace sa::domain
