#pragma once

#include <string>
#include <cstddef>

namespace sa::domain {

/**
 * @brief Abstract interface for high-level audio processing operations.
 * Decoupled from specific hardware acceleration (GPU/CPU).
 */
class IAudioProcessor {
public:
    virtual ~IAudioProcessor() = default;

    /**
     * @brief Initialize the hardware and underlying processing engine.
     * @return true if successful, false otherwise.
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
     * @brief Toggle specific processing modes (e.g., Deep Filtering).
     */
    virtual void set_deep_filtering_enabled(bool enabled) = 0;
};

} // namespace sa::domain
