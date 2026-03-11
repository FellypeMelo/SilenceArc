#pragma once

#include <vector>
#include <string>
#include <memory>

namespace sa::domain {

/**
 * @brief Domain interface for GPU-accelerated noise suppression operations.
 * Following Clean Architecture, this interface remains pure and decoupled from SYCL specifics.
 */
class GPUAccelerator {
public:
    virtual ~GPUAccelerator() = default;

    /**
     * @brief Initialize the GPU device.
     * @return true if successful, false otherwise.
     */
    virtual bool initialize() = 0;

    /**
     * @brief Get the name of the active GPU device.
     */
    virtual std::string get_device_name() const = 0;

    /**
     * @brief Process a frame of audio data on the GPU.
     * @param input Input buffer (time-domain or frequency-domain).
     * @param output Output buffer.
     * @param size Number of elements.
     */
    virtual void process_frame(const float* input, float* output, size_t size) = 0;

    /**
     * @brief Toggle Deep Filtering (Complex Convolution) path.
     * If disabled, only ERB masking is applied (more natural sound).
     */
    virtual void set_deep_filtering_enabled(bool enabled) = 0;
};

} // namespace sa::domain
