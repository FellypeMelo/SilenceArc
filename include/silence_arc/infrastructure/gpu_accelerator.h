#pragma once

#include <vector>
#include <string>
#include <memory>

namespace silence_arc::infrastructure {

/**
 * @brief Infrastructure-internal abstraction for GPU-accelerated frame
 * processing. It is the DSP half of a Bridge that keeps the SYCL/oneMKL device
 * code separable from the neural-network engine; it is NOT a backend-selection
 * seam (that is domain::INoiseSuppressor). Private to the SYCL adapter.
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

} // namespace silence_arc::infrastructure
