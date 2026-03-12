#pragma once

#include <vector>
#include <string>

namespace sa::domain {

struct AudioBuffer {
    std::vector<float> data;
    size_t sample_rate;
    size_t num_channels;
};

/**
 * @brief Interface for audio I/O streaming systems (e.g. Miniaudio).
 */
class IAudioPipeline {
public:
    virtual ~IAudioPipeline() = default;

    virtual bool Start(const std::string& input_device_id, const std::string& output_device_id) = 0;
    virtual void Stop() = 0;
    virtual bool IsRunning() const = 0;
};

} // namespace sa::domain
