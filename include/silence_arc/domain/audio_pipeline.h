#ifndef SILENCE_ARC_DOMAIN_AUDIO_PIPELINE_H_
#define SILENCE_ARC_DOMAIN_AUDIO_PIPELINE_H_

#include <vector>
#include <functional>
#include <string>

namespace silence_arc {
namespace domain {

struct AudioBuffer {
    std::vector<float> data;
    size_t sample_rate = 48000;
};

class IAudioPipeline {
public:
    virtual ~IAudioPipeline() = default;

    virtual bool Start() = 0;
    virtual void Stop() = 0;
    virtual bool IsRunning() const = 0;

    using ProcessCallback = std::function<void(const AudioBuffer& input, AudioBuffer& output)>;
    virtual void SetProcessCallback(ProcessCallback callback) = 0;
};

} // namespace domain
} // namespace silence_arc

#endif // SILENCE_ARC_DOMAIN_AUDIO_PIPELINE_H_
