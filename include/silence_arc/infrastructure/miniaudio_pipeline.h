#ifndef SILENCE_ARC_INFRASTRUCTURE_MINIAUDIO_PIPELINE_H_
#define SILENCE_ARC_INFRASTRUCTURE_MINIAUDIO_PIPELINE_H_

#include "silence_arc/domain/audio_pipeline.h"
#include <memory>
#include <atomic>

namespace silence_arc {
namespace infrastructure {

class MiniaudioPipeline : public domain::IAudioPipeline {
public:
    MiniaudioPipeline();
    ~MiniaudioPipeline() override;

    bool Start(const std::string& input_device_id = "", const std::string& output_device_id = "") override;
    void Stop() override;
    bool IsRunning() const override;

    void SetProcessCallback(domain::IAudioPipeline::ProcessCallback callback) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace infrastructure
} // namespace silence_arc

#endif // SILENCE_ARC_INFRASTRUCTURE_MINIAUDIO_PIPELINE_H_
