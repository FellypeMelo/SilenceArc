#pragma once

#include "silence_arc/domain/audio_pipeline.h"
#include <string>
#include <memory>
#include <functional>

namespace sa::infrastructure {

class MiniaudioPipeline : public domain::IAudioPipeline {
public:
    using ProcessCallback = std::function<void(const domain::AudioBuffer&, domain::AudioBuffer&)>;

    MiniaudioPipeline();
    ~MiniaudioPipeline() override;

    bool Start(const std::string& input_device_id, const std::string& output_device_id) override;
    void Stop() override;
    bool IsRunning() const override;

    void SetProcessCallback(ProcessCallback callback);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace sa::infrastructure
