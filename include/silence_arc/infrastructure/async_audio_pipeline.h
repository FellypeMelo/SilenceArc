#pragma once

#include "silence_arc/domain/audio_pipeline.h"
#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>

namespace sa::infrastructure {

class AsyncAudioPipeline : public domain::IAudioPipeline {
public:
    AsyncAudioPipeline();
    ~AsyncAudioPipeline() override;

    bool Start(const std::string& input_device_id = "", const std::string& output_device_id = "") override;
    void Stop() override;
    bool IsRunning() const override { return is_running_; }

    void SetProcessCallback(std::function<void(const domain::AudioBuffer&, domain::AudioBuffer&)> callback);

    void PushInput(const domain::AudioBuffer& buffer);
    bool PopOutput(domain::AudioBuffer& buffer);

private:
    void ProcessingLoop();

    bool is_running_;
    std::thread processing_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    
    std::function<void(const domain::AudioBuffer&, domain::AudioBuffer&)> callback_;
    
    std::vector<domain::AudioBuffer> input_queue_;
    std::vector<domain::AudioBuffer> output_queue_;
};

} // namespace sa::infrastructure
