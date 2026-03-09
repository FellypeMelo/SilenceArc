#ifndef SILENCE_ARC_INFRASTRUCTURE_ASYNC_AUDIO_PIPELINE_H_
#define SILENCE_ARC_INFRASTRUCTURE_ASYNC_AUDIO_PIPELINE_H_

#include "silence_arc/domain/audio_pipeline.h"
#include <thread>
#include <atomic>
#include <mutex>

namespace silence_arc {
namespace infrastructure {

class AsyncAudioPipeline : public domain::IAudioPipeline {
public:
    AsyncAudioPipeline();
    ~AsyncAudioPipeline() override;

    bool Start(const std::string& input_device_id = "", const std::string& output_device_id = "") override;
    void Stop() override;
    bool IsRunning() const override { return is_running_; }

    void SetProcessCallback(domain::IAudioPipeline::ProcessCallback callback) override;

    // Simulation methods for testing
    void PushInput(const domain::AudioBuffer& buffer);
    bool PopOutput(domain::AudioBuffer& buffer);

private:
    void ThreadLoop();

    std::atomic<bool> is_running_{false};
    std::thread worker_thread_;
    domain::IAudioPipeline::ProcessCallback callback_;
    mutable std::mutex callback_mutex_;

    std::vector<domain::AudioBuffer> input_queue_;
    std::vector<domain::AudioBuffer> output_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable cv_;
};

} // namespace infrastructure
} // namespace silence_arc

#endif // SILENCE_ARC_INFRASTRUCTURE_ASYNC_AUDIO_PIPELINE_H_
