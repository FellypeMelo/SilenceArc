#ifndef SILENCE_ARC_INFRASTRUCTURE_ASYNC_AUDIO_PIPELINE_H_
#define SILENCE_ARC_INFRASTRUCTURE_ASYNC_AUDIO_PIPELINE_H_

#include "silence_arc/domain/audio_pipeline.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <deque>
#include <cstddef>
#include <cstdint>
#include <condition_variable>

namespace silence_arc {
namespace infrastructure {

// Runs the (potentially slow, GPU-bound) noise-suppression callback on a
// dedicated TIME_CRITICAL worker thread so it never blocks the real-time audio
// device callback. The audio callback stays a thin push/pop shim.
//
// Both queues are BOUNDED with a drop-oldest policy: if the producer (audio
// callback) outruns the worker, or the consumer falls behind the worker, the
// stalest frame is discarded and `FramesDropped()` is incremented rather than
// letting latency grow unbounded (the old std::vector queues were unbounded with
// O(n) erase(begin())).
class AsyncAudioPipeline : public domain::IAudioPipeline {
public:
    AsyncAudioPipeline();
    ~AsyncAudioPipeline() override;

    bool Start(const std::string& input_device_id = "", const std::string& output_device_id = "") override;
    void Stop() override;
    bool IsRunning() const override { return is_running_; }

    void SetProcessCallback(domain::IAudioPipeline::ProcessCallback callback) override;

    // Producer/consumer API used by the composing device callback (and tests).
    void PushInput(const domain::AudioBuffer& buffer);
    bool PopOutput(domain::AudioBuffer& buffer);

    // Telemetry / tuning.
    void SetMaxQueueDepth(size_t depth) { max_queue_depth_ = depth ? depth : 1; }
    size_t MaxQueueDepth() const { return max_queue_depth_; }
    uint64_t FramesDropped() const { return frames_dropped_.load(std::memory_order_relaxed); }

private:
    void ThreadLoop();

    std::atomic<bool> is_running_{false};
    std::thread worker_thread_;
    domain::IAudioPipeline::ProcessCallback callback_;
    mutable std::mutex callback_mutex_;

    std::deque<domain::AudioBuffer> input_queue_;
    std::deque<domain::AudioBuffer> output_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable cv_;

    std::atomic<size_t> max_queue_depth_{8};   // ~80 ms of 10 ms frames
    std::atomic<uint64_t> frames_dropped_{0};
};

} // namespace infrastructure
} // namespace silence_arc

#endif // SILENCE_ARC_INFRASTRUCTURE_ASYNC_AUDIO_PIPELINE_H_
