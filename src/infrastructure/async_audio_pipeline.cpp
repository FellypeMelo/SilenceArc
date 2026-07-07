#include "silence_arc/infrastructure/async_audio_pipeline.h"
#include <windows.h>

namespace silence_arc {
namespace infrastructure {

AsyncAudioPipeline::AsyncAudioPipeline() {}

AsyncAudioPipeline::~AsyncAudioPipeline() {
    Stop();
}

bool AsyncAudioPipeline::Start(const std::string& input_device_id, const std::string& output_device_id) {
    if (is_running_) return false;
    is_running_ = true;
    worker_thread_ = std::thread(&AsyncAudioPipeline::ThreadLoop, this);
    
    // Set high priority for the audio thread
    HANDLE handle = reinterpret_cast<HANDLE>(worker_thread_.native_handle());
    SetThreadPriority(handle, THREAD_PRIORITY_TIME_CRITICAL);
    
    return true;
}

void AsyncAudioPipeline::Stop() {
    is_running_ = false;
    cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void AsyncAudioPipeline::SetProcessCallback(domain::IAudioPipeline::ProcessCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callback_ = callback;
}

void AsyncAudioPipeline::PushInput(const domain::AudioBuffer& buffer) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        const size_t cap = max_queue_depth_.load(std::memory_order_relaxed);
        while (input_queue_.size() >= cap) {
            // Producer outran the worker: drop the STALEST input so we track the
            // live signal instead of accumulating latency.
            input_queue_.pop_front();
            frames_dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        input_queue_.push_back(buffer);
    }
    cv_.notify_one();
}

bool AsyncAudioPipeline::PopOutput(domain::AudioBuffer& buffer) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (output_queue_.empty()) return false;
    buffer = std::move(output_queue_.front());
    output_queue_.pop_front();
    return true;
}

void AsyncAudioPipeline::ThreadLoop() {
    while (is_running_) {
        domain::AudioBuffer input;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { return !is_running_ || !input_queue_.empty(); });

            if (!is_running_) break;

            input = std::move(input_queue_.front());
            input_queue_.pop_front();
        }

        domain::AudioBuffer output;
        output.sample_rate = input.sample_rate;
        output.data.resize(input.data.size());

        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (callback_) {
                callback_(input, output);
            } else {
                output.data = input.data; // Pass-through
            }
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            const size_t cap = max_queue_depth_.load(std::memory_order_relaxed);
            while (output_queue_.size() >= cap) {
                // Consumer (audio callback) fell behind: drop the stalest output.
                output_queue_.pop_front();
                frames_dropped_.fetch_add(1, std::memory_order_relaxed);
            }
            output_queue_.push_back(std::move(output));
        }
    }
}

} // namespace infrastructure
} // namespace silence_arc
