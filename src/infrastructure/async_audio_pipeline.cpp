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
        input_queue_.push_back(buffer);
    }
    cv_.notify_one();
}

bool AsyncAudioPipeline::PopOutput(domain::AudioBuffer& buffer) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (output_queue_.empty()) return false;
    buffer = output_queue_.front();
    output_queue_.erase(output_queue_.begin());
    return true;
}

void AsyncAudioPipeline::ThreadLoop() {
    while (is_running_) {
        domain::AudioBuffer input;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { return !is_running_ || !input_queue_.empty(); });
            
            if (!is_running_) break;
            
            input = input_queue_.front();
            input_queue_.erase(input_queue_.begin());
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
            output_queue_.push_back(output);
        }
    }
}

} // namespace infrastructure
} // namespace silence_arc
