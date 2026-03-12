#include "silence_arc/infrastructure/async_audio_pipeline.h"
#include <iostream>

namespace sa::infrastructure {

AsyncAudioPipeline::AsyncAudioPipeline() : is_running_(false) {}

AsyncAudioPipeline::~AsyncAudioPipeline() {
    Stop();
}

bool AsyncAudioPipeline::Start(const std::string& input_device_id, const std::string& output_device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_running_) return true;
    
    is_running_ = true;
    processing_thread_ = std::thread(&AsyncAudioPipeline::ProcessingLoop, this);
    return true;
}

void AsyncAudioPipeline::Stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!is_running_) return;
        is_running_ = false;
    }
    cv_.notify_all();
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

void AsyncAudioPipeline::SetProcessCallback(std::function<void(const domain::AudioBuffer&, domain::AudioBuffer&)> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = std::move(callback);
}

void AsyncAudioPipeline::PushInput(const domain::AudioBuffer& buffer) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        input_queue_.push_back(buffer);
    }
    cv_.notify_one();
}

bool AsyncAudioPipeline::PopOutput(domain::AudioBuffer& buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (output_queue_.empty()) return false;
    
    buffer = std::move(output_queue_.front());
    output_queue_.erase(output_queue_.begin());
    return true;
}

void AsyncAudioPipeline::ProcessingLoop() {
    while (true) {
        domain::AudioBuffer input;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !is_running_ || !input_queue_.empty(); });
            
            if (!is_running_ && input_queue_.empty()) break;
            
            if (!input_queue_.empty()) {
                input = std::move(input_queue_.front());
                input_queue_.erase(input_queue_.begin());
            }
        }

        if (input.data.empty()) continue;

        domain::AudioBuffer output;
        output.sample_rate = input.sample_rate;
        output.num_channels = input.num_channels;
        output.data.resize(input.data.size(), 0.0f);

        if (callback_) {
            callback_(input, output);
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            output_queue_.push_back(std::move(output));
        }
    }
}

} // namespace sa::infrastructure
