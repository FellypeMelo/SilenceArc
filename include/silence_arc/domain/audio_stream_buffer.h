#ifndef SILENCE_ARC_DOMAIN_AUDIO_STREAM_BUFFER_H_
#define SILENCE_ARC_DOMAIN_AUDIO_STREAM_BUFFER_H_

#include <vector>
#include <cstdint>

namespace silence_arc {
namespace domain {

class AudioStreamBuffer {
public:
    AudioStreamBuffer() = default;

    void Push(const float* data, size_t size) {
        if (size == 0) return;
        
        // If read_index_ has advanced far enough, we can compact the buffer
        // to avoid infinite growth
        if (read_index_ > 48000) { // e.g. 1 second worth
            buffer_.erase(buffer_.begin(), buffer_.begin() + read_index_);
            read_index_ = 0;
        }

        buffer_.insert(buffer_.end(), data, data + size);
    }

    size_t Available() const {
        if (read_index_ > buffer_.size()) return 0;
        return buffer_.size() - read_index_;
    }

    void Pop(float* out_data, size_t size) {
        size_t avail = Available();
        size_t to_copy = (size < avail) ? size : avail;
        
        if (to_copy > 0) {
            std::copy(buffer_.begin() + read_index_, buffer_.begin() + read_index_ + to_copy, out_data);
            read_index_ += to_copy;
        }
        
        // Zero-fill the rest if requested size > available
        if (size > to_copy) {
            std::fill(out_data + to_copy, out_data + size, 0.0f);
        }
    }

    void Reset() {
        buffer_.clear();
        read_index_ = 0;
    }

private:
    std::vector<float> buffer_;
    size_t read_index_ = 0;
};

} // namespace domain
} // namespace silence_arc

#endif // SILENCE_ARC_DOMAIN_AUDIO_STREAM_BUFFER_H_
