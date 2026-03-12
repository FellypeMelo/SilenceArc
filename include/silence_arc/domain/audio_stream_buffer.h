#pragma once

#include <vector>
#include <mutex>
#include <cstddef>

namespace sa::domain {

/**
 * @brief Circular buffer for thread-safe audio streaming between devices and processors.
 */
class AudioStreamBuffer {
public:
    explicit AudioStreamBuffer(size_t capacity = 192000); // ~4 seconds at 48kHz

    void Push(const float* data, size_t size);
    size_t Pop(float* data, size_t size);
    
    size_t Available() const;
    void Reset();

private:
    std::vector<float> m_buffer;
    size_t m_head = 0;
    size_t m_tail = 0;
    size_t m_count = 0;
    mutable std::mutex m_mutex;
};

} // namespace sa::domain
