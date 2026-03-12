#include "silence_arc/domain/audio_stream_buffer.h"
#include <algorithm>
#include <cstring>

namespace sa::domain {

AudioStreamBuffer::AudioStreamBuffer(size_t capacity) 
    : m_buffer(capacity), m_head(0), m_tail(0), m_count(0) {}

void AudioStreamBuffer::Push(const float* data, size_t size) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    for (size_t i = 0; i < size; ++i) {
        m_buffer[m_head] = data[i];
        m_head = (m_head + 1) % m_buffer.size();
        
        if (m_count < m_buffer.size()) {
            m_count++;
        } else {
            // Buffer overflow: move tail forward to overwrite oldest data
            m_tail = (m_tail + 1) % m_buffer.size();
        }
    }
}

size_t AudioStreamBuffer::Pop(float* data, size_t size) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    size_t to_pop = std::min(size, m_count);
    for (size_t i = 0; i < to_pop; ++i) {
        data[i] = m_buffer[m_tail];
        m_tail = (m_tail + 1) % m_buffer.size();
    }
    m_count -= to_pop;
    
    return to_pop;
}

size_t AudioStreamBuffer::Available() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_count;
}

void AudioStreamBuffer::Reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_head = 0;
    m_tail = 0;
    m_count = 0;
}

} // namespace sa::domain
