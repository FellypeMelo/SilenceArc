#include "silence_arc/infrastructure/miniaudio_pipeline.h"
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include <iostream>
#include <algorithm>

namespace sa::infrastructure {

struct MiniaudioPipeline::Impl {
    ma_device device;
    bool is_running = false;
    ProcessCallback callback;
};

static void ma_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    auto impl = reinterpret_cast<MiniaudioPipeline::Impl*>(pDevice->pUserData);
    if (!impl || !impl->callback) return;

    domain::AudioBuffer in_buf;
    in_buf.data.assign(static_cast<const float*>(pInput), static_cast<const float*>(pInput) + (frameCount * pDevice->capture.channels));
    in_buf.sample_rate = pDevice->sampleRate;
    in_buf.num_channels = pDevice->capture.channels;

    domain::AudioBuffer out_buf;
    out_buf.data.resize(frameCount * pDevice->playback.channels, 0.0f);
    out_buf.sample_rate = pDevice->sampleRate;
    out_buf.num_channels = pDevice->playback.channels;

    impl->callback(in_buf, out_buf);

    std::copy(out_buf.data.begin(), out_buf.data.end(), static_cast<float*>(pOutput));
}

MiniaudioPipeline::MiniaudioPipeline() : m_impl(std::make_unique<Impl>()) {}

MiniaudioPipeline::~MiniaudioPipeline() {
    Stop();
}

bool MiniaudioPipeline::Start(const std::string& input_device_id, const std::string& output_device_id) {
    if (m_impl->is_running) Stop();

    ma_device_config config = ma_device_config_init(ma_device_type_duplex);
    config.playback.format   = ma_format_f32;
    config.playback.channels = 1; 
    config.capture.format    = ma_format_f32;
    config.capture.channels  = 1;
    config.sampleRate        = 48000;
    config.dataCallback      = ma_callback;
    config.pUserData         = m_impl.get();
    
    // Increase buffer size to 100ms for extreme stability
    config.periodSizeInFrames = 480; // 10ms per period
    config.periods = 10;            // 10 periods = 100ms total hardware buffer
    if (ma_device_init(NULL, &config, &m_impl->device) != MA_SUCCESS) {
        return false;
    }

    if (ma_device_start(&m_impl->device) != MA_SUCCESS) {
        ma_device_uninit(&m_impl->device);
        return false;
    }

    m_impl->is_running = true;
    return true;
}

void MiniaudioPipeline::Stop() {
    if (m_impl->is_running) {
        ma_device_stop(&m_impl->device);
        ma_device_uninit(&m_impl->device);
        m_impl->is_running = false;
    }
}

bool MiniaudioPipeline::IsRunning() const {
    return m_impl->is_running;
}

void MiniaudioPipeline::SetProcessCallback(ProcessCallback callback) {
    m_impl->callback = std::move(callback);
}

} // namespace sa::infrastructure
