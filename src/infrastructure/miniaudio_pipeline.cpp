#include "silence_arc/infrastructure/miniaudio_pipeline.h"
#include "miniaudio.h"
#include <iostream>
#include <mutex>
#include <vector>

namespace silence_arc {
namespace infrastructure {

struct MiniaudioPipeline::Impl {
    ma_context context;
    ma_device device;
    bool is_initialized = false;
    bool context_initialized = false;
    domain::IAudioPipeline::ProcessCallback user_callback;
    std::mutex callback_mutex;

    static bool HexToDeviceId(const std::string& hex, ma_device_id& id) {
        if (hex.length() != sizeof(ma_device_id) * 2) return false;
        for (size_t i = 0; i < sizeof(ma_device_id); ++i) {
            std::string byte_str = hex.substr(i * 2, 2);
            ((unsigned char*)&id)[i] = (unsigned char)std::stoul(byte_str, nullptr, 16);
        }
        return true;
    }

    static void DataCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
        auto* impl = static_cast<Impl*>(pDevice->pUserData);
        
        if (!pInput || !pOutput) return;

        std::lock_guard<std::mutex> lock(impl->callback_mutex);
        if (impl->user_callback) {
            domain::AudioBuffer input_buffer;
            input_buffer.sample_rate = pDevice->sampleRate;
            input_buffer.data.assign((const float*)pInput, (const float*)pInput + frameCount); // Mono assumption

            domain::AudioBuffer output_buffer;
            output_buffer.sample_rate = pDevice->sampleRate;
            output_buffer.data.resize(frameCount, 0.0f);

            impl->user_callback(input_buffer, output_buffer);

            // Write back to output
            float* out_ptr = (float*)pOutput;
            for (ma_uint32 i = 0; i < frameCount; ++i) {
                out_ptr[i] = output_buffer.data[i];
            }
        }
    }
};

MiniaudioPipeline::MiniaudioPipeline() : impl_(std::make_unique<Impl>()) {
}

MiniaudioPipeline::~MiniaudioPipeline() {
    Stop();
    if (impl_->context_initialized) {
        ma_context_uninit(&impl_->context);
    }
}

bool MiniaudioPipeline::Start(const std::string& input_device_id, const std::string& output_device_id) {
    if (impl_->is_initialized) {
        Stop();
    }

    if (!impl_->context_initialized) {
        if (ma_context_init(NULL, 0, nullptr, &impl_->context) != MA_SUCCESS) {
            return false;
        }
        impl_->context_initialized = true;
    }

    ma_device_info* pPlaybackInfos;
    ma_uint32 playbackCount;
    ma_device_info* pCaptureInfos;
    ma_uint32 captureCount;

    if (ma_context_get_devices(&impl_->context, &pPlaybackInfos, &playbackCount, &pCaptureInfos, &captureCount) != MA_SUCCESS) {
        return false;
    }

    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_duplex);
    
    ma_device_id input_id, output_id;
    if (!input_device_id.empty() && Impl::HexToDeviceId(input_device_id, input_id)) {
        for (ma_uint32 i = 0; i < captureCount; ++i) {
            if (memcmp(&pCaptureInfos[i].id, &input_id, sizeof(ma_device_id)) == 0) {
                deviceConfig.capture.pDeviceID = &pCaptureInfos[i].id;
                break;
            }
        }
    }

    if (!output_device_id.empty() && Impl::HexToDeviceId(output_device_id, output_id)) {
        for (ma_uint32 i = 0; i < playbackCount; ++i) {
            if (memcmp(&pPlaybackInfos[i].id, &output_id, sizeof(ma_device_id)) == 0) {
                deviceConfig.playback.pDeviceID = &pPlaybackInfos[i].id;
                break;
            }
        }
    }

    deviceConfig.capture.format   = ma_format_f32;
    deviceConfig.capture.channels = 1; // Mono processing required by DFNet
    deviceConfig.capture.shareMode = ma_share_mode_shared;

    deviceConfig.playback.format  = ma_format_f32;
    deviceConfig.playback.channels = 1;

    deviceConfig.sampleRate = 48000;
    deviceConfig.dataCallback = Impl::DataCallback;
    deviceConfig.pUserData = impl_.get();

    if (ma_device_init(&impl_->context, &deviceConfig, &impl_->device) != MA_SUCCESS) {
        std::cerr << "Failed to initialize duplex audio device." << std::endl;
        return false;
    }

    if (ma_device_start(&impl_->device) != MA_SUCCESS) {
        ma_device_uninit(&impl_->device);
        return false;
    }

    impl_->is_initialized = true;
    return true;
}

void MiniaudioPipeline::Stop() {
    if (impl_->is_initialized) {
        ma_device_uninit(&impl_->device);
        impl_->is_initialized = false;
    }
}

bool MiniaudioPipeline::IsRunning() const {
    return impl_->is_initialized;
}

void MiniaudioPipeline::SetProcessCallback(domain::IAudioPipeline::ProcessCallback callback) {
    std::lock_guard<std::mutex> lock(impl_->callback_mutex);
    impl_->user_callback = callback;
}

} // namespace infrastructure
} // namespace silence_arc
