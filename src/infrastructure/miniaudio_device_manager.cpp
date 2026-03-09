#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include "silence_arc/infrastructure/miniaudio_device_manager.h"
#include <iostream>

namespace silence_arc {
namespace infrastructure {

bool MiniaudioDeviceManager::EnumerateDevices(domain::UIState& state) {
    ma_context context;
    if (ma_context_init(NULL, 0, nullptr, &context) != MA_SUCCESS) {
        return false;
    }

    ma_device_info* pPlaybackInfos;
    ma_uint32 playbackCount;
    ma_device_info* pCaptureInfos;
    ma_uint32 captureCount;

    if (ma_context_get_devices(&context, &pPlaybackInfos, &playbackCount, &pCaptureInfos, &captureCount) != MA_SUCCESS) {
        ma_context_uninit(&context);
        return false;
    }

    state.output_devices.clear();
    for (ma_uint32 i = 0; i < playbackCount; i++) {
        domain::AudioDevice dev;
        dev.name = pPlaybackInfos[i].name;
        // Using index as ID for simplicity
        dev.id = std::to_string(i);
        state.output_devices.push_back(dev);
    }

    state.input_devices.clear();
    for (ma_uint32 i = 0; i < captureCount; i++) {
        domain::AudioDevice dev;
        dev.name = pCaptureInfos[i].name;
        // Using index as ID for simplicity
        dev.id = std::to_string(i);
        state.input_devices.push_back(dev);
    }

    if (state.selected_input_device == -1 && state.input_devices.size() > 0) state.selected_input_device = 0;
    if (state.selected_output_device == -1 && state.output_devices.size() > 0) state.selected_output_device = 0;

    ma_context_uninit(&context);
    return true;
}

} // namespace infrastructure
} // namespace silence_arc
