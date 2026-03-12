#include "silence_arc/infrastructure/miniaudio_device_manager.h"
#include "miniaudio.h"
#include <iostream>

namespace sa::infrastructure {

void MiniaudioDeviceManager::EnumerateDevices(UIState& state) {
    ma_context context;
    if (ma_context_init(NULL, 0, NULL, &context) != MA_SUCCESS) {
        std::cerr << "[ERROR] Failed to initialize miniaudio context for enumeration." << std::endl;
        return;
    }

    ma_device_info* pCaptureDeviceInfos;
    ma_uint32 captureDeviceCount;
    ma_device_info* pPlaybackDeviceInfos;
    ma_uint32 playbackDeviceCount;

    if (ma_context_get_devices(&context, &pPlaybackDeviceInfos, &playbackDeviceCount, &pCaptureDeviceInfos, &captureDeviceCount) != MA_SUCCESS) {
        std::cerr << "[ERROR] Failed to retrieve audio devices." << std::endl;
        ma_context_uninit(&context);
        return;
    }

    state.input_devices.clear();
    state.output_devices.clear();

    // Enumerate Inputs (Capture)
    for (ma_uint32 i = 0; i < captureDeviceCount; i++) {
        state.input_devices.push_back({
            std::to_string(i), // Use index as ID for now
            pCaptureDeviceInfos[i].name
        });
    }

    // Enumerate Outputs (Playback)
    for (ma_uint32 i = 0; i < playbackDeviceCount; i++) {
        state.output_devices.push_back({
            std::to_string(i),
            pPlaybackDeviceInfos[i].name
        });
    }

    // Auto-select first devices if nothing selected
    if (state.selected_input_device < 0 && !state.input_devices.empty()) state.selected_input_device = 0;
    if (state.selected_output_device < 0 && !state.output_devices.empty()) state.selected_output_device = 0;

    ma_context_uninit(&context);
}

} // namespace sa::infrastructure
