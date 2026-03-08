#ifndef SILENCE_ARC_DOMAIN_UI_STATE_H_
#define SILENCE_ARC_DOMAIN_UI_STATE_H_

#include <string>
#include <vector>

namespace silence_arc {
namespace domain {

struct AudioDevice {
    std::string name;
    std::string id;
};

struct TelemetryData {
    float gpu_utilization = 0.0f;
    float processing_latency_ms = 0.0f;
    float memory_footprint_mb = 0.0f;
};

struct UIState {
    // Toggles
    bool noise_suppression_enabled = false;
    bool voice_enhancement_enabled = false;

    // Monitoring (0.0 to 1.0)
    float input_level = 0.0f;
    float output_level = 0.0f;
    float suppression_depth = 0.0f;
    float db_reduction = 0.0f;

    // Telemetry
    float gpu_utilization = 0.0f;
    float processing_latency_ms = 0.0f;
    float memory_footprint_mb = 0.0f;

    // Configuration
    int selected_audio_api = 0; // 0: WASAPI, 1: ASIO
    int selected_input_device = -1;
    int selected_output_device = -1;
    std::vector<AudioDevice> input_devices;
    std::vector<AudioDevice> output_devices;

    // UI Window state
    float transparency = 1.0f;
    bool is_minimized_to_tray = false;
};

} // namespace domain
} // namespace silence_arc

#endif // SILENCE_ARC_DOMAIN_UI_STATE_H_
