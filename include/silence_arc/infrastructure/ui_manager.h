#pragma once

#include "imgui.h"
#include <string>
#include <vector>
#include <memory>

namespace sa::infrastructure {

struct DeviceInfo {
    std::string id;
    std::string name;
};

struct UIState {
    bool noise_suppression_enabled = true;
    float suppression_limit_db = 40.0f;
    
    std::vector<DeviceInfo> input_devices;
    std::vector<DeviceInfo> output_devices;
    int selected_input_device = -1;
    int selected_output_device = -1;
};

struct TelemetryData {
    float gpu_load = 0.0f;
    float vram_usage_mb = 0.0f;
    float processing_latency_ms = 0.0f;
    std::string device_name;
};

class UIManager {
public:
    UIManager();
    ~UIManager();

    bool Init(const std::string& title, int width, int height);
    void Shutdown();

    bool ShouldClose() const;
    void BeginFrame();
    void Render();
    void EndFrame();

    void UpdateTelemetry(const TelemetryData& data);
    void UpdateSignalLevels(float input, float output, float reduction);

    UIState& GetState() { return m_state; }

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    UIState m_state;
};

} // namespace sa::infrastructure
