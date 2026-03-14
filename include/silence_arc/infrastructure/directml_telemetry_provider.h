#pragma once

#include "silence_arc/domain/telemetry_provider.h"
#include "silence_arc/infrastructure/ui_manager.h"
#include <string>
#include <memory>
#include <mutex>

namespace sa::infrastructure {

/**
 * @brief Telemetry provider for DirectML/ONNX Runtime.
 */
class DirectMLTelemetryProvider : public domain::ITelemetryProvider {
public:
    DirectMLTelemetryProvider();
    virtual ~DirectMLTelemetryProvider();

    TelemetryData GetLatestData();
    float get_gpu_load() override;
    float get_vram_usage() override;
    void SetProcessingLatency(float ms);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    float m_latency_ms = 0.0f;
    std::mutex m_mutex;
};

} // namespace sa::infrastructure
