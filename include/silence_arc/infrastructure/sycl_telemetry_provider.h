#pragma once

#include "silence_arc/domain/telemetry_provider.h"
#include "silence_arc/infrastructure/ui_manager.h"
#include <string>
#include <memory>
#include <mutex>

namespace sa::infrastructure {

/**
 * @brief Provides real-time GPU telemetry using Intel Level Zero Sysman.
 */
class SyclTelemetryProvider : public domain::ITelemetryProvider {
public:
    SyclTelemetryProvider();
    ~SyclTelemetryProvider() override;

    TelemetryData GetLatestData();
    
    // Domain Interface
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
