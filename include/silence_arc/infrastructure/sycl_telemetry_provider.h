#ifndef SILENCE_ARC_INFRASTRUCTURE_SYCL_TELEMETRY_PROVIDER_H_
#define SILENCE_ARC_INFRASTRUCTURE_SYCL_TELEMETRY_PROVIDER_H_

#include "silence_arc/domain/telemetry_provider.h"
#include <sycl/sycl.hpp>
#include <level_zero/zes_api.h>
#include <memory>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

namespace silence_arc {
namespace infrastructure {

class SyclTelemetryProvider : public domain::ITelemetryProvider {
public:
    SyclTelemetryProvider();
    ~SyclTelemetryProvider() override;

    domain::TelemetryData GetLatestData() override;
    void Update() override;

    // Internal method to update latency from the engine
    void SetProcessingLatency(float latency_ms);

private:
    domain::TelemetryData data_;
    std::mutex data_mutex_;
    std::atomic<float> external_latency_ms_{0.0f};
    
    zes_device_handle_t hSysmanDevice = nullptr;
    zes_engine_handle_t hEngineAll = nullptr;
    zes_mem_handle_t hMainMemory = nullptr;
    
    zes_engine_stats_t last_engine_stats_ = {0};
    uint64_t last_timestamp_ = 0;
    bool sysman_initialized_ = false;

    std::thread worker_thread_;
    std::atomic<bool> run_polling_{false};

    void InitializeSysman();
    void PollingLoop();
};

} // namespace infrastructure
} // namespace silence_arc

#endif // SILENCE_ARC_INFRASTRUCTURE_SYCL_TELEMETRY_PROVIDER_H_
