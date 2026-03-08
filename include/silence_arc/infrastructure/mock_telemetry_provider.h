#ifndef SILENCE_ARC_INFRASTRUCTURE_MOCK_TELEMETRY_PROVIDER_H_
#define SILENCE_ARC_INFRASTRUCTURE_MOCK_TELEMETRY_PROVIDER_H_

#include "silence_arc/domain/telemetry_provider.h"
#include <random>

namespace silence_arc {
namespace infrastructure {

class MockTelemetryProvider : public domain::ITelemetryProvider {
public:
    MockTelemetryProvider() : rd_(), gen_(rd_()), dis_(0.0f, 1.0f) {}

    domain::TelemetryData GetLatestData() override {
        return data_;
    }

    void Update() override {
        // Simulate small fluctuations
        data_.gpu_utilization = 0.15f + dis_(gen_) * 0.05f;
        data_.processing_latency_ms = 4.2f + dis_(gen_) * 1.5f;
        data_.memory_footprint_mb = 124.0f + dis_(gen_) * 10.0f;
    }

private:
    domain::TelemetryData data_;
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_real_distribution<float> dis_;
};

} // namespace infrastructure
} // namespace silence_arc

#endif // SILENCE_ARC_INFRASTRUCTURE_MOCK_TELEMETRY_PROVIDER_H_
