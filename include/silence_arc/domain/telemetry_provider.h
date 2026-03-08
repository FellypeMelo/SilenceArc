#ifndef SILENCE_ARC_DOMAIN_TELEMETRY_PROVIDER_H_
#define SILENCE_ARC_DOMAIN_TELEMETRY_PROVIDER_H_

namespace silence_arc {
namespace domain {

struct TelemetryData {
    float gpu_utilization = 0.0f;
    float processing_latency_ms = 0.0f;
    float memory_footprint_mb = 0.0f;
};

class ITelemetryProvider {
public:
    virtual ~ITelemetryProvider() = default;
    virtual TelemetryData GetLatestData() = 0;
    virtual void Update() = 0;
};

} // namespace domain
} // namespace silence_arc

#endif // SILENCE_ARC_DOMAIN_TELEMETRY_PROVIDER_H_
