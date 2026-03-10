#include <gtest/gtest.h>
#include "silence_arc/infrastructure/sycl_telemetry_provider.h"
#include <thread>
#include <chrono>

namespace silence_arc {
namespace infrastructure {

TEST(SyclTelemetryProviderTest, RetrievesValidData) {
    // This will fail to compile initially because sycl_telemetry_provider.h doesn't exist
    SyclTelemetryProvider provider;
    
    // Initial data should be zeroes or some default
    auto data = provider.GetLatestData();
    
    provider.Update();
    data = provider.GetLatestData();
    
    // On a machine with Intel Arc, these should ideally be non-zero during processing
    // For the test, we just check if the call succeeds and returns a valid structure
    EXPECT_GE(data.gpu_utilization, 0.0f);
    EXPECT_LE(data.gpu_utilization, 1.0f);
    EXPECT_GE(data.memory_footprint_mb, 0.0f);
    EXPECT_GE(data.processing_latency_ms, 0.0f);
}

} // namespace infrastructure
} // namespace silence_arc
