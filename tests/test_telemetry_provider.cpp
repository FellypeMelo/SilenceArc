#include <gtest/gtest.h>
#include "silence_arc/infrastructure/sycl_telemetry_provider.h"
#include <iostream>

using namespace sa::infrastructure;

TEST(TelemetryProviderTest, BasicQuery) {
    SyclTelemetryProvider provider;
    
    // Test that we can get data without crash
    TelemetryData data = provider.GetLatestData();
    
    EXPECT_GE(data.gpu_load, 0.0f);
    EXPECT_GE(data.vram_usage_mb, 0.0f);
}
