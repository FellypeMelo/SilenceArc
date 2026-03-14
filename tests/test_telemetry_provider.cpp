#include <gtest/gtest.h>
#include "silence_arc/infrastructure/directml_telemetry_provider.h"

using namespace sa::infrastructure;

TEST(TelemetryProviderTest, BasicQuery) {
    DirectMLTelemetryProvider provider;
    auto data = provider.GetLatestData();
    
    // In stub mode, we expect 0 or static data
    EXPECT_EQ(data.gpu_load, 0.0f);
    EXPECT_EQ(data.vram_usage_mb, 0.0f);
    EXPECT_TRUE(data.device_name.find("DirectML") != std::string::npos);
}
