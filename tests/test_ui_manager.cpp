#include <gtest/gtest.h>
#include "silence_arc/infrastructure/ui_manager.h"
#include <iostream>

using namespace sa::infrastructure;

TEST(UIManagerTest, Initialization) {
    UIManager ui;
    // Basic init doesn't create a real window in tests usually
    EXPECT_TRUE(ui.Init("Test", 100, 100));
}

TEST(UIManagerTest, StateAccess) {
    UIManager ui;
    UIState& state = ui.GetState();
    state.noise_suppression_enabled = false;
    EXPECT_FALSE(ui.GetState().noise_suppression_enabled);
}

TEST(UIManagerTest, TelemetryUpdate) {
    UIManager ui;
    TelemetryData data;
    data.gpu_load = 0.5f;
    data.processing_latency_ms = 10.0f;
    ui.UpdateTelemetry(data);
    // Smoke test for no crash
}
