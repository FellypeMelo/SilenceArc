#include "silence_arc/infrastructure/ui_manager.h"
#include <iostream>
#include <cassert>

using silence_arc::infrastructure::UIManager;

void TestUIInitialization() {
    std::cout << "Running TestUIInitialization..." << std::endl;
    UIManager ui;
    assert(!ui.IsInitialized());
    std::cout << "TestUIInitialization passed (Initial State)." << std::endl;
}

void TestWindowCreation() {
    std::cout << "Running TestWindowCreation..." << std::endl;
    UIManager ui;
    
    // Add dummy devices for UI testing
    ui.GetState().input_devices.push_back({"Mic 1", "id1"});
    ui.GetState().input_devices.push_back({"Mic 2", "id2"});
    ui.GetState().output_devices.push_back({"Speakers", "id3"});
    
    // This will try to create a real window and D3D device.
    // It might fail on headless CI, but on a dev machine with DX11 it should work.
    bool success = ui.Init("Test Window", 800, 600);
    
    if (success) {
        std::cout << "UI Initialized successfully." << std::endl;
        assert(ui.IsInitialized());
        ui.SetTransparency(0.8f);
        ui.Shutdown();
        assert(!ui.IsInitialized());
        std::cout << "TestWindowCreation passed." << std::endl;
    } else {
        std::cout << "UI Initialization failed (expected if no GPU/display)." << std::endl;
    }
}

void TestUIState() {
    std::cout << "Running TestUIState..." << std::endl;
    UIManager ui;
    auto& state = ui.GetState();
    
    assert(!state.noise_suppression_enabled);
    state.noise_suppression_enabled = true;
    assert(ui.GetState().noise_suppression_enabled);
    
    assert(state.input_level == 0.0f);
    state.input_level = 0.5f;
    assert(ui.GetState().input_level == 0.5f);
    
    std::cout << "TestUIState passed." << std::endl;
}

void TestTelemetryUpdate() {
    std::cout << "Running TestTelemetryUpdate..." << std::endl;
    UIManager ui;
    
    silence_arc::domain::TelemetryData data;
    data.gpu_utilization = 0.42f;
    data.processing_latency_ms = 5.5f;
    data.memory_footprint_mb = 256.0f;
    
    ui.UpdateTelemetry(data);
    
    auto& state = ui.GetState();
    assert(state.gpu_utilization == 0.42f);
    assert(state.processing_latency_ms == 5.5f);
    assert(state.memory_footprint_mb == 256.0f);
    
    std::cout << "TestTelemetryUpdate passed." << std::endl;
}

void TestTrayIntegration() {
    std::cout << "Running TestTrayIntegration..." << std::endl;
    UIManager ui;
    
    // We can test the state transitions without a real window if we wanted to,
    // but Init/Shutdown are needed for full tray logic.
    // Let's at least test the public API state.
    bool success = ui.Init("Tray Test", 800, 600);
    if (success) {
        assert(!ui.IsMinimizedToTray());
        ui.ShowWindow(false);
        assert(ui.IsMinimizedToTray());
        ui.ShowWindow(true);
        assert(!ui.IsMinimizedToTray());
        ui.Shutdown();
        std::cout << "TestTrayIntegration passed." << std::endl;
    } else {
        std::cout << "TestTrayIntegration skipped (Initialization failed)." << std::endl;
    }
}

int main() {
    TestUIInitialization();
    TestUIState();
    TestTelemetryUpdate();
    TestTrayIntegration();
    TestWindowCreation();
    std::cout << "All UIManager tests completed!" << std::endl;
    return 0;
}
