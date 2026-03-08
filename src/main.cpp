#include "silence_arc/infrastructure/ui_manager.h"
#include "silence_arc/infrastructure/deep_filter_adapter.h"
#include "silence_arc/infrastructure/async_audio_pipeline.h"
#include <iostream>
#include <filesystem>
#include <windows.h>

int main() {
    std::cout << "Starting Silence Arc..." << std::endl;

    silence_arc::infrastructure::UIManager ui;
    if (!ui.Init("Silence Arc", 400, 600)) {
        std::cerr << "Failed to initialize UI." << std::endl;
        return 1;
    }

    silence_arc::infrastructure::DeepFilterAdapter suppressor;
    auto model_path = std::filesystem::current_path() / "DeepFilterNet" / "models" / "DeepFilterNet3_onnx.tar.gz";
    
    if (!suppressor.Init(model_path.string())) {
        std::cerr << "Failed to initialize DeepFilterNet model." << std::endl;
        // We can still continue, maybe the user can select a model later
    }

    silence_arc::infrastructure::AsyncAudioPipeline pipeline;
    pipeline.SetProcessCallback([&](const silence_arc::domain::AudioBuffer& input, silence_arc::domain::AudioBuffer& output) {
        if (ui.GetState().noise_suppression_enabled) {
            float snr = suppressor.ProcessFrame(input.data.data(), output.data.data());
            ui.UpdateSignalLevels(0.5f, 0.5f, 10.0f); // Mock signal levels for now
        } else {
            output.data = input.data; // Pass-through
            ui.UpdateSignalLevels(0.5f, 0.5f, 0.0f);
        }
    });

    pipeline.Start();

    // Main loop
    while (!ui.ShouldClose()) {
        ui.BeginFrame();
        
        // Update telemetry (Mock)
        silence_arc::domain::TelemetryData telemetry;
        telemetry.gpu_utilization = 0.15f;
        telemetry.processing_latency_ms = 1.5f;
        telemetry.memory_footprint_mb = 120.0f;
        ui.UpdateTelemetry(telemetry);

        ui.Render();
        ui.EndFrame();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }

    pipeline.Stop();
    ui.Shutdown();

    return 0;
}
