#include "silence_arc/infrastructure/ui_manager.h"
#include "silence_arc/infrastructure/deep_filter_adapter.h"
#include "silence_arc/infrastructure/miniaudio_pipeline.h"
#include "silence_arc/infrastructure/miniaudio_device_manager.h"
#include "silence_arc/domain/audio_stream_buffer.h"
#include <iostream>
#include <filesystem>
#include <windows.h>
#include <thread>
#include <chrono>

int main() {
    std::cout << "Starting Silence Arc..." << std::endl;

    silence_arc::infrastructure::UIManager ui;
    if (!ui.Init("Silence Arc", 400, 600)) {
        std::cerr << "Failed to initialize UI." << std::endl;
        return 1;
    }

    silence_arc::infrastructure::DeepFilterAdapter suppressor;
    auto path = std::filesystem::current_path();
    if (path.filename() == "build") {
        path = path.parent_path();
    }
    auto model_path = path / "DeepFilterNet" / "models" / "DeepFilterNet3_onnx.tar.gz";
    
    if (!suppressor.Init(model_path.string())) {
        std::cerr << "Failed to initialize DeepFilterNet model." << std::endl;
    }

    silence_arc::infrastructure::MiniaudioDeviceManager::EnumerateDevices(ui.GetState());

    silence_arc::domain::AudioStreamBuffer in_buffer;
    silence_arc::domain::AudioStreamBuffer out_buffer;
    size_t frame_size = suppressor.GetFrameLength();

    silence_arc::infrastructure::MiniaudioPipeline pipeline;
    pipeline.SetProcessCallback([&](const silence_arc::domain::AudioBuffer& input, silence_arc::domain::AudioBuffer& output) {
        in_buffer.Push(input.data.data(), input.data.size());

        while (in_buffer.Available() >= frame_size) {
            std::vector<float> frame_in(frame_size, 0.0f);
            std::vector<float> frame_out(frame_size, 0.0f);
            in_buffer.Pop(frame_in.data(), frame_size);

            if (ui.GetState().noise_suppression_enabled) {
                // Set attention limit from UI
                suppressor.SetAttenuationLimit(ui.GetState().suppression_limit_db);
                suppressor.ProcessFrame(frame_in.data(), frame_out.data());
            } else {
                frame_out = frame_in; // Pass-through
            }
            out_buffer.Push(frame_out.data(), frame_size);
        }

        // Output exactly what miniaudio requested to prevent dropouts/desync
        size_t requested_size = output.data.size();
        size_t available_out = out_buffer.Available();
        size_t push_size = (requested_size < available_out) ? requested_size : available_out;
        
        if (push_size > 0) {
            out_buffer.Pop(output.data.data(), push_size);
        }
        
        // If we don't have enough, the rest of output.data is already 0.0 from initialization
        
        // Mock signal levels for now
        ui.UpdateSignalLevels(0.5f, 0.5f, ui.GetState().noise_suppression_enabled ? 10.0f : 0.0f);
    });

    // Initial signal level update
    ui.UpdateSignalLevels(0.0f, 0.0f, 0.0f);

    int current_input_idx = -1;
    int current_output_idx = -1;

    // Main loop
    while (!ui.ShouldClose()) {
        ui.BeginFrame();
        
        auto& state = ui.GetState();

        // Detect and handle device changes
        if (state.selected_input_device != current_input_idx || 
            state.selected_output_device != current_output_idx) {
            
            if (state.selected_input_device >= 0 && state.selected_output_device >= 0) {
                std::cout << "Audio device change detected. Selected Input: " 
                          << state.input_devices[state.selected_input_device].name 
                          << ", Output: " << state.output_devices[state.selected_output_device].name << std::endl;
                
                pipeline.Start(std::to_string(state.selected_input_device), 
                               std::to_string(state.selected_output_device));
                
                current_input_idx = state.selected_input_device;
                current_output_idx = state.selected_output_device;
            }
        }

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
