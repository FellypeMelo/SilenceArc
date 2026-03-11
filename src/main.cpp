#include "silence_arc/infrastructure/ui_manager.h"
#include "silence_arc/infrastructure/deep_filter_adapter.h"
#include "silence_arc/infrastructure/sycl_noise_suppressor.h"
#include "silence_arc/infrastructure/miniaudio_pipeline.h"
#include "silence_arc/infrastructure/miniaudio_device_manager.h"
#include "silence_arc/infrastructure/sycl_accelerator.h"
#include "silence_arc/infrastructure/sycl_telemetry_provider.h"
#include "silence_arc/domain/audio_stream_buffer.h"
#include <iostream>
#include <filesystem>
#include <windows.h>
#include <thread>
#include <chrono>

int main() {
    std::cout << "Starting Silence Arc..." << std::endl;

    bool sycl_available = false;
    // Initialize SYCL Acceleration (Arc GPU)
    if (sycl_init()) {
        char dev_name[256];
        sycl_get_device_name(dev_name, 256);
        std::cout << "[SUCCESS] Hardware Acceleration enabled on: " << dev_name << std::endl;
        sycl_available = true;
    } else {
        std::cout << "[WARN] Hardware Acceleration not available. Using CPU fallback." << std::endl;
    }

    silence_arc::infrastructure::UIManager ui;
    if (!ui.Init("Silence Arc", 400, 600)) {
        std::cerr << "Failed to initialize UI." << std::endl;
        return 1;
    }

    silence_arc::infrastructure::SyclTelemetryProvider telemetry_provider;

    std::unique_ptr<silence_arc::domain::INoiseSuppressor> suppressor;
    
    if (sycl_available) {
        suppressor = std::make_unique<silence_arc::infrastructure::SyclNoiseSuppressor>();
        std::cout << "[INFO] Using Native SYCL Noise Suppressor." << std::endl;
    } else {
        suppressor = std::make_unique<silence_arc::infrastructure::DeepFilterAdapter>();
        std::cout << "[INFO] Using DeepFilterNet CPU Adapter (Rust)." << std::endl;
    }

    auto path = std::filesystem::current_path();
    if (path.filename() == "build") {
        path = path.parent_path();
    }
    auto model_path = path / "DeepFilterNet" / "models" / "DeepFilterNet3_onnx.tar.gz";
    
    if (!suppressor->Init(model_path.string())) {
        std::cerr << "Failed to initialize suppressor implementation." << std::endl;
    }

    silence_arc::infrastructure::MiniaudioDeviceManager::EnumerateDevices(ui.GetState());

    silence_arc::domain::AudioStreamBuffer in_buffer;
    silence_arc::domain::AudioStreamBuffer out_buffer;
    size_t frame_size = suppressor->GetFrameLength();

    silence_arc::infrastructure::MiniaudioPipeline pipeline;
    pipeline.SetProcessCallback([&](const silence_arc::domain::AudioBuffer& input, silence_arc::domain::AudioBuffer& output) {
        auto start_time = std::chrono::steady_clock::now();
        
        in_buffer.Push(input.data.data(), input.data.size());

        while (in_buffer.Available() >= frame_size) {
            std::vector<float> frame_in(frame_size, 0.0f);
            std::vector<float> frame_out(frame_size, 0.0f);
            in_buffer.Pop(frame_in.data(), frame_size);

            if (ui.GetState().noise_suppression_enabled) {
                // Set attention limit from UI
                suppressor->SetAttenuationLimit(ui.GetState().suppression_limit_db);
                suppressor->ProcessFrame(frame_in.data(), frame_out.data());
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
        
        auto end_time = std::chrono::steady_clock::now();
        auto process_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        telemetry_provider.SetProcessingLatency(process_duration.count() / 1000.0f);

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
                
                pipeline.Stop(); // Explicitly stop before starting new devices
                in_buffer.Reset();
                out_buffer.Reset();

                if (pipeline.Start(std::to_string(state.selected_input_device), 
                                   std::to_string(state.selected_output_device))) {
                    current_input_idx = state.selected_input_device;
                    current_output_idx = state.selected_output_device;
                }
            }
        }

        // Update telemetry from live provider
        ui.UpdateTelemetry(telemetry_provider.GetLatestData());

        ui.Render();
        ui.EndFrame();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }

    pipeline.Stop();
    ui.Shutdown();

    return 0;
}
