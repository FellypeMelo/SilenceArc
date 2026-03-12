#include "silence_arc/infrastructure/ui_manager.h"
#include "silence_arc/infrastructure/deep_filter_adapter.h"
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

using namespace sa;

int main() {
    std::cout << "Starting Silence Arc (v2 Modular)..." << std::endl;

    // Initialize SYCL Acceleration (Arc GPU) for Telemetry
    if (sycl_init()) {
        char dev_name[256];
        sycl_get_device_name(dev_name, 256);
        std::cout << "[SUCCESS] Hardware Acceleration enabled on: " << dev_name << std::endl;
    } else {
        std::cout << "[WARN] Hardware Acceleration not available. Telemetry may be limited." << std::endl;
    }

    infrastructure::UIManager ui;
    if (!ui.Init("Silence Arc", 400, 600)) {
        std::cerr << "Failed to initialize UI." << std::endl;
        return 1;
    }

    infrastructure::SyclTelemetryProvider telemetry_provider;

    // Resolve model path
    auto path = std::filesystem::current_path();
    if (path.filename() == "build") path = path.parent_path();
    auto model_path = path / "DeepFilterNet" / "models" / "DeepFilterNet3_onnx.tar.gz";

    // Use stable Rust DeepFilterAdapter via IAudioProcessor interface
    std::unique_ptr<domain::IAudioProcessor> processor = std::make_unique<infrastructure::DeepFilterAdapter>(model_path.string());
    
    if (!processor->initialize()) {
        std::cerr << "[ERROR] Failed to initialize audio processor." << std::endl;
    } else {
        std::cout << "[INFO] Audio Processor initialized: " << processor->get_device_name() << std::endl;
    }

    infrastructure::MiniaudioDeviceManager::EnumerateDevices(ui.GetState());

    domain::AudioStreamBuffer in_buffer;
    domain::AudioStreamBuffer out_buffer;
    size_t frame_size = processor->get_frame_size();

    infrastructure::MiniaudioPipeline pipeline;
    pipeline.SetProcessCallback([&](const domain::AudioBuffer& input, domain::AudioBuffer& output) {
        auto start_time = std::chrono::steady_clock::now();
        
        in_buffer.Push(input.data.data(), input.data.size());

        while (in_buffer.Available() >= frame_size) {
            std::vector<float> frame_in(frame_size, 0.0f);
            std::vector<float> frame_out(frame_size, 0.0f);
            in_buffer.Pop(frame_in.data(), frame_size);

            if (ui.GetState().noise_suppression_enabled) {
                processor->set_attenuation_limit(ui.GetState().suppression_limit_db);
                processor->process_frame(frame_in.data(), frame_out.data(), frame_size);
            } else {
                frame_out = frame_in; // Pass-through
            }
            out_buffer.Push(frame_out.data(), frame_size);
        }

        size_t requested_size = output.data.size();
        size_t available_out = out_buffer.Available();
        size_t push_size = (requested_size < available_out) ? requested_size : available_out;
        
        if (push_size > 0) {
            out_buffer.Pop(output.data.data(), push_size);
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto process_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        telemetry_provider.SetProcessingLatency(process_duration.count() / 1000.0f);

        ui.UpdateSignalLevels(0.5f, 0.5f, ui.GetState().noise_suppression_enabled ? 10.0f : 0.0f);
    });

    int current_input_idx = -1;
    int current_output_idx = -1;

    // Main loop
    while (!ui.ShouldClose()) {
        ui.BeginFrame();
        auto& state = ui.GetState();

        if (state.selected_input_device != current_input_idx || state.selected_output_device != current_output_idx) {
            if (state.selected_input_device >= 0 && state.selected_output_device >= 0) {
                pipeline.Stop();
                in_buffer.Reset();
                out_buffer.Reset();
                processor->reset();

                if (pipeline.Start(std::to_string(state.selected_input_device), std::to_string(state.selected_output_device))) {
                    current_input_idx = state.selected_input_device;
                    current_output_idx = state.selected_output_device;
                }
            }
        }

        ui.UpdateTelemetry(telemetry_provider.GetLatestData());
        ui.Render();
        ui.EndFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    pipeline.Stop();
    ui.Shutdown();
    return 0;
}
