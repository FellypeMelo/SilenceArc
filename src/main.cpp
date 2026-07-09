#include "silence_arc/infrastructure/ui_manager.h"
#include "silence_arc/infrastructure/deep_filter_adapter.h"
#include "silence_arc/infrastructure/sycl_noise_suppressor.h"
#include "silence_arc/infrastructure/miniaudio_pipeline.h"
#include "silence_arc/infrastructure/async_audio_pipeline.h"
#include "silence_arc/infrastructure/miniaudio_device_manager.h"
#include "silence_arc/infrastructure/sycl_accelerator.h"
#include "silence_arc/infrastructure/sycl_telemetry_provider.h"
#include "silence_arc/domain/audio_stream_buffer.h"
#include <iostream>
#include <filesystem>
#include <windows.h>
#include <thread>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

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

    // The heavy per-frame work (GPU/NN inference) runs on this dedicated
    // TIME_CRITICAL worker thread, NOT on the real-time audio device callback.
    // Its lifetime spans all of main(), independent of device Start/Stop cycles.
    silence_arc::infrastructure::AsyncAudioPipeline async;
    async.SetProcessCallback([&](const silence_arc::domain::AudioBuffer& frame_in,
                                 silence_arc::domain::AudioBuffer& frame_out) {
        auto start_time = std::chrono::steady_clock::now();

        frame_out.data.resize(frame_in.data.size());
        // enable/limit are written by the UI thread; a torn read of a bool/float
        // is benign here (worst case one frame uses the previous setting).
        if (ui.GetState().noise_suppression_enabled) {
            suppressor->SetAttenuationLimit(ui.GetState().suppression_limit_db);
            suppressor->ProcessFrame(frame_in.data.data(), frame_out.data.data());
        } else {
            frame_out.data = frame_in.data; // Pass-through
        }

        auto end_time = std::chrono::steady_clock::now();
        auto process_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        telemetry_provider.SetProcessingLatency(process_duration.count() / 1000.0f);

        // Real signal metering, computed here on the worker thread (never on the
        // RT device callback). RMS of f32 samples in [-1,1] already maps to the
        // 0..1 range the ProgressBar meters expect.
        auto rms = [](const std::vector<float>& d) -> float {
            if (d.empty()) return 0.0f;
            double acc = 0.0;
            for (float s : d) acc += static_cast<double>(s) * s;
            return static_cast<float>(std::sqrt(acc / static_cast<double>(d.size())));
        };
        const float in_level = std::min(rms(frame_in.data), 1.0f);
        const float out_level = std::min(rms(frame_out.data), 1.0f);
        float reduction_db = 0.0f;
        if (ui.GetState().noise_suppression_enabled && out_level > 1e-6f && in_level > out_level) {
            reduction_db = 20.0f * std::log10(in_level / out_level);
        }
        ui.UpdateSignalLevels(in_level, out_level, reduction_db);
    });
    async.Start();

    silence_arc::infrastructure::MiniaudioPipeline pipeline;
    // The device callback is now a thin, non-blocking shim: frame the input, hand
    // whole frames to the async worker, and drain whatever the worker has already
    // finished back to the device. Never calls ProcessFrame on the audio thread.
    pipeline.SetProcessCallback([&](const silence_arc::domain::AudioBuffer& input, silence_arc::domain::AudioBuffer& output) {
        in_buffer.Push(input.data.data(), input.data.size());

        while (in_buffer.Available() >= frame_size) {
            silence_arc::domain::AudioBuffer frame;
            frame.data.resize(frame_size);
            in_buffer.Pop(frame.data.data(), frame_size);
            async.PushInput(frame);
        }

        // Collect frames the worker finished (from earlier callbacks).
        silence_arc::domain::AudioBuffer done;
        while (async.PopOutput(done)) {
            out_buffer.Push(done.data.data(), done.data.size());
        }

        // Emit exactly what miniaudio requested; short-fall stays zero-filled.
        size_t requested_size = output.data.size();
        size_t available_out = out_buffer.Available();
        size_t push_size = (requested_size < available_out) ? requested_size : available_out;
        if (push_size > 0) {
            out_buffer.Pop(output.data.data(), push_size);
        }
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
        // Surface the async pipeline's bounded-queue drop counter in the UI.
        ui.GetState().frames_dropped = async.FramesDropped();

        ui.Render();
        ui.EndFrame();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }

    pipeline.Stop();
    async.Stop();
    ui.Shutdown();

    return 0;
}
