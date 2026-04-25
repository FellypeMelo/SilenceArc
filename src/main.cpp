#include "silence_arc/infrastructure/ui_manager.h"
#include "silence_arc/infrastructure/directml_audio_engine.h"
#include "silence_arc/infrastructure/miniaudio_pipeline.h"
#include "silence_arc/infrastructure/miniaudio_device_manager.h"
#include "silence_arc/infrastructure/directml_telemetry_provider.h"
#include "silence_arc/domain/audio_stream_buffer.h"
#include <iostream>
#include <filesystem>
#include <windows.h>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <atomic>

using namespace sa;

int main() {
    std::cout << "Starting Silence Arc (DirectML Edition)..." << std::endl;

    infrastructure::UIManager ui;
    if (!ui.Init("Silence Arc", 400, 600)) return 1;

    infrastructure::DirectMLTelemetryProvider telemetry_provider;
    std::unique_ptr<domain::IAudioProcessor> processor(new infrastructure::directml_impl::DirectMLAudioEngine());
    
    if (!processor->initialize()) {
        std::cerr << "[ERROR] Failed to initialize Audio Processor." << std::endl;
    }

    infrastructure::MiniaudioDeviceManager::EnumerateDevices(ui.GetState());

    domain::AudioStreamBuffer in_buffer;
    domain::AudioStreamBuffer out_buffer;
    size_t frame_size = processor->get_frame_size();
    
    std::atomic<bool> processing_running{true};
    std::mutex proc_mtx;
    std::condition_variable proc_cv;

    // Dedicated processing thread with INSTANT wake-up
    std::thread processing_thread([&]() {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        while (processing_running) {
            std::unique_lock<std::mutex> lock(proc_mtx);
            proc_cv.wait(lock, [&] { return !processing_running || in_buffer.Available() >= frame_size; });
            
            if (!processing_running) break;

            auto start_time = std::chrono::steady_clock::now();
            std::vector<float> frame_in(frame_size, 0.0f);
            std::vector<float> frame_out(frame_size, 0.0f);
            
            in_buffer.Pop(frame_in.data(), frame_size);
            lock.unlock(); // Release lock while GPU is working

            if (ui.GetState().noise_suppression_enabled) {
                processor->set_attenuation_limit(ui.GetState().suppression_limit_db);
                processor->process_frame(frame_in.data(), frame_out.data(), frame_size);
            } else {
                frame_out = frame_in;
            }

            out_buffer.Push(frame_out.data(), frame_size);

            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            telemetry_provider.SetProcessingLatency(duration.count() / 1000.0f);
        }
    });

    infrastructure::MiniaudioPipeline pipeline;
    pipeline.SetProcessCallback([&](const domain::AudioBuffer& input, domain::AudioBuffer& output) {
        in_buffer.Push(input.data.data(), input.data.size());
        proc_cv.notify_one(); // Wake up AI thread immediately

        size_t requested = output.data.size();
        size_t available = out_buffer.Available();
        size_t pop_size = (requested < available) ? requested : available;

        if (pop_size > 0) {
            out_buffer.Pop(output.data.data(), pop_size);
        }
        if (pop_size < requested) {
            std::fill(output.data.begin() + pop_size, output.data.end(), 0.0f);
        }
        ui.UpdateSignalLevels(0.5f, 0.5f, ui.GetState().noise_suppression_enabled ? 10.0f : 0.0f);
    });

    int cur_in = -1, cur_out = -1;
    while (!ui.ShouldClose()) {
        ui.BeginFrame();
        auto& state = ui.GetState();
        if (state.selected_input_device != cur_in || state.selected_output_device != cur_out) {
            if (state.selected_input_device >= 0 && state.selected_output_device >= 0) {
                pipeline.Stop();
                in_buffer.Reset();
                out_buffer.Reset();
                processor->reset();
                if (pipeline.Start(std::to_string(state.selected_input_device), std::to_string(state.selected_output_device))) {
                    cur_in = state.selected_input_device;
                    cur_out = state.selected_output_device;
                }
            }
        }
        ui.UpdateTelemetry(telemetry_provider.GetLatestData());
        ui.Render();
        ui.EndFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    processing_running = false;
    proc_cv.notify_all();
    if (processing_thread.joinable()) processing_thread.join();
    pipeline.Stop();
    ui.Shutdown();
    return 0;
}
