#include <gtest/gtest.h>
#include "silence_arc/infrastructure/wav_loader.h"
#include "silence_arc/infrastructure/deep_filter_adapter.h"
#include "silence_arc/infrastructure/async_audio_pipeline.h"
#include <filesystem>
#include <chrono>
#include <numeric>
#include <cmath>

namespace silence_arc {
namespace testing {

float CalculateRMS(const std::vector<float>& samples) {
    if (samples.empty()) return 0.0f;
    float sum_sq = std::accumulate(samples.begin(), samples.end(), 0.0f, [](float sum, float val) {
        return sum + val * val;
    });
    return std::sqrt(sum_sq / samples.size());
}

TEST(E2ELoopbackTest, NoiseReductionVerification) {
    infrastructure::WavData noise_data;
    auto noise_path = std::filesystem::current_path() / "tests" / "samples" / "High-Noise.wav";
    ASSERT_TRUE(infrastructure::WavLoader::Load(noise_path.string(), noise_data));
    ASSERT_EQ(noise_data.sample_rate, 48000);

    infrastructure::DeepFilterAdapter suppressor;
    auto model_path = std::filesystem::current_path() / "DeepFilterNet" / "models" / "DeepFilterNet3_onnx.tar.gz";
    ASSERT_TRUE(suppressor.Init(model_path.string()));

    infrastructure::AsyncAudioPipeline pipeline;
    std::vector<float> callback_latencies;
    pipeline.SetProcessCallback([&](const domain::AudioBuffer& input, domain::AudioBuffer& output) {
        auto cb_start = std::chrono::high_resolution_clock::now();
        suppressor.ProcessFrame(input.data.data(), output.data.data());
        auto cb_end = std::chrono::high_resolution_clock::now();
        float cb_latency = std::chrono::duration<float, std::milli>(cb_end - cb_start).count();
        callback_latencies.push_back(cb_latency);
    });

    pipeline.Start();

    size_t frame_size = suppressor.GetFrameLength();
    std::cout << "Frame size: " << frame_size << " samples" << std::endl;
    size_t num_frames = 100; // Test with first 100 frames
    
    std::vector<float> original_samples;
    std::vector<float> processed_samples;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_frames; ++i) {
        domain::AudioBuffer input;
        input.data.assign(noise_data.samples.begin() + i * frame_size, 
                          noise_data.samples.begin() + (i + 1) * frame_size);
        original_samples.insert(original_samples.end(), input.data.begin(), input.data.end());
        
        pipeline.PushInput(input);
        
        domain::AudioBuffer output;
        auto wait_start = std::chrono::steady_clock::now();
        while (!pipeline.PopOutput(output) && std::chrono::steady_clock::now() - wait_start < std::chrono::milliseconds(100)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        ASSERT_FALSE(output.data.empty()) << "Pipeline failed to produce output for frame " << i;
        processed_samples.insert(processed_samples.end(), output.data.begin(), output.data.end());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    float avg_latency = static_cast<float>(total_duration) / num_frames;

    float rms_original = CalculateRMS(original_samples);
    float rms_processed = CalculateRMS(processed_samples);

    std::cout << "Original RMS: " << rms_original << std::endl;
    std::cout << "Processed RMS: " << rms_processed << std::endl;
    std::cout << "Avg Latency per frame: " << avg_latency << " ms" << std::endl;

    float total_cb_latency = std::accumulate(callback_latencies.begin(), callback_latencies.end(), 0.0f);
    float avg_cb_latency = total_cb_latency / callback_latencies.size();
    std::cout << "Avg Callback Latency: " << avg_cb_latency << " ms" << std::endl;

    // Verify noise reduction (RMS should be significantly lower)
    EXPECT_LT(rms_processed, rms_original * 0.5f); // At least 50% reduction in amplitude
    
    // Verify latency (processing should be well below 10ms)
    EXPECT_LT(avg_cb_latency, 10.0f);

    pipeline.Stop();
}

} // namespace testing
} // namespace silence_arc
