#include <gtest/gtest.h>
#include "silence_arc/infrastructure/deep_filter_adapter.h"
#include "silence_arc/infrastructure/wav_loader.h"
#include "silence_arc/domain/audio_metrics.h"
#include "utils/audio_gen.h"
#include <vector>
#include <filesystem>
#include <iostream>

namespace silence_arc {
namespace testing {

using namespace silence_arc::infrastructure;
using namespace silence_arc::tests::utils;
using namespace silence_arc::domain;

class E2ESamplesTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto path = std::filesystem::current_path();
        if (path.filename() == "build") {
            path = path.parent_path();
        }
        model_path = (path / "DeepFilterNet" / "models" / "DeepFilterNet3_onnx.tar.gz").string();
        
        sample_rate = 48000;
        duration = 1.0f;
    }

    std::string model_path;
    uint32_t sample_rate;
    float duration;

    size_t GetModelDelay(DeepFilterAdapter& adapter) {
        size_t frame_size = adapter.GetFrameLength();
        std::vector<float> input(frame_size * 10, 0.0f);
        std::vector<float> output(input.size(), 0.0f);
        
        // Put an impulse at the start of frame 2
        input[frame_size + 10] = 1.0f;

        for (size_t i = 0; i + frame_size <= input.size(); i += frame_size) {
            adapter.ProcessFrame(&input[i], &output[i]);
        }

        // Find the peak in output
        size_t peak_pos = 0;
        float max_val = 0.0f;
        for (size_t i = 0; i < output.size(); ++i) {
            if (std::abs(output[i]) > max_val) {
                max_val = std::abs(output[i]);
                peak_pos = i;
            }
        }
        return (peak_pos > (frame_size + 10)) ? (peak_pos - (frame_size + 10)) : 0;
    }
};

TEST_F(E2ESamplesTest, NoiseReductionTest) {
    DeepFilterAdapter adapter;
    ASSERT_TRUE(adapter.Init(model_path));

    auto voice = AudioGen::GenerateSine(440.0f, duration, sample_rate, 0.5f);
    auto noise = AudioGen::GenerateWhiteNoise(duration, sample_rate, 0.2f);
    auto mixed = AudioGen::Mix(voice, noise);

    size_t frame_size = adapter.GetFrameLength();
    std::vector<float> processed(mixed.size(), 0.0f);

    for (size_t i = 0; i + frame_size <= mixed.size(); i += frame_size) {
        adapter.ProcessFrame(&mixed[i], &processed[i]);
    }

    float db_reduction = AudioMetrics::CalculateDbReduction(mixed, processed);
    std::cout << "Measured dB Reduction: " << db_reduction << " dB" << std::endl;
    
    // Requirement: > 20dB reduction
    EXPECT_GT(db_reduction, 20.0f);
}

TEST_F(E2ESamplesTest, SignalIntegrityTest) {
    DeepFilterAdapter adapter;
    ASSERT_TRUE(adapter.Init(model_path));

    auto samples_dir = std::filesystem::current_path();
    if (samples_dir.filename() == "build") {
        samples_dir = samples_dir.parent_path();
    }
    samples_dir = samples_dir / "tests" / "samples";
    
    WavData clean_data;
    ASSERT_TRUE(WavLoader::Load((samples_dir / "No-Noise.wav").string(), clean_data));

    size_t delay = GetModelDelay(adapter);
    std::cout << "Detected model delay: " << delay << " samples" << std::endl;

    size_t frame_size = adapter.GetFrameLength();
    std::vector<float> processed(clean_data.samples.size(), 0.0f);

    for (size_t i = 0; i + frame_size <= clean_data.samples.size(); i += frame_size) {
        adapter.ProcessFrame(&clean_data.samples[i], &processed[i]);
    }

    // Align buffers for RMSE calculation
    std::vector<float> aligned_orig;
    std::vector<float> aligned_proc;
    
    if (clean_data.samples.size() > delay + frame_size * 4) {
        size_t start = frame_size * 4; // Skip transients
        size_t end = clean_data.samples.size() - delay - frame_size;
        for (size_t i = start; i < end; ++i) {
            aligned_orig.push_back(clean_data.samples[i]);
            aligned_proc.push_back(processed[i + delay]);
        }
    }

    float rmse = AudioMetrics::CalculateRMSE(aligned_orig, aligned_proc);
    std::cout << "Measured Aligned RMSE (No-Noise.wav): " << rmse << std::endl;

    // Requirement: RMSE < 0.05
    EXPECT_LT(rmse, 0.05f);
}

} // namespace testing
} // namespace silence_arc
