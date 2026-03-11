#include <gtest/gtest.h>
#include "silence_arc/infrastructure/sycl_noise_suppressor.h"
#include "silence_arc/infrastructure/wav_loader.h"
#include "silence_arc/domain/audio_metrics.h"
#include <vector>
#include <filesystem>
#include <iostream>

namespace silence_arc {
namespace testing {

using namespace silence_arc::infrastructure;
using namespace silence_arc::domain;

class SyclInferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto path = std::filesystem::current_path();
        if (path.filename() == "build") {
            path = path.parent_path();
        }
        samples_dir = path / "tests" / "samples";
    }

    std::filesystem::path samples_dir;
};

TEST_F(SyclInferenceTest, NativeSyclNoiseReduction) {
    SyclNoiseSuppressor suppressor;
    // Suppressor loads weights internally from models/df3_weights
    ASSERT_TRUE(suppressor.Init(""));

    WavData noisy_data;
    ASSERT_TRUE(WavLoader::Load((samples_dir / "High-Noise.wav").string(), noisy_data));

    size_t frame_size = suppressor.GetFrameLength();
    std::vector<float> processed(noisy_data.samples.size(), 0.0f);

    std::cout << "[INFO] Processing " << noisy_data.samples.size() << " samples via SYCL..." << std::endl;

    for (size_t i = 0; i + frame_size <= noisy_data.samples.size(); i += frame_size) {
        suppressor.ProcessFrame(&noisy_data.samples[i], &processed[i]);
    }

    float db_reduction = AudioMetrics::CalculateDbReduction(noisy_data.samples, processed);
    std::cout << "Measured dB Reduction (SYCL): " << db_reduction << " dB" << std::endl;
    
    // Requirement: > 10dB reduction for the native engine (tuning might be needed for parity)
    EXPECT_GT(db_reduction, 10.0f);

    // Save for manual inspection
    WavData out_data = noisy_data;
    out_data.samples = processed;
    WavWriter::Save((samples_dir / "High-Noise_sycl_processed.wav").string(), out_data);
}

TEST_F(SyclInferenceTest, SyclSignalIntegrity) {
    SyclNoiseSuppressor suppressor;
    ASSERT_TRUE(suppressor.Init(""));

    WavData clean_data;
    ASSERT_TRUE(WavLoader::Load((samples_dir / "No-Noise.wav").string(), clean_data));

    size_t frame_size = suppressor.GetFrameLength();
    std::vector<float> processed(clean_data.samples.size(), 0.0f);

    for (size_t i = 0; i + frame_size <= clean_data.samples.size(); i += frame_size) {
        suppressor.ProcessFrame(&clean_data.samples[i], &processed[i]);
    }

    // RMSE check - native engine should not distort clean voice significantly
    // Note: We might need alignment if there's a fixed delay
    float rmse = AudioMetrics::CalculateRMSE(clean_data.samples, processed);
    std::cout << "Measured RMSE (SYCL No-Noise): " << rmse << std::endl;

    EXPECT_LT(rmse, 0.1f);
}

} // namespace testing
} // namespace silence_arc
