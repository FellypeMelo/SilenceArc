#include <gtest/gtest.h>
#include "silence_arc/infrastructure/sycl_noise_suppressor.h"
#include "silence_arc/infrastructure/wav_loader.h"
#include "silence_arc/domain/audio_metrics.h"
#include <vector>
#include <filesystem>
#include <iostream>
#include <cmath>

namespace silence_arc {
namespace testing {

using namespace silence_arc::infrastructure;
using namespace silence_arc::domain;

class DfIntegrationTest : public ::testing::Test {
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

TEST_F(DfIntegrationTest, DfFilteringImprovesSNR) {
    SyclNoiseSuppressor suppressor;
    ASSERT_TRUE(suppressor.Init(""));
    sycl_reset();

    WavData noisy_data;
    ASSERT_TRUE(WavLoader::Load((samples_dir / "High-Noise.wav").string(), noisy_data));

    size_t frame_size = suppressor.GetFrameLength();
    std::vector<float> processed(noisy_data.samples.size(), 0.0f);

    std::cout << "[INFO] Processing " << noisy_data.samples.size()
              << " samples via SYCL (ERB+DF)..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    for (size_t i = 0; i + frame_size <= noisy_data.samples.size(); i += frame_size) {
        suppressor.ProcessFrame(&noisy_data.samples[i], &processed[i]);
        frame_count++;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    std::cout << "Total frames: " << frame_count << std::endl;
    std::cout << "Avg Latency per frame: " << (duration / frame_count) / 1000.0f << " ms" << std::endl;

    float db_reduction = AudioMetrics::CalculateDbReduction(noisy_data.samples, processed);
    std::cout << "Measured dB Reduction (ERB+DF): " << db_reduction << " dB" << std::endl;

    // DF filtering must improve upon ERB-only baseline (>10 dB)
    // With DF active, expect strictly higher suppression
    EXPECT_GT(db_reduction, 12.0f)
        << "DF filtering should provide >12 dB reduction (ERB-only baseline is ~10 dB)";

    WavData out_data = noisy_data;
    out_data.samples = processed;
    WavWriter::Save((samples_dir / "High-Noise_df_processed.wav").string(), out_data);
}

TEST_F(DfIntegrationTest, DfFilteringPreservesCleanSpeech) {
    SyclNoiseSuppressor suppressor;
    ASSERT_TRUE(suppressor.Init(""));
    sycl_reset();

    WavData clean_data;
    ASSERT_TRUE(WavLoader::Load((samples_dir / "No-Noise.wav").string(), clean_data));

    size_t frame_size = suppressor.GetFrameLength();
    std::vector<float> processed(clean_data.samples.size(), 0.0f);

    for (size_t i = 0; i + frame_size <= clean_data.samples.size(); i += frame_size) {
        suppressor.ProcessFrame(&clean_data.samples[i], &processed[i]);
    }

    float rmse = AudioMetrics::CalculateRMSE(clean_data.samples, processed);
    std::cout << "Measured RMSE (ERB+DF No-Noise): " << rmse << std::endl;

    // DF must not introduce artifacts into clean speech
    EXPECT_LT(rmse, 0.1f)
        << "DF filtering should not distort clean speech beyond RMSE 0.1";
}

TEST_F(DfIntegrationTest, ErbOnlyProvidesNaturalSound) {
    SyclNoiseSuppressor suppressor;
    ASSERT_TRUE(suppressor.Init(""));
    
    // Disable robotic DF path
    suppressor.SetDeepFilteringEnabled(false);

    WavData noisy_data;
    ASSERT_TRUE(WavLoader::Load((samples_dir / "High-Noise.wav").string(), noisy_data));

    size_t frame_size = suppressor.GetFrameLength();
    std::vector<float> processed(noisy_data.samples.size(), 0.0f);

    std::cout << "[INFO] Processing via SYCL (ERB-Only/Natural)..." << std::endl;

    for (size_t i = 0; i + frame_size <= noisy_data.samples.size(); i += frame_size) {
        suppressor.ProcessFrame(&noisy_data.samples[i], &processed[i]);
    }

    float db_reduction = AudioMetrics::CalculateDbReduction(noisy_data.samples, processed);
    std::cout << "Measured dB Reduction (ERB-Only): " << db_reduction << " dB" << std::endl;

    // ERB-only should still provide good reduction (>10 dB)
    EXPECT_GT(db_reduction, 10.0f);

    WavData out_data = noisy_data;
    out_data.samples = processed;
    WavWriter::Save((samples_dir / "High-Noise_erb_only_processed.wav").string(), out_data);
}

} // namespace testing
} // namespace silence_arc
