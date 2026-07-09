// Unit tests for domain::AudioMetrics, the signal-metering math that main.cpp
// now feeds into the UI meters (replacing the old hardcoded 0.5 fake levels).
#include <gtest/gtest.h>

#include "silence_arc/domain/audio_metrics.h"

#include <cmath>
#include <vector>

using silence_arc::domain::AudioMetrics;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TEST(AudioMetricsTest, RmsOfConstantEqualsMagnitude) {
    std::vector<float> frame(480, 0.5f);
    EXPECT_NEAR(AudioMetrics::CalculateRMS(frame), 0.5f, 1e-6f);
}

// The regression guard for Fix 1: a quiet 0.2-amplitude frame must read ~0.2,
// NOT the old hardcoded 0.5 that the UI used to display for every frame.
TEST(AudioMetricsTest, RmsReflectsRealSignalNotHardcodedHalf) {
    std::vector<float> quiet(480, 0.2f);
    const float rms = AudioMetrics::CalculateRMS(quiet);
    EXPECT_NEAR(rms, 0.2f, 1e-6f);
    EXPECT_LT(rms, 0.5f);
}

TEST(AudioMetricsTest, RmsOfEmptyFrameIsZero) {
    EXPECT_FLOAT_EQ(AudioMetrics::CalculateRMS(std::vector<float>{}), 0.0f);
}

TEST(AudioMetricsTest, RmsOfFullScaleSineIsRootHalf) {
    const size_t n = 48000;
    std::vector<float> sine(n);
    for (size_t i = 0; i < n; ++i) {
        sine[i] = static_cast<float>(std::sin(2.0 * kPi * 440.0 * static_cast<double>(i) / 48000.0));
    }
    EXPECT_NEAR(AudioMetrics::CalculateRMS(sine), 0.70710678f, 1e-3f);
}

TEST(AudioMetricsTest, DbReductionOfHalvedAmplitudeIsAboutSixDb) {
    std::vector<float> original(480, 0.8f);
    std::vector<float> processed(480, 0.4f); // -6.02 dB
    EXPECT_NEAR(AudioMetrics::CalculateDbReduction(original, processed), 6.0206f, 1e-2f);
}

TEST(AudioMetricsTest, DbReductionOfIdenticalSignalIsZero) {
    std::vector<float> frame(480, 0.5f);
    EXPECT_NEAR(AudioMetrics::CalculateDbReduction(frame, frame), 0.0f, 1e-4f);
}
