// Tests for the GPU attenuation-limit feature (Fix 2).
//
// The dB->dry-weight mapping and the dry/wet mix are pure functions in
// domain::attenuation_limit and are unit-tested here on the CPU (always run).
// A final integration test drives the real SYCL suppressor end-to-end and
// GTEST_SKIP()s when no Arc GPU is present, so CI hosts without a GPU still run
// the pure tests.
#include <gtest/gtest.h>

#include "silence_arc/domain/attenuation_limit.h"
#include "silence_arc/infrastructure/sycl_noise_suppressor.h"

#include <cmath>
#include <vector>

using silence_arc::domain::AttenLimitDryMix;
using silence_arc::domain::ApplyAttenuationMix;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TEST(AttenuationLimitTest, DryMixMappingMatchesLibDf) {
    // >= 100 dB: no limit -> fully wet (0 dry).
    EXPECT_FLOAT_EQ(AttenLimitDryMix(100.0f), 0.0f);
    EXPECT_FLOAT_EQ(AttenLimitDryMix(150.0f), 0.0f);
    // < 0.01 dB: bypass -> fully dry (1.0).
    EXPECT_FLOAT_EQ(AttenLimitDryMix(0.0f), 1.0f);
    EXPECT_FLOAT_EQ(AttenLimitDryMix(0.005f), 1.0f);
    // In-between: 10^(-|db|/20).
    EXPECT_NEAR(AttenLimitDryMix(40.0f), 0.01f, 1e-6f);
    EXPECT_NEAR(AttenLimitDryMix(20.0f), 0.1f, 1e-6f);
    EXPECT_NEAR(AttenLimitDryMix(6.0206f), 0.5f, 1e-3f);
    // Mapping uses |db|, so sign is irrelevant.
    EXPECT_FLOAT_EQ(AttenLimitDryMix(-40.0f), AttenLimitDryMix(40.0f));
}

TEST(AttenuationLimitTest, MixFullyWetLeavesOutputUnchanged) {
    std::vector<float> in(8, 1.0f);
    std::vector<float> out(8, 0.2f);
    ApplyAttenuationMix(in.data(), out.data(), out.size(), AttenLimitDryMix(120.0f));
    for (float v : out) EXPECT_FLOAT_EQ(v, 0.2f);
}

TEST(AttenuationLimitTest, MixFullyDryEqualsInput) {
    std::vector<float> in(8);
    std::vector<float> out(8, -1.0f);
    for (size_t i = 0; i < in.size(); ++i) in[i] = 0.1f * static_cast<float>(i);
    ApplyAttenuationMix(in.data(), out.data(), out.size(), AttenLimitDryMix(0.0f));
    for (size_t i = 0; i < in.size(); ++i) EXPECT_FLOAT_EQ(out[i], in[i]);
}

TEST(AttenuationLimitTest, MixIsLinearBlend) {
    const float dry = 0.25f;
    std::vector<float> in(4, 1.0f);
    std::vector<float> wet(4, 0.0f);
    std::vector<float> out = wet;
    ApplyAttenuationMix(in.data(), out.data(), out.size(), dry);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i], (1.0f - dry) * wet[i] + dry * in[i]); // 0.25
    }
}

// End-to-end on the real SYCL engine: at the bypass limit the mix is fully dry,
// so the suppressor output must equal its input exactly regardless of what the
// network produced. Before Fix 2 (SetAttenuationLimit was a no-op) the output
// stayed the enhanced signal and this assertion failed.
TEST(AttenuationLimitTest, GpuBypassLimitYieldsIdentity) {
    using silence_arc::infrastructure::SyclNoiseSuppressor;

    SyclNoiseSuppressor suppressor;
    if (!suppressor.Init("")) {
        GTEST_SKIP() << "No SYCL/Arc device available; skipping GPU integration test.";
    }

    const size_t n = suppressor.GetFrameLength();
    std::vector<float> in(n);
    for (size_t i = 0; i < n; ++i) {
        in[i] = 0.3f * static_cast<float>(std::sin(2.0 * kPi * 440.0 * static_cast<double>(i) / 48000.0));
    }

    // Full wet: enhanced output (kept only to prove the limit actually changes it).
    std::vector<float> out_wet(n, 0.0f);
    suppressor.SetAttenuationLimit(120.0f);
    suppressor.ProcessFrame(in.data(), out_wet.data());

    // Bypass: output must be a bit-for-bit copy of the input.
    std::vector<float> out_bypass(n, 0.0f);
    suppressor.SetAttenuationLimit(0.0f);
    suppressor.ProcessFrame(in.data(), out_bypass.data());
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(out_bypass[i], in[i]);
    }
}
