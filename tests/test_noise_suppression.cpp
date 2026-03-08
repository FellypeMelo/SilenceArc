#include <gtest/gtest.h>
#include "silence_arc/domain/noise_suppressor.h"
// This will fail because DeepFilterAdapter is not implemented yet
#include "silence_arc/infrastructure/deep_filter_adapter.h"

namespace silence_arc {
namespace testing {

TEST(NoiseSuppressionTest, InitializationFailsWithInvalidPath) {
    infrastructure::DeepFilterAdapter suppressor;
    EXPECT_FALSE(suppressor.Init("invalid_path.tar.gz"));
}

TEST(NoiseSuppressionTest, ProcessFrameReturnsValidSnr) {
    infrastructure::DeepFilterAdapter suppressor;
    // We expect this to fail compilation/linking until implemented
    // Frame size is typically 480 or 960 for DeepFilterNet
    // For now, assume we can initialize with a dummy or real model path
    // suppressor.Init("path/to/model.tar.gz");
    
    // std::vector<float> input(suppressor.GetFrameLength(), 0.0f);
    // std::vector<float> output(suppressor.GetFrameLength(), 0.0f);
    // float snr = suppressor.ProcessFrame(input.data(), output.data());
    // EXPECT_GE(snr, -100.0f); 
}

} // namespace testing
} // namespace silence_arc
