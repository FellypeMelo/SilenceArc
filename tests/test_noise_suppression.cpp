#include <gtest/gtest.h>
#include "silence_arc/domain/noise_suppressor.h"
#include "silence_arc/infrastructure/deep_filter_adapter.h"
#include <vector>
#include <filesystem>

namespace silence_arc {
namespace testing {

std::string GetModelPath() {
    auto path = std::filesystem::current_path() / "DeepFilterNet" / "models" / "DeepFilterNet3_onnx.tar.gz";
    return path.string();
}

TEST(NoiseSuppressionTest, InitializationFailsWithInvalidPath) {
    infrastructure::DeepFilterAdapter suppressor;
    EXPECT_FALSE(suppressor.Init("invalid_path.tar.gz"));
}

TEST(NoiseSuppressionTest, InitializationSucceedsWithValidModel) {
    infrastructure::DeepFilterAdapter suppressor;
    EXPECT_TRUE(suppressor.Init(GetModelPath()));
}

TEST(NoiseSuppressionTest, ProcessFrameReturnsValidSnr) {
    infrastructure::DeepFilterAdapter suppressor;
    ASSERT_TRUE(suppressor.Init(GetModelPath()));
    
    size_t frame_len = suppressor.GetFrameLength();
    ASSERT_GT(frame_len, 0);

    std::vector<float> input(frame_len, 0.1f); // Some dummy signal
    std::vector<float> output(frame_len, 0.0f);
    
    float snr = suppressor.ProcessFrame(input.data(), output.data());
    EXPECT_GE(snr, -100.0f); 
}

} // namespace testing
} // namespace silence_arc
