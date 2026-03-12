#include <gtest/gtest.h>
#include "silence_arc/infrastructure/deep_filter_adapter.h"
#include <vector>
#include <filesystem>

using namespace sa::infrastructure;

class NoiseSuppressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto path = std::filesystem::current_path();
        if (path.filename() == "build") path = path.parent_path();
        model_path = (path / "DeepFilterNet" / "models" / "DeepFilterNet3_onnx.tar.gz").string();
    }

    std::string model_path;
};

TEST_F(NoiseSuppressionTest, Initialization) {
    DeepFilterAdapter suppressor(model_path);
    EXPECT_TRUE(suppressor.initialize());
}

TEST_F(NoiseSuppressionTest, FrameSize) {
    DeepFilterAdapter suppressor(model_path);
    suppressor.initialize();
    EXPECT_GT(suppressor.get_frame_size(), 0);
}

TEST_F(NoiseSuppressionTest, ProcessFrame) {
    DeepFilterAdapter suppressor(model_path);
    suppressor.initialize();
    
    size_t size = suppressor.get_frame_size();
    std::vector<float> input(size, 0.1f);
    std::vector<float> output(size, 0.0f);
    
    suppressor.process_frame(input.data(), output.data(), size);
    
    // Output should be different from input (processed)
    bool different = false;
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(input[i] - output[i]) > 1e-6f) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}
