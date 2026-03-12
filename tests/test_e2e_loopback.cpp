#include <gtest/gtest.h>
#include "silence_arc/infrastructure/deep_filter_adapter.h"
#include "silence_arc/domain/audio_pipeline.h"
#include <vector>
#include <filesystem>

using namespace sa;

TEST(E2ELoopbackTest, FullProcessingCycle) {
    auto path = std::filesystem::current_path();
    if (path.filename() == "build") path = path.parent_path();
    auto model_path = path / "DeepFilterNet" / "models" / "DeepFilterNet3_onnx.tar.gz";

    infrastructure::DeepFilterAdapter suppressor(model_path.string());
    ASSERT_TRUE(suppressor.initialize());

    size_t frame_size = suppressor.get_frame_size();
    domain::AudioBuffer input;
    input.data.assign(frame_size, 0.5f);
    input.num_channels = 1;
    input.sample_rate = 48000;

    domain::AudioBuffer output;
    output.data.resize(frame_size);
    output.num_channels = 1;
    output.sample_rate = 48000;

    // Process
    suppressor.process_frame(input.data.data(), output.data.data(), frame_size);

    // Verify
    EXPECT_EQ(output.data.size(), frame_size);
}
