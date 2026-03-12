#include <gtest/gtest.h>
#include "silence_arc/infrastructure/async_audio_pipeline.h"
#include <vector>
#include <thread>
#include <chrono>

using namespace sa;

TEST(AudioPipelineTest, AsyncProcessing) {
    infrastructure::AsyncAudioPipeline pipeline;
    
    bool processed = false;
    pipeline.SetProcessCallback([&](const domain::AudioBuffer& input, domain::AudioBuffer& output) {
        processed = true;
        output.data = input.data;
    });

    pipeline.Start();

    domain::AudioBuffer input;
    input.data.assign(480, 0.1f);
    input.sample_rate = 48000;
    input.num_channels = 1;

    pipeline.PushInput(input);

    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    domain::AudioBuffer output;
    bool has_output = pipeline.PopOutput(output);

    EXPECT_TRUE(processed);
    EXPECT_TRUE(has_output);
    EXPECT_EQ(output.data.size(), input.data.size());

    pipeline.Stop();
}
