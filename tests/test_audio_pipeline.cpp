#include <gtest/gtest.h>
#include "silence_arc/infrastructure/async_audio_pipeline.h"
#include <chrono>
#include <thread>

namespace silence_arc {
namespace testing {

TEST(AudioPipelineTest, StartStop) {
    infrastructure::AsyncAudioPipeline pipeline;
    EXPECT_FALSE(pipeline.IsRunning());
    EXPECT_TRUE(pipeline.Start());
    EXPECT_TRUE(pipeline.IsRunning());
    pipeline.Stop();
    EXPECT_FALSE(pipeline.IsRunning());
}

TEST(AudioPipelineTest, ProcessCallbackIsCalled) {
    infrastructure::AsyncAudioPipeline pipeline;
    bool callback_called = false;
    
    pipeline.SetProcessCallback([&](const domain::AudioBuffer& input, domain::AudioBuffer& output) {
        callback_called = true;
        output.data = input.data;
    });
    
    pipeline.Start();
    
    domain::AudioBuffer input;
    input.data = {0.1f, 0.2f, 0.3f};
    pipeline.PushInput(input);
    
    // Wait for processing (with timeout)
    auto start = std::chrono::steady_clock::now();
    while (!callback_called && std::chrono::steady_clock::now() - start < std::chrono::seconds(1)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    EXPECT_TRUE(callback_called);
    
    domain::AudioBuffer output;
    EXPECT_TRUE(pipeline.PopOutput(output));
    EXPECT_EQ(output.data.size(), input.data.size());
    EXPECT_FLOAT_EQ(output.data[0], 0.1f);
    
    pipeline.Stop();
}

} // namespace testing
} // namespace silence_arc
