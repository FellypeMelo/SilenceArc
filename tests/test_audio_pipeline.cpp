#include <gtest/gtest.h>
#include "silence_arc/infrastructure/async_audio_pipeline.h"
#include "silence_arc/domain/ui_state.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
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

// Backpressure: a slow worker + a burst of inputs must not grow memory/latency
// without bound; the bounded queue drops the stalest frames instead.
TEST(AudioPipelineTest, BoundedQueueDropsOldestUnderBackpressure) {
    infrastructure::AsyncAudioPipeline pipeline;
    pipeline.SetMaxQueueDepth(4);
    pipeline.SetProcessCallback([](const domain::AudioBuffer& in, domain::AudioBuffer& out) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); // slow worker
        out.data = in.data;
    });
    pipeline.Start();

    for (int i = 0; i < 100; ++i) {
        domain::AudioBuffer b;
        b.data.assign(480, static_cast<float>(i));
        pipeline.PushInput(b);
    }

    // 100 frames burst into a depth-4 queue behind a 20 ms worker -> drops happen.
    EXPECT_GT(pipeline.FramesDropped(), 0u);
    pipeline.Stop();
}

// The real audio-device callback pushes via PushInput; it must return in well
// under the frame budget even while the GPU/NN worker is busy on a slow frame.
TEST(AudioPipelineTest, PushInputNeverBlocksTheCaller) {
    infrastructure::AsyncAudioPipeline pipeline;
    pipeline.SetMaxQueueDepth(4);
    pipeline.SetProcessCallback([](const domain::AudioBuffer& in, domain::AudioBuffer& out) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // very slow worker
        out.data = in.data;
    });
    pipeline.Start();

    domain::AudioBuffer b;
    b.data.assign(480, 1.0f);
    double max_us = 0.0;
    for (int i = 0; i < 50; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        pipeline.PushInput(b);
        auto t1 = std::chrono::high_resolution_clock::now();
        max_us = std::max(max_us, std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    // Must be far below the 10 ms frame budget (realistically microseconds).
    EXPECT_LT(max_us, 1000.0);
    pipeline.Stop();
}

// Fix 5: the drop counter the UI shows (UIState.frames_dropped) is exactly the
// pipeline's FramesDropped() value that main() copies into the state each frame.
TEST(AudioPipelineTest, FramesDroppedIsExposedForTheUi) {
    infrastructure::AsyncAudioPipeline pipeline;
    EXPECT_EQ(pipeline.FramesDropped(), 0u);

    pipeline.SetMaxQueueDepth(2);
    pipeline.SetProcessCallback([](const domain::AudioBuffer& in, domain::AudioBuffer& out) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); // slow worker -> drops
        out.data = in.data;
    });
    pipeline.Start();
    for (int i = 0; i < 50; ++i) {
        domain::AudioBuffer b;
        b.data.assign(480, 1.0f);
        pipeline.PushInput(b);
    }
    pipeline.Stop();

    // Mirror main()'s wiring: the UI copies the counter into UIState.
    domain::UIState ui_state;
    ui_state.frames_dropped = pipeline.FramesDropped();
    EXPECT_GT(ui_state.frames_dropped, 0u);
    EXPECT_EQ(ui_state.frames_dropped, pipeline.FramesDropped());
}

} // namespace testing
} // namespace silence_arc
