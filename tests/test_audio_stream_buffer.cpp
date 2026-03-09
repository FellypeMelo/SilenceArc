#include <gtest/gtest.h>
#include "silence_arc/domain/audio_stream_buffer.h"
#include <vector>

namespace silence_arc {
namespace testing {

using namespace silence_arc::domain;

TEST(AudioStreamBufferTest, PushAndPopCorrectSizes) {
    AudioStreamBuffer buffer;
    
    EXPECT_EQ(buffer.Available(), 0);

    // Push 10 items
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    buffer.Push(input.data(), 10);
    EXPECT_EQ(buffer.Available(), 10);

    // Pop 4 items
    std::vector<float> out1(4, 0.0f);
    buffer.Pop(out1.data(), 4);
    EXPECT_EQ(buffer.Available(), 6);
    EXPECT_FLOAT_EQ(out1[0], 1.0f);
    EXPECT_FLOAT_EQ(out1[3], 4.0f);

    // Push 3 more items
    std::vector<float> input2 = {11.0f, 12.0f, 13.0f};
    buffer.Push(input2.data(), 3);
    EXPECT_EQ(buffer.Available(), 9);

    // Pop the rest
    std::vector<float> out2(9, 0.0f);
    buffer.Pop(out2.data(), 9);
    EXPECT_EQ(buffer.Available(), 0);
    EXPECT_FLOAT_EQ(out2[0], 5.0f);
    EXPECT_FLOAT_EQ(out2[5], 10.0f);
    EXPECT_FLOAT_EQ(out2[6], 11.0f);
    EXPECT_FLOAT_EQ(out2[8], 13.0f);
}

} // namespace testing
} // namespace silence_arc
