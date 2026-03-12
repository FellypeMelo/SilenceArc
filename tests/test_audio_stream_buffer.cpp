#include <gtest/gtest.h>
#include "silence_arc/domain/audio_stream_buffer.h"
#include <vector>

using namespace sa::domain;

TEST(AudioStreamBufferTest, BasicPushPop) {
    AudioStreamBuffer buffer(10);
    float input[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float output[5] = {0.0f};

    buffer.Push(input, 5);
    EXPECT_EQ(buffer.Available(), 5);

    size_t popped = buffer.Pop(output, 5);
    EXPECT_EQ(popped, 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(output[i], input[i]);
    }
    EXPECT_EQ(buffer.Available(), 0);
}

TEST(AudioStreamBufferTest, OverflowHandling) {
    AudioStreamBuffer buffer(5);
    float input[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    buffer.Push(input, 10);
    EXPECT_EQ(buffer.Available(), 5); // Should only keep last 5
    
    float output[5];
    buffer.Pop(output, 5);
    EXPECT_EQ(output[0], 6.0f);
    EXPECT_EQ(output[4], 10.0f);
}
