#include <gtest/gtest.h>
#include "silence_arc/infrastructure/sycl/native_sycl_engine.h"
#include <vector>
#include <filesystem>

using namespace sa::infrastructure::sycl_impl;

TEST(NativeSyclEngineTest, InitializationAndPassThrough) {
    NativeSyclEngine engine;
    
    // Test Initialization (includes weight loading)
    ASSERT_TRUE(engine.initialize());
    
    std::cout << "[INFO] Engine initialized on: " << engine.get_device_name() << std::endl;

    // Test Pass-through (since inference is currently stubbed)
    const size_t size = 480;
    std::vector<float> input(size, 0.1f);
    std::vector<float> output(size, 0.0f);
    
    engine.process_frame(input.data(), output.data(), size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(input[i], output[i]);
    }
}

TEST(NativeSyclEngineTest, Reset) {
    NativeSyclEngine engine;
    engine.initialize();
    engine.reset(); // Should not crash
}
