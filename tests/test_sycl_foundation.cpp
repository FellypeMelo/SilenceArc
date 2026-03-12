#include <gtest/gtest.h>
#include "silence_arc/infrastructure/sycl/sycl_memory_manager.h"
#include <sycl/sycl.hpp>

using namespace sa::infrastructure::sycl_impl;

TEST(SyclFoundationTest, USMAllocationAndKernel) {
    sycl::queue q{sycl::gpu_selector_v};
    std::cout << "[INFO] Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    SyclMemoryManager mem_manager(q);

    const size_t size = 1024;
    float* data = mem_manager.allocate_shared<float>(size);
    ASSERT_NE(data, nullptr);

    // Initialize on CPU
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Execute on GPU
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            data[idx] *= 2.0f;
        });
    }).wait();

    // Verify on CPU
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i) * 2.0f);
    }

    mem_manager.free(data);
}
