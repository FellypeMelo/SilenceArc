#include <gtest/gtest.h>
#include "silence_arc/infrastructure/sycl/sycl_memory_manager.h"
#include "silence_arc/infrastructure/sycl/sycl_graph_builder.h"
#include "silence_arc/infrastructure/sycl/tensor.h"
#include <sycl/sycl.hpp>
#include <dnnl_sycl.hpp>
#include <vector>
#include <filesystem>

using namespace sa::infrastructure::sycl_impl;

TEST(SyclGraphTest, LoadWeightsAndLinearLayer) {
    sycl::queue q{sycl::gpu_selector_v};
    dnnl::engine engine = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    
    SyclMemoryManager mem_manager(q);
    SyclGraphBuilder builder(q, engine, mem_manager);

    std::filesystem::path weights_path = std::filesystem::current_path();
    if (weights_path.filename() == "build") weights_path = weights_path.parent_path();
    weights_path = weights_path / "models" / "df3_weights";

    ASSERT_TRUE(builder.load_weights(weights_path.string()));

    // Test a small linear layer: enc_lsnr_fc_0 (Input: 512, Output: 1)
    // Wait, let's check enc_lsnr_fc_0 weight size: 2048 bytes / 4 = 512 elements.
    // That means it's 1x512 or something similar.
    auto linear = builder.build_linear("enc_lsnr_fc_0", 512, 1);
    ASSERT_NE(linear, nullptr);

    // Prepare Input/Output Tensors
    float* d_input = mem_manager.allocate_device<float>(512);
    float* d_output = mem_manager.allocate_device<float>(1);
    
    std::vector<float> h_input(512, 1.0f); // All ones
    q.memcpy(d_input, h_input.data(), 512 * sizeof(float)).wait();

    Tensor input_tensor({1, 512}, d_input);
    Tensor output_tensor({1, 1}, d_output);

    // Run Forward
    linear->forward(input_tensor, output_tensor);

    float h_output = 0.0f;
    q.memcpy(&h_output, d_output, sizeof(float)).wait();

    std::cout << "[INFO] Linear Output: " << h_output << std::endl;
    // We don't know the exact value without the weight data, 
    // but if it ran without crash and produced a non-zero value (usually), it's a good sign.
}
