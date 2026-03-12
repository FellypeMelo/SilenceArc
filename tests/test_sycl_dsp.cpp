#include <gtest/gtest.h>
#include "silence_arc/infrastructure/sycl/sycl_memory_manager.h"
#include "silence_arc/infrastructure/sycl/sycl_dsp_engine.h"
#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>

using namespace sa::infrastructure::sycl_impl;

TEST(SyclDspTest, SignalLoopback) {
    sycl::queue q{sycl::gpu_selector_v};
    SyclMemoryManager mem_manager(q);
    SyclDspEngine dsp(q, mem_manager);
    dsp.initialize();

    const size_t hop_size = 480;
    const size_t num_frames = 10;
    
    std::vector<float> input_audio(hop_size * num_frames);
    std::vector<float> output_audio(hop_size * num_frames);

    // Generate 1kHz Sine Wave
    const float freq = 1000.0f;
    const float sample_rate = 48000.0f;
    for (size_t i = 0; i < input_audio.size(); ++i) {
        input_audio[i] = std::sin(2.0f * 3.14159f * freq * i / sample_rate);
    }

    float* d_input_hop = mem_manager.allocate_device<float>(hop_size);
    float* d_output_hop = mem_manager.allocate_device<float>(hop_size);
    std::complex<float>* d_freq_480 = mem_manager.allocate_device<std::complex<float>>(hop_size);

    // Process Frames
    for (size_t f = 0; f < num_frames; ++f) {
        q.memcpy(d_input_hop, input_audio.data() + f * hop_size, hop_size * sizeof(float)).wait();
        
        dsp.analyze(d_input_hop, d_freq_480);
        dsp.synthesize(d_freq_480, d_output_hop);
        
        q.memcpy(output_audio.data() + f * hop_size, d_output_hop, hop_size * sizeof(float)).wait();
    }

    // Verify Signal (Ignore first frame due to OLA cold start)
    for (size_t i = hop_size; i < output_audio.size(); ++i) {
        EXPECT_NEAR(input_audio[i], output_audio[i], 1e-3f) << "Signal mismatch at sample " << i;
    }
}
