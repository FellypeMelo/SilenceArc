#include "sycl_test_harness.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

using namespace sa::test;

extern "C" {
    bool sycl_init();
    void sycl_process(const float* input, float* output, size_t size);
}

void test_complex_signal_preservation() {
    SA_ASSERT(sycl_init(), "Init failed");
    
    const size_t hop_size = 480;
    const double sample_rate = 48000.0;
    const size_t num_frames = 100;
    const double pi = 3.14159265358979323846;
    
    std::vector<float> input(hop_size);
    std::vector<float> output(hop_size);
    
    // Generate a signal: sum of sines (1kHz and 5kHz)
    auto generate_frame = [&](size_t frame_idx) {
        for (size_t i = 0; i < hop_size; ++i) {
            double t = static_cast<double>(frame_idx * hop_size + i) / sample_rate;
            input[i] = static_cast<float>(0.5 * std::sin(2.0 * pi * 1000.0 * t) + 
                                         0.3 * std::sin(2.0 * pi * 5000.0 * t));
        }
    };

    double total_mse = 0;
    size_t count = 0;

    for (size_t f = 0; f < num_frames; ++f) {
        generate_frame(f);
        sycl_process(input.data(), output.data(), hop_size);
        
        // Skip first few frames to account for STFT latency
        if (f > 10) {
            for (size_t i = 0; i < hop_size; ++i) {
                float diff = input[i] - output[i];
                total_mse += diff * diff;
                count++;
            }
        }
    }

    double mse = total_mse / count;
    std::cout << "[KERNELS] Mean Squared Error (Synthetic Signal): " << mse << std::endl;
    
    // Check if MSE is low enough (preservation of signal)
    SA_ASSERT(mse < 1e-4, "Signal preservation failed (MSE too high)");
}

int main() {
    TestHarness::instance().add_test("SignalPreservation", test_complex_signal_preservation);
    return TestHarness::instance().run_all();
}
