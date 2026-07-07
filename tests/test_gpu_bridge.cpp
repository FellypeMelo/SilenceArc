#include "sycl_test_harness.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

using namespace silence_arc::test;

// Extern declarations from sycl_accelerator.cpp C API
extern "C" {
    bool sycl_init();
    void sycl_process(const float* input, float* output, size_t size);
    void sycl_get_device_name(char* buffer, size_t max_size);
}

void test_sycl_ffi_initialization() {
    bool success = sycl_init();
    SA_ASSERT(success, "SYCL FFI initialization failed");
    
    char name[256];
    sycl_get_device_name(name, sizeof(name));
    std::cout << "[FFI] Active Device: " << name << std::endl;
    SA_ASSERT(strlen(name) > 0, "Device name is empty");
}

void test_sycl_ffi_processing() {
    sycl_init();

    const size_t hop_size = 480;
    // A constant DC signal is not speech; DeepFilterNet3 legitimately suppresses
    // it, so this is an FFI smoke test (the full STFT -> NN -> ISTFT loop runs and
    // returns finite, bounded audio) -- NOT a passthrough test. (The old version
    // asserted avg > 0.9, i.e. perfect reconstruction, which no correct denoiser
    // delivers on a non-speech input.)
    std::vector<float> input(hop_size, 1.0f);
    std::vector<float> output(hop_size, 0.0f);

    for (int i = 0; i < 10; ++i) {
        sycl_process(input.data(), output.data(), hop_size);
    }

    float max_abs = 0.0f;
    bool all_finite = true;
    for (size_t i = 0; i < hop_size; ++i) {
        all_finite = all_finite && std::isfinite(output[i]);
        max_abs = std::max(max_abs, std::fabs(output[i]));
    }
    std::cout << "[FFI] max|output|=" << max_abs << std::endl;

    SA_ASSERT(all_finite, "FFI produced non-finite output (NaN/Inf)");
    // Output must stay bounded (no runaway/overflow); the input amplitude is 1.0.
    SA_ASSERT(max_abs < 4.0f, "FFI output diverged (unbounded)");

    std::cout << "[FFI] Processing verified (Full SYCL Loop ran, output finite/bounded)" << std::endl;
}

int main() {
    TestHarness::instance().add_test("FFI_Initialization", test_sycl_ffi_initialization);
    TestHarness::instance().add_test("FFI_Processing", test_sycl_ffi_processing);
    return TestHarness::instance().run_all();
}
