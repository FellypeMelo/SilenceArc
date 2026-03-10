#include "sycl_test_harness.h"
#include <iostream>
#include <vector>
#include <cstring>

using namespace sa::test;

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
    std::vector<float> input(hop_size, 1.0f);
    std::vector<float> output(hop_size, 0.0f);
    
    // Process several frames to fill the pipeline and history
    // Due to STFT/ISTFT and overlap-add, perfect reconstruction of 1.0f 
    // might take a few frames or have windowing effects at boundaries.
    for (int i = 0; i < 10; ++i) {
        sycl_process(input.data(), output.data(), hop_size);
    }
    
    // Verify last output frame
    // Since we pass 1.0f continuously, and Vorbis window satisfies Princen-Bradley,
    // the output should eventually converge to ~1.0f.
    float sum = 0;
    for(size_t i = 0; i < hop_size; ++i) {
        sum += output[i];
    }
    float avg = sum / hop_size;
    std::cout << "[FFI] Average Output Level: " << avg << std::endl;
    
    // Check if it's reasonably close to 1.0 (some boundary effects might exist)
    SA_ASSERT(avg > 0.9f, "FFI Processing failed (Signal lost or too low)");
    
    std::cout << "[FFI] Processing verified (Full SYCL Loop)" << std::endl;
}

int main() {
    TestHarness::instance().add_test("FFI_Initialization", test_sycl_ffi_initialization);
    TestHarness::instance().add_test("FFI_Processing", test_sycl_ffi_processing);
    return TestHarness::instance().run_all();
}
