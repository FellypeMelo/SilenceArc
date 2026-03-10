#include "sycl_test_harness.h"
#include <sycl/sycl.hpp>
#include <iostream>

using namespace sa::test;

void test_detects_intel_arc_gpu() {
    bool found_arc = false;
    auto platforms = sycl::platform::get_platforms();
    
    std::cout << "[INFO] Searching for SYCL devices..." << std::endl;
    for (auto& platform : platforms) {
        auto devices = platform.get_devices();
        for (auto& device : devices) {
            std::string name = device.get_info<sycl::info::device::name>();
            std::cout << "[DEVICE] " << name << " (" 
                      << (device.is_gpu() ? "GPU" : "CPU") << ")" << std::endl;
            
            if (device.is_gpu() && name.find("Arc") != std::string::npos) {
                found_arc = true;
            }
        }
    }
    
    SA_ASSERT(found_arc, "No Intel Arc GPU detected via SYCL!");
}

int main() {
    TestHarness::instance().add_test("DetectsIntelArcGPU", test_detects_intel_arc_gpu);
    return TestHarness::instance().run_all();
}
