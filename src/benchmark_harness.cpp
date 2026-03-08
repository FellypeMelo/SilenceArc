#include "benchmark_harness.h"
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace silence_arc {

BenchmarkHarness::BenchmarkHarness() : queue_(sycl::default_selector_v) {}

BenchmarkHarness::~BenchmarkHarness() {}

bool BenchmarkHarness::Init() {
    try {
        auto device = queue_.get_device();
        std::cout << "Running on: " << device.get_info<sycl::info::device::name>() << std::endl;
        
        // Basic check: is it a GPU? (Ideally an Intel Arc)
        if (!device.is_gpu()) {
            std::cout << "Warning: Device is not a GPU." << std::endl;
        }
        
        is_initialized_ = true;
        return true;
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

BenchmarkResult BenchmarkHarness::RunDummyInference(int num_frames) {
    if (!is_initialized_) return {};

    std::cout << "Allocating 1MB on device..." << std::endl;
    std::vector<double> latencies;
    latencies.reserve(num_frames);

    const int buffer_size = 1024 * 1024; // 1MB dummy buffer
    float* d_buf = nullptr;
    try {
        d_buf = sycl::malloc_device<float>(buffer_size, queue_);
    } catch (const sycl::exception& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        return {};
    }

    if (!d_buf) {
        std::cerr << "Memory allocation returned nullptr" << std::endl;
        return {};
    }

    std::cout << "Starting benchmark loop for " << num_frames << " frames..." << std::endl;
    for (int i = 0; i < num_frames; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        try {
            // Dummy GPU kernel to simulate work
            queue_.submit([&](sycl::handler& h) {
                h.parallel_for<class DummyInferenceKernel>(sycl::range<1>(buffer_size), [=](sycl::id<1> idx) {
                    d_buf[idx] = d_buf[idx] * 2.0f + 1.0f;
                });
            }).wait();
        } catch (const sycl::exception& e) {
            std::cerr << "Kernel execution failed at frame " << i << ": " << e.what() << std::endl;
            break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        latencies.push_back(diff.count());
    }

    std::cout << "Freeing device memory..." << std::endl;
    sycl::free(d_buf, queue_);

    BenchmarkResult result;
    if (!latencies.empty()) {
        result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        std::sort(latencies.begin(), latencies.end());
        result.p99_latency_ms = latencies[static_cast<size_t>(num_frames * 0.99)];
        result.peak_vram_mb = (buffer_size * sizeof(float)) / (1024.0f * 1024.0f);
    }

    return result;
}

BenchmarkResult BenchmarkHarness::RunRNNoiseSimulation(int num_frames) {
    if (!is_initialized_) return {};

    std::vector<double> latencies;
    latencies.reserve(num_frames);

    // RNNoise uses small dimensions
    const int state_dim = 256;
    const int input_dim = 42; // Common feature count
    float* d_state = sycl::malloc_device<float>(state_dim, queue_);
    float* d_input = sycl::malloc_device<float>(input_dim, queue_);
    float* d_weights = sycl::malloc_device<float>(state_dim * (input_dim + state_dim) * 3, queue_);

    for (int i = 0; i < num_frames; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        queue_.submit([&](sycl::handler& h) {
            h.parallel_for<class RNNoiseGRU>(sycl::range<1>(state_dim), [=](sycl::id<1> idx) {
                // Simulate GRU gate computation
                float val = 0.0f;
                for(int j=0; j<input_dim; ++j) val += d_input[j] * d_weights[idx[0]*input_dim + j];
                d_state[idx] = 1.0f / (1.0f + sycl::exp(-val));
            });
        }).wait();
        auto end = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    sycl::free(d_state, queue_);
    sycl::free(d_input, queue_);
    sycl::free(d_weights, queue_);

    BenchmarkResult result;
    if (!latencies.empty()) {
        result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        std::sort(latencies.begin(), latencies.end());
        result.p99_latency_ms = latencies[static_cast<size_t>(num_frames * 0.99)];
        result.peak_vram_mb = (state_dim * (input_dim + state_dim) * 3 * sizeof(float)) / (1024.0f * 1024.0f);
    }
    return result;
}

BenchmarkResult BenchmarkHarness::RunDeepFilterNetSimulation(int num_frames) {
    if (!is_initialized_) return {};

    std::vector<double> latencies;
    latencies.reserve(num_frames);

    // DeepFilterNet uses larger frequency domain processing
    const int num_bins = 481;
    const int num_taps = 5;
    const int conv_channels = 64;

    sycl::float2* d_spec = sycl::malloc_device<sycl::float2>(num_bins, queue_);
    sycl::float2* d_coeffs = sycl::malloc_device<sycl::float2>(num_bins * num_taps, queue_);

    for (int i = 0; i < num_frames; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        queue_.submit([&](sycl::handler& h) {
            h.parallel_for<class DeepFilterKernel>(sycl::range<1>(num_bins), [=](sycl::id<1> idx) {
                sycl::float2 sum = {0.0f, 0.0f};
                for(int t=0; t<num_taps; ++t) {
                    sycl::float2 c = d_coeffs[idx[0] * num_taps + t];
                    sycl::float2 s = d_spec[idx[0]]; // Simplified: using same bin
                    sum.x() += c.x() * s.x() - c.y() * s.y();
                    sum.y() += c.x() * s.y() + c.y() * s.x();
                }
                d_spec[idx] = sum;
            });
        }).wait();
        auto end = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    sycl::free(d_spec, queue_);
    sycl::free(d_coeffs, queue_);

    BenchmarkResult result;
    if (!latencies.empty()) {
        result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        std::sort(latencies.begin(), latencies.end());
        result.p99_latency_ms = latencies[static_cast<size_t>(num_frames * 0.99)];
        result.peak_vram_mb = (num_bins * num_taps * sizeof(sycl::float2)) / (1024.0f * 1024.0f);
    }
    return result;
}

} // namespace silence_arc
