#include <iostream>
#include <iomanip>
#include <cassert>
#include "benchmark_harness.h"

void PrintResult(const std::string& name, const silence_arc::BenchmarkResult& result) {
    std::cout << "\n--- Benchmark: " << name << " ---" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Avg Latency: " << result.avg_latency_ms << " ms" << std::endl;
    std::cout << "P99 Latency: " << result.p99_latency_ms << " ms" << std::endl;
    std::cout << "Peak VRAM:   " << result.peak_vram_mb << " MB" << std::endl;
}

int main() {
    std::cout << "Running Silence Arc Candidate Benchmarks..." << std::endl;

    silence_arc::BenchmarkHarness harness;
    if (!harness.Init()) {
        std::cerr << "Failed to initialize SYCL." << std::endl;
        return 1;
    }

    const int num_frames = 1000; // More frames for better average

    // 1. RNNoise Simulation
    auto rnnoise_res = harness.RunRNNoiseSimulation(num_frames);
    PrintResult("RNNoise (Simulation)", rnnoise_res);

    // 2. DeepFilterNet Simulation
    auto dfn_res = harness.RunDeepFilterNetSimulation(num_frames);
    PrintResult("DeepFilterNet (Simulation)", dfn_res);

    // Basic Validations
    assert(rnnoise_res.avg_latency_ms > 0);
    assert(dfn_res.avg_latency_ms > 0);

    std::cout << "\nBenchmarking complete." << std::endl;
    return 0;
}
