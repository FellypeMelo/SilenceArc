#ifndef SILENCE_ARC_BENCHMARK_HARNESS_H_
#define SILENCE_ARC_BENCHMARK_HARNESS_H_

#include <sycl/sycl.hpp>
#include <vector>

namespace silence_arc {

struct BenchmarkResult {
    float avg_latency_ms = 0.0f;
    float p99_latency_ms = 0.0f;
    float peak_vram_mb = 0.0f;
};

class BenchmarkHarness {
public:
    BenchmarkHarness();
    ~BenchmarkHarness();

    bool Init();
    BenchmarkResult RunDummyInference(int num_frames);
    BenchmarkResult RunRNNoiseSimulation(int num_frames);
    BenchmarkResult RunDeepFilterNetSimulation(int num_frames);

private:
    sycl::queue queue_;
    bool is_initialized_ = false;
};

} // namespace silence_arc

#endif // SILENCE_ARC_BENCHMARK_HARNESS_H_
