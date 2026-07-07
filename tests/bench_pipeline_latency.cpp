// Real end-to-end latency benchmark for the SYCL/oneDNN noise-suppression
// pipeline. Drives the actual sycl_process() hot path (STFT -> features -> NN ->
// filter -> ISTFT) over many synthetic frames and reports the per-frame wall
// clock percentiles.
//
// This is the "minimum performance impact" proof point: capture the baseline
// here BEFORE the hot-path refactor (removing redundant .wait() syncs, moving the
// EMA normalisation onto the device, fusing kernels), then re-run afterwards.
//
// The frame budget for a 480-sample hop at 48 kHz is 10 ms; p99 must stay under
// that hard ceiling or the pipeline cannot keep up with real-time audio.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

extern "C" {
    bool sycl_init();
    void sycl_process(const float* input, float* output, size_t size);
    void sycl_reset();
}

namespace {
constexpr size_t kHop = 480;
constexpr double kSampleRate = 48000.0;
constexpr double kPi = 3.14159265358979323846;

double Pct(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>(p * (sorted.size() - 1));
    return sorted[idx];
}
}  // namespace

int main() {
    if (!sycl_init()) {
        // No SYCL device (e.g. CI without an Arc GPU): skip, don't fail the suite.
        std::printf("[BENCH] No SYCL device available -- skipping latency benchmark.\n");
        return 0;
    }

    // A discrete Arc GPU ramps its power/clock state over the first few SECONDS,
    // not the first few frames -- a fixed short warmup measures the cold ramp, not
    // steady state. The production pipeline runs continuously, so steady-state p99
    // is the real-time metric that matters. Warm until BOTH a minimum frame count
    // and a minimum wall-clock budget are met so the GPU has reached steady clocks
    // regardless of how fast this particular host schedules the frames.
    const int warmup_min_frames = 2000;
    const double warmup_min_ms = 3000.0;
    const int measured = 10000;

    std::vector<float> in(kHop), out(kHop);
    auto fill = [&](long long frame) {
        for (size_t i = 0; i < kHop; ++i) {
            double t = static_cast<double>(frame * static_cast<long long>(kHop) + i) / kSampleRate;
            // Speech-ish: a couple of formant tones plus a little broadband noise.
            in[i] = static_cast<float>(0.3 * std::sin(2 * kPi * 220.0 * t) +
                                       0.2 * std::sin(2 * kPi * 700.0 * t) +
                                       0.1 * std::sin(2 * kPi * 2500.0 * t));
        }
    };

    sycl_reset();
    {
        auto warm_start = std::chrono::high_resolution_clock::now();
        int f = 0;
        for (;;) {
            fill(f);
            sycl_process(in.data(), out.data(), kHop);
            ++f;
            double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - warm_start).count();
            if (f >= warmup_min_frames && elapsed >= warmup_min_ms) break;
        }
    }
    const int warmup = warmup_min_frames;

    std::vector<double> lat;
    lat.reserve(measured);
    for (int f = 0; f < measured; ++f) {
        fill(warmup + f);
        auto t0 = std::chrono::high_resolution_clock::now();
        sycl_process(in.data(), out.data(), kHop);
        auto t1 = std::chrono::high_resolution_clock::now();
        lat.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double avg = std::accumulate(lat.begin(), lat.end(), 0.0) / lat.size();
    std::sort(lat.begin(), lat.end());
    double p50 = Pct(lat, 0.50), p90 = Pct(lat, 0.90), p99 = Pct(lat, 0.99), mx = lat.back();

    std::printf("[BENCH] frames=%d  avg=%.3f ms  p50=%.3f  p90=%.3f  p99=%.3f  max=%.3f ms\n",
                measured, avg, p50, p90, p99, mx);

    if (p99 > 10.0) {
        std::printf("[BENCH] FAIL: p99 %.3f ms exceeds the 10 ms real-time frame budget.\n", p99);
        return 1;
    }
    std::printf("[BENCH] OK: p99 within real-time budget.\n");
    return 0;
}
