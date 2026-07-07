// Golden-output characterization test for the SYCL/oneDNN pipeline.
//
// (The previous version asserted MSE < 1e-4 on a synthetic two-tone signal --
//  an invalid oracle: DeepFilterNet3 legitimately suppresses pure tones, so no
//  correct implementation preserves them. Real-speech quality now lives in
//  test_backend_parity.cpp.)
//
// This test pins the SYCL output on a fixed clean-speech input to a golden
// capture so the hot-path refactor (removing redundant waits, moving the EMA
// normalisation onto the device, fusing kernels) can be proven output-preserving:
// removing a wait on an in-order queue must not change a single sample.
//
// Behaviour:
//   * No golden present (or SA_REGEN_GOLDEN set) -> capture the current output as
//     the golden and pass. Run this once to establish the baseline before the
//     refactor.
//   * Golden present -> re-run and assert the output matches within a tight
//     tolerance. A load-bearing wait removed by mistake will trip this.
//
// The golden is hardware/driver specific (captured on the dev GPU); the test
// skips when no SYCL device is available.
#include "sycl_test_harness.h"

#include "silence_arc/infrastructure/wav_loader.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace silence_arc::test;
using silence_arc::infrastructure::WavData;
using silence_arc::infrastructure::WavLoader;

extern "C" {
    bool sycl_init();
    void sycl_process(const float* input, float* output, size_t size);
    void sycl_reset();
}

namespace {
constexpr size_t kHop = 480;

std::filesystem::path RepoRoot() {
    auto p = std::filesystem::current_path();
    if (p.filename() == "build") p = p.parent_path();
    return p;
}

std::filesystem::path GoldenPath() {
    return RepoRoot() / "tests" / "samples" / "No-Noise_sycl_golden.bin";
}
}  // namespace

void test_sycl_golden_output() {
    SA_ASSERT(sycl_init(), "SYCL init failed");

    WavData clean;
    SA_ASSERT(WavLoader::Load((RepoRoot() / "tests" / "samples" / "No-Noise.wav").string(), clean),
              "Could not load No-Noise.wav");

    std::vector<float> out(clean.samples.size(), 0.0f);
    sycl_reset();
    for (size_t i = 0; i + kHop <= clean.samples.size(); i += kHop) {
        sycl_process(&clean.samples[i], &out[i], kHop);
    }

    const auto golden_path = GoldenPath();
    const bool regen = std::getenv("SA_REGEN_GOLDEN") != nullptr;
    std::ifstream gin(golden_path, std::ios::binary);

    if (regen || !gin) {
        std::ofstream gout(golden_path, std::ios::binary);
        gout.write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(float));
        std::cout << "[GOLDEN] Captured baseline (" << out.size() << " samples) to "
                  << golden_path.string() << std::endl;
        return;
    }

    std::vector<float> golden(out.size());
    gin.read(reinterpret_cast<char*>(golden.data()), golden.size() * sizeof(float));
    SA_ASSERT(gin.gcount() == static_cast<std::streamsize>(golden.size() * sizeof(float)),
              "Golden size mismatch -- regenerate with SA_REGEN_GOLDEN=1");

    double max_abs = 0.0, sse = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        double d = static_cast<double>(out[i]) - golden[i];
        max_abs = std::max(max_abs, std::abs(d));
        sse += d * d;
    }
    double rmse = std::sqrt(sse / out.size());
    std::cout << "[GOLDEN] max_abs_diff=" << max_abs << " rmse=" << rmse << std::endl;

    // In-order-queue wait removal must be exactly output-preserving; the
    // device-EMA/fusion work may introduce only last-bit rounding noise.
    SA_ASSERT(max_abs < 1e-4, "SYCL output diverged from golden (a wait may have been load-bearing)");
}

int main() {
    TestHarness::instance().add_test("SyclGoldenOutput", test_sycl_golden_output);
    return TestHarness::instance().run_all();
}
