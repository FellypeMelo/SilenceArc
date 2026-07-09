#include "silence_arc/infrastructure/sycl_noise_suppressor.h"
#include "silence_arc/infrastructure/sycl_accelerator.h"
#include <cmath>

namespace silence_arc {
namespace infrastructure {

namespace {
constexpr size_t kHopSize = 480; // Matches SYCLAccelerator m_hop_size
}

SyclNoiseSuppressor::SyclNoiseSuppressor() {}

bool SyclNoiseSuppressor::Init(const std::string& /*model_path*/) {
    // sycl_init is already called in main.cpp, but calling it again is safe due to the mutex and null-check
    return sycl_init();
}

size_t SyclNoiseSuppressor::GetFrameLength() const {
    return kHopSize;
}

float SyclNoiseSuppressor::ProcessFrame(const float* input, float* output) {
    sycl_process(input, output, kHopSize);

    // Attenuation limit: mix a fraction of the dry (noisy) signal back in so the
    // effective noise attenuation never exceeds the configured limit. This mirrors
    // libDF's post-inference mix (out = (1-lim)*enh + lim*noisy). Done here on the
    // worker thread, after inference -- the RT device callback is untouched.
    const float dry = dry_mix_;
    if (dry > 0.0f) {
        const float wet = 1.0f - dry;
        for (size_t i = 0; i < kHopSize; ++i) {
            output[i] = wet * output[i] + dry * input[i];
        }
    }
    return 0.0f;
}

void SyclNoiseSuppressor::SetAttenuationLimit(float limit_db) {
    // Same mapping as libDF (DFState::set_atten_lim): |db| >= 100 disables the
    // limit (fully wet), |db| < 0.01 is treated as full bypass (fully dry), and
    // anything in between yields a linear dry weight of 10^(-|db|/20).
    const float lim = std::fabs(limit_db);
    if (lim >= 100.0f) {
        dry_mix_ = 0.0f;
    } else if (lim < 0.01f) {
        dry_mix_ = 1.0f;
    } else {
        dry_mix_ = std::pow(10.0f, -lim / 20.0f);
    }
}

void SyclNoiseSuppressor::SetDeepFilteringEnabled(bool enabled) {
    sycl_set_df_enabled(enabled);
}

} // namespace infrastructure
} // namespace silence_arc
