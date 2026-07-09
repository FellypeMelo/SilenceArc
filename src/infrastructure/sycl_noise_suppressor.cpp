#include "silence_arc/infrastructure/sycl_noise_suppressor.h"
#include "silence_arc/infrastructure/sycl_accelerator.h"
#include "silence_arc/domain/attenuation_limit.h"

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
    domain::ApplyAttenuationMix(input, output, kHopSize, dry_mix_);
    return 0.0f;
}

void SyclNoiseSuppressor::SetAttenuationLimit(float limit_db) {
    dry_mix_ = domain::AttenLimitDryMix(limit_db);
}

void SyclNoiseSuppressor::SetDeepFilteringEnabled(bool enabled) {
    sycl_set_df_enabled(enabled);
}

} // namespace infrastructure
} // namespace silence_arc
