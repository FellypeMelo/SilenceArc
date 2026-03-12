#include "silence_arc/infrastructure/sycl_noise_suppressor.h"
#include "silence_arc/infrastructure/sycl_accelerator.h"

namespace silence_arc {
namespace infrastructure {

SyclNoiseSuppressor::SyclNoiseSuppressor() {}

bool SyclNoiseSuppressor::Init(const std::string& /*model_path*/) {
    // sycl_init is already called in main.cpp, but calling it again is safe due to the mutex and null-check
    return sycl_init();
}

size_t SyclNoiseSuppressor::GetFrameLength() const {
    return 480; // Matches SYCLAccelerator m_hop_size
}

float SyclNoiseSuppressor::ProcessFrame(const float* input, float* output) {
    sycl_process(input, output, 480);
    return 0.0f;
}

void SyclNoiseSuppressor::SetAttenuationLimit(float /*limit_db*/) {
    // Future: Implement via global SYCL parameters if needed
}

void SyclNoiseSuppressor::SetDeepFilteringEnabled(bool enabled) {
    sycl_set_df_enabled(enabled);
}

} // namespace infrastructure
} // namespace silence_arc
