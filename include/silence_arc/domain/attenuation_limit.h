#ifndef SILENCE_ARC_DOMAIN_ATTENUATION_LIMIT_H_
#define SILENCE_ARC_DOMAIN_ATTENUATION_LIMIT_H_

#include <cmath>
#include <cstddef>

namespace silence_arc {
namespace domain {

// Maps an attenuation limit in dB to the linear weight of the DRY (noisy) signal
// used in the post-inference dry/wet mix, matching libDF's DFState::set_atten_lim:
//   |db| >= 100  -> 0.0  (no limit, fully wet / fully enhanced)
//   |db| <  0.01 -> 1.0  (bypass, fully dry / no suppression)
//   otherwise    -> 10^(-|db|/20)
inline float AttenLimitDryMix(float limit_db) {
    const float lim = std::fabs(limit_db);
    if (lim >= 100.0f) return 0.0f;
    if (lim < 0.01f) return 1.0f;
    return std::pow(10.0f, -lim / 20.0f);
}

// Applies the dry/wet mix in place: out[i] = (1 - dry) * out[i] + dry * in[i].
// `out` holds the enhanced (wet) frame on entry; `in` is the original (dry) frame.
// A dry weight of 0 leaves `out` untouched (fully wet).
inline void ApplyAttenuationMix(const float* in, float* out, size_t n, float dry) {
    if (dry <= 0.0f) return;
    const float wet = 1.0f - dry;
    for (size_t i = 0; i < n; ++i) {
        out[i] = wet * out[i] + dry * in[i];
    }
}

} // namespace domain
} // namespace silence_arc

#endif // SILENCE_ARC_DOMAIN_ATTENUATION_LIMIT_H_
