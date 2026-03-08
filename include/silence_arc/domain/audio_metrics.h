#ifndef SILENCE_ARC_DOMAIN_AUDIO_METRICS_H_
#define SILENCE_ARC_DOMAIN_AUDIO_METRICS_H_

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace silence_arc {
namespace domain {

class AudioMetrics {
public:
    static float CalculateRMS(const std::vector<float>& buffer) {
        if (buffer.empty()) return 0.0f;
        float sum_sq = std::accumulate(buffer.begin(), buffer.end(), 0.0f, [](float sum, float val) {
            return sum + (val * val);
        });
        return std::sqrt(sum_sq / buffer.size());
    }

    static float CalculateDbReduction(const std::vector<float>& original, const std::vector<float>& processed) {
        float rms_orig = CalculateRMS(original);
        float rms_proc = CalculateRMS(processed);
        
        if (rms_orig < 1e-9f) return 0.0f;
        if (rms_proc < 1e-9f) return 100.0f; // Infinite reduction for silence

        return 20.0f * std::log10(rms_orig / rms_proc);
    }

    static float CalculateRMSE(const std::vector<float>& reference, const std::vector<float>& processed) {
        size_t size = std::min(reference.size(), processed.size());
        if (size == 0) return 0.0f;

        float sum_sq_diff = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            float diff = reference[i] - processed[i];
            sum_sq_diff += (diff * diff);
        }
        return std::sqrt(sum_sq_diff / size);
    }
};

} // namespace domain
} // namespace silence_arc

#endif // SILENCE_ARC_DOMAIN_AUDIO_METRICS_H_
