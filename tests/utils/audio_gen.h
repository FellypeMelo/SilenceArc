#ifndef SILENCE_ARC_TESTS_UTILS_AUDIO_GEN_H_
#define SILENCE_ARC_TESTS_UTILS_AUDIO_GEN_H_

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace silence_arc {
namespace tests {
namespace utils {

class AudioGen {
public:
    static std::vector<float> GenerateSine(float frequency, float duration_sec, uint32_t sample_rate, float amplitude = 0.5f) {
        size_t num_samples = static_cast<size_t>(duration_sec * sample_rate);
        std::vector<float> buffer(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            buffer[i] = amplitude * std::sin(2.0f * M_PI * frequency * i / sample_rate);
        }
        return buffer;
    }

    static std::vector<float> GenerateWhiteNoise(float duration_sec, uint32_t sample_rate, float amplitude = 0.1f, uint32_t seed = 42) {
        size_t num_samples = static_cast<size_t>(duration_sec * sample_rate);
        std::vector<float> buffer(num_samples);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(-amplitude, amplitude);
        for (size_t i = 0; i < num_samples; ++i) {
            buffer[i] = dis(gen);
        }
        return buffer;
    }

    static std::vector<float> Mix(const std::vector<float>& signal, const std::vector<float>& noise, float noise_level = 1.0f) {
        size_t size = std::min(signal.size(), noise.size());
        std::vector<float> mixed(size);
        for (size_t i = 0; i < size; ++i) {
            mixed[i] = signal[i] + (noise[i] * noise_level);
            // Simple clipping
            if (mixed[i] > 1.0f) mixed[i] = 1.0f;
            if (mixed[i] < -1.0f) mixed[i] = -1.0f;
        }
        return mixed;
    }
};

} // namespace utils
} // namespace tests
} // namespace silence_arc

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#endif // SILENCE_ARC_TESTS_UTILS_AUDIO_GEN_H_
