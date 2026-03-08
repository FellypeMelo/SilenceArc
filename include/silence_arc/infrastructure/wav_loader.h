#ifndef SILENCE_ARC_INFRASTRUCTURE_WAV_LOADER_H_
#define SILENCE_ARC_INFRASTRUCTURE_WAV_LOADER_H_

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

namespace silence_arc {
namespace infrastructure {

struct WavData {
    uint32_t sample_rate;
    uint16_t num_channels;
    uint16_t bits_per_sample;
    std::vector<float> samples;
};

class WavLoader {
public:
    static bool Load(const std::string& file_path, WavData& out_data) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) return false;

        char header[44];
        if (!file.read(header, 44)) return false;

        // Simple RIFF WAV header parsing
        if (std::string(header, 4) != "RIFF" || std::string(header + 8, 4) != "WAVE") return false;

        out_data.num_channels = *reinterpret_cast<uint16_t*>(header + 22);
        out_data.sample_rate = *reinterpret_cast<uint32_t*>(header + 24);
        out_data.bits_per_sample = *reinterpret_cast<uint16_t*>(header + 34);
        uint32_t data_size = *reinterpret_cast<uint32_t*>(header + 40);

        if (out_data.bits_per_sample != 16) return false; // Only support 16-bit PCM for now

        std::vector<int16_t> raw_samples(data_size / 2);
        if (!file.read(reinterpret_cast<char*>(raw_samples.data()), data_size)) return false;

        out_data.samples.resize(raw_samples.size());
        for (size_t i = 0; i < raw_samples.size(); ++i) {
            out_data.samples[i] = raw_samples[i] / 32768.0f;
        }

        return true;
    }
};

} // namespace infrastructure
} // namespace silence_arc

#endif // SILENCE_ARC_INFRASTRUCTURE_WAV_LOADER_H_
