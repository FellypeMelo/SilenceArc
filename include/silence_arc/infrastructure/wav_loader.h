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

        // Downmix to mono if stereo or more
        size_t frames = raw_samples.size() / out_data.num_channels;
        out_data.samples.resize(frames);
        
        for (size_t i = 0; i < frames; ++i) {
            float sum = 0.0f;
            for (size_t ch = 0; ch < out_data.num_channels; ++ch) {
                sum += raw_samples[i * out_data.num_channels + ch] / 32768.0f;
            }
            out_data.samples[i] = sum / out_data.num_channels;
        }
        
        // Ensure state reflects mono
        out_data.num_channels = 1;

        return true;
    }
};

class WavWriter {
public:
    static bool Save(const std::string& file_path, const WavData& data) {
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open()) return false;

        uint32_t data_size = static_cast<uint32_t>(data.samples.size() * 2);
        uint32_t file_size = 36 + data_size;

        file.write("RIFF", 4);
        file.write(reinterpret_cast<const char*>(&file_size), 4);
        file.write("WAVE", 4);
        file.write("fmt ", 4);
        uint32_t fmt_size = 16;
        file.write(reinterpret_cast<const char*>(&fmt_size), 4);
        uint16_t audio_format = 1; // PCM
        file.write(reinterpret_cast<const char*>(&audio_format), 2);
        file.write(reinterpret_cast<const char*>(&data.num_channels), 2);
        file.write(reinterpret_cast<const char*>(&data.sample_rate), 4);
        uint32_t byte_rate = data.sample_rate * data.num_channels * 2;
        file.write(reinterpret_cast<const char*>(&byte_rate), 4);
        uint16_t block_align = data.num_channels * 2;
        file.write(reinterpret_cast<const char*>(&block_align), 2);
        uint16_t bits_per_sample = 16;
        file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
        file.write("data", 4);
        file.write(reinterpret_cast<const char*>(&data_size), 4);

        for (float sample : data.samples) {
            int16_t val = static_cast<int16_t>(std::max(-1.0f, std::min(1.0f, sample)) * 32767.0f);
            file.write(reinterpret_cast<const char*>(&val), 2);
        }

        return true;
    }
};

} // namespace infrastructure
} // namespace silence_arc

#endif // SILENCE_ARC_INFRASTRUCTURE_WAV_LOADER_H_
