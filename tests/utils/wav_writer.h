#pragma once

#include "miniaudio.h"
#include <vector>
#include <string>
#include <stdexcept>

class WavWriter {
public:
    static void Write(const std::string& file_path, const std::vector<float>& samples, 
                      unsigned int sample_rate, unsigned int num_channels) {
        ma_encoder_config config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, num_channels, sample_rate);
        ma_encoder encoder;
        if (ma_encoder_init_file(file_path.c_str(), &config, &encoder) != MA_SUCCESS) {
            throw std::runtime_error("Failed to create WAV file: " + file_path);
        }

        ma_uint64 frames_written;
        ma_encoder_write_pcm_frames(&encoder, samples.data(), samples.size() / num_channels, &frames_written);

        ma_encoder_uninit(&encoder);
    }
};
