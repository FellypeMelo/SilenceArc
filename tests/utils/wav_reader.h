#pragma once

#include "miniaudio.h"
#include <vector>
#include <string>
#include <stdexcept>

struct WavData {
    std::vector<float> samples;
    unsigned int sample_rate;
    unsigned int num_channels;
};

class WavReader {
public:
    static WavData Read(const std::string& file_path) {
        ma_decoder decoder;
        if (ma_decoder_init_file(file_path.c_str(), NULL, &decoder) != MA_SUCCESS) {
            throw std::runtime_error("Failed to open WAV file: " + file_path);
        }

        ma_uint64 total_frames;
        ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames);

        WavData data;
        data.sample_rate = decoder.outputSampleRate;
        data.num_channels = decoder.outputChannels;
        data.samples.resize(total_frames * data.num_channels);

        ma_uint64 frames_read;
        ma_decoder_read_pcm_frames(&decoder, data.samples.data(), total_frames, &frames_read);

        ma_decoder_uninit(&decoder);
        return data;
    }
};
