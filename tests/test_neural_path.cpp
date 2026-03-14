#include <gtest/gtest.h>
#include "silence_arc/infrastructure/directml_audio_engine.h"
#include <vector>
#include <cmath>
#include <numeric>
#include <chrono>

using namespace sa::infrastructure::directml_impl;

class NeuralPathTestFixture : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        s_engine = new DirectMLAudioEngine();
        s_engine->initialize();
    }

    static void TearDownTestSuite() {
        delete s_engine;
        s_engine = nullptr;
    }

    static DirectMLAudioEngine* s_engine;
};

DirectMLAudioEngine* NeuralPathTestFixture::s_engine = nullptr;

TEST_F(NeuralPathTestFixture, GraphAssembly) {
    ASSERT_NE(s_engine, nullptr);
    EXPECT_EQ(s_engine->get_frame_size(), 480);
}

TEST_F(NeuralPathTestFixture, SineWaveEnergyPass) {
    const size_t hop_size = 480;
    const size_t num_frames = 50;
    std::vector<float> input_frame(hop_size);
    std::vector<float> output_frame(hop_size);
    
    float total_output_energy = 0.0f;
    float input_rms = 0.0f;

    std::cout << "[INFO] Processing " << num_frames << " frames of 1kHz sine wave..." << std::endl;

    for (size_t f = 0; f < num_frames; ++f) {
        // Generate 1kHz sine wave for this frame
        for (size_t i = 0; i < hop_size; ++i) {
            float t = (float)(f * hop_size + i) / 48000.0f;
            input_frame[i] = std::sin(2.0f * 3.14159f * 1000.0f * t);
        }

        auto start = std::chrono::high_resolution_clock::now();
        s_engine->process_frame(input_frame.data(), output_frame.data(), hop_size);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (f % 10 == 0) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "[PERF] Frame " << f << " processing time: " << ms << "ms" << std::endl;
        }

        // Only accumulate energy after the first 10 frames (OLA + Model latency)
        if (f > 10) {
            for (size_t i = 0; i < hop_size; ++i) {
                total_output_energy += output_frame[i] * output_frame[i];
            }
        }
        
        // Calculate input RMS for reference (on last frame)
        if (f == num_frames - 1) {
            float sum = 0;
            for (auto v : input_frame) sum += v * v;
            input_rms = std::sqrt(sum / hop_size);
        }
    }

    float output_rms = std::sqrt(total_output_energy / ((num_frames - 11) * hop_size));
    std::cout << "[INFO] Input RMS: " << input_rms << ", Output RMS: " << output_rms << std::endl;

    // Verify that energy is not zero (silence bug)
    EXPECT_GT(output_rms, 1e-3) << "Output is too silent! Signal might be getting zeroed out or model is over-suppressing.";
}
