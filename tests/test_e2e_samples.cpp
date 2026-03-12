#include <gtest/gtest.h>
#include "silence_arc/infrastructure/deep_filter_adapter.h"
#include "utils/wav_reader.h"
#include "utils/wav_writer.h"
#include <vector>
#include <filesystem>
#include <iostream>

using namespace sa;

class E2ESamplesTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto path = std::filesystem::current_path();
        if (path.filename() == "build") path = path.parent_path();
        model_path = (path / "DeepFilterNet" / "models" / "DeepFilterNet3_onnx.tar.gz").string();
        samples_dir = path / "tests" / "samples";
    }

    size_t GetModelDelay(domain::IAudioProcessor& adapter) {
        return adapter.get_latency();
    }

    void ProcessAudio(domain::IAudioProcessor& adapter, const std::vector<float>& input, std::vector<float>& output) {
        size_t frame_size = adapter.get_frame_size();
        output.resize(input.size());
        for (size_t i = 0; i + frame_size <= input.size(); i += frame_size) {
            adapter.process_frame(&input[i], &output[i], frame_size);
        }
    }

    std::string model_path;
    std::filesystem::path samples_dir;
};

TEST_F(E2ESamplesTest, ProcessHighNoiseSample) {
    infrastructure::DeepFilterAdapter adapter(model_path);
    ASSERT_TRUE(adapter.initialize());

    std::filesystem::path input_path = samples_dir / "High-Noise.wav";
    if (!std::filesystem::exists(input_path)) {
        GTEST_SKIP() << "Sample file not found: " << input_path;
    }

    auto mixed = WavReader::Read(input_path.string());
    std::vector<float> processed;
    ProcessAudio(adapter, mixed.samples, processed);

    std::filesystem::path output_path = samples_dir / "High-Noise_df_processed.wav";
    WavWriter::Write(output_path.string(), processed, mixed.sample_rate, mixed.num_channels);
}
