#pragma once

#include "silence_arc/domain/audio_processor.h"
#include "silence_arc/infrastructure/cpu_dsp_engine.h"
#include "silence_arc/infrastructure/onnx_adapter.h"
#include "silence_arc/infrastructure/feature_extractor.h"
#include <memory>
#include <vector>
#include <deque>

namespace sa::infrastructure::directml_impl {

/**
 * @brief Stable DirectML Engine for DeepFilterNet3.
 * Uses ONNX Runtime with DirectML for GPU acceleration.
 */
class DirectMLAudioEngine : public domain::IAudioProcessor {
public:
    DirectMLAudioEngine();
    ~DirectMLAudioEngine() override;

    // IAudioProcessor Implementation
    bool initialize() override;
    std::string get_device_name() const override;
    void process_frame(const float* input, float* output, size_t size) override;
    size_t get_frame_size() const override { return 480; }
    size_t get_latency() const override { return 960; }
    void set_deep_filtering_enabled(bool enabled) override { m_df_enabled = enabled; }
    void set_attenuation_limit(float limit_db) override { m_attenuation_limit = limit_db; }
    void reset() override;

private:
    void build_onnx_sessions();
    void apply_mask(std::complex<float>* spec, const float* mask);
    void compute_df_block(std::complex<float>* spec_df_out, const float* df_coeffs, size_t ref_idx);
    void apply_post_filter(std::complex<float>* enhanced, const std::complex<float>* noisy);

    // ONNX Adapters
    std::unique_ptr<OnnxAdapter> m_enc_onnx;
    std::unique_ptr<OnnxAdapter> m_erb_dec_onnx;
    std::unique_ptr<OnnxAdapter> m_df_dec_onnx;

    // Core Resources (CPU-based DSP)
    std::unique_ptr<CpuDspEngine> m_dsp;
    std::unique_ptr<FeatureExtractor> m_features;

    // State
    bool m_initialized = false;
    bool m_df_enabled = true;
    float m_attenuation_limit = 40.0f;

    // Circular buffer for Deep Filtering (storing past spectra)
    std::vector<std::vector<std::complex<float>>> m_spec_history;
    size_t m_history_idx = 0;

    // Conversion Buffers
    std::vector<float> m_feat_erb;
    std::vector<float> m_feat_spec;
    std::vector<float> m_erb_mask;
    std::vector<float> m_prev_erb_mask;
    std::vector<float> m_df_coeffs;

    // Intermediate ONNX outputs
    std::vector<float> m_emb;
    std::vector<float> m_e0, m_e1, m_e2, m_e3, m_c0;
};

} // namespace sa::infrastructure::directml_impl

