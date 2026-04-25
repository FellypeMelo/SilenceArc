#include "silence_arc/infrastructure/directml_audio_engine.h"
#include "silence_arc/infrastructure/onnx_adapter.h"
#include "silence_arc/infrastructure/cpu_dsp_engine.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace sa::infrastructure::directml_impl {

DirectMLAudioEngine::DirectMLAudioEngine() : m_initialized(false), m_df_enabled(true) {
}

DirectMLAudioEngine::~DirectMLAudioEngine() {
    m_enc_onnx.reset();
    m_erb_dec_onnx.reset();
    m_df_dec_onnx.reset();
    m_dsp.reset();
}

bool DirectMLAudioEngine::initialize() {
    if (m_initialized) return true;

    std::cout << "[INFO] Initializing DirectML Engine (481 Bins Mode)..." << std::endl;

    try {
        // 1. DSP Engine - 481 bins (Nyquist inclusive)
        m_dsp = std::make_unique<CpuDspEngine>(960, 480);
        m_dsp->initialize();

        // 2. Feature Extractor
        m_features = std::make_unique<FeatureExtractor>(48000, 960, 32);
        m_features->initialize();

        // 3. Initialize ONNX Sessions
        build_onnx_sessions();

        // 4. Pre-allocate buffers correctly (481 bins)
        m_feat_erb.resize(32);
        m_feat_spec.resize(96 * 2); // Neural path only uses 96 complex bins
        m_erb_mask.resize(32);
        m_prev_erb_mask.assign(32, 1.0f);
        m_df_coeffs.resize(96 * 5 * 2);

        m_emb.resize(512);
        m_e0.resize(m_enc_onnx->get_output_size("e0"));
        m_e1.resize(m_enc_onnx->get_output_size("e1"));
        m_e2.resize(m_enc_onnx->get_output_size("e2"));
        m_e3.resize(m_enc_onnx->get_output_size("e3"));
        m_c0.resize(m_enc_onnx->get_output_size("c0"));

        // 5. Initialize history for DF (order 5 + lookahead) - Expanded to 100
        m_spec_history.assign(100, std::vector<std::complex<float>>(481, 0.0f));
        m_history_idx = 0;

        m_initialized = true;
        std::cout << "[SUCCESS] DirectML Engine active on GPU." << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Engine init failed: " << e.what() << std::endl;
        return false;
    }
}

std::string DirectMLAudioEngine::get_device_name() const {
    return "Intel Arc B580 (DirectML Acceleration)";
}

void DirectMLAudioEngine::build_onnx_sessions() {
    auto path = std::filesystem::current_path();
    while (path.has_parent_path() && !std::filesystem::exists(path / "models" / "onnx")) {
        path = path.parent_path();
    }
    auto onnx_dir = path / "models" / "onnx";

    m_enc_onnx = std::make_unique<OnnxAdapter>();
    if (!m_enc_onnx->initialize((onnx_dir / "enc.onnx").string(), true)) {
        throw std::runtime_error("Failed to init ONNX GPU Session");
    }

    m_erb_dec_onnx = std::make_unique<OnnxAdapter>();
    m_erb_dec_onnx->initialize((onnx_dir / "erb_dec.onnx").string(), true);

    m_df_dec_onnx = std::make_unique<OnnxAdapter>();
    m_df_dec_onnx->initialize((onnx_dir / "df_dec.onnx").string(), true);
}

void DirectMLAudioEngine::process_frame(const float* input, float* output, size_t size) {
    if (!m_initialized || size != 480) return;

    // 1. STFT Analysis (CPU)
    std::vector<std::complex<float>> spec_t(481);
    m_dsp->analyze(input, spec_t.data());

    // Update history (Buffer of 100)
    m_spec_history[m_history_idx] = spec_t;

    // 2. Feature Extraction (Uses current frame t)
    m_features->compute_feat_erb(spec_t.data(), m_feat_erb.data());
    m_features->compute_feat_spec(spec_t.data(), m_feat_spec.data());

    // 3. Neural Inference (GPU)
    m_enc_onnx->set_input("feat_erb", m_feat_erb.data(), {1, 1, 1, 32});
    m_enc_onnx->set_input("feat_spec", m_feat_spec.data(), {1, 2, 1, 96});
    m_enc_onnx->run();

    m_enc_onnx->get_output("emb", m_emb.data(), m_emb.size());
    m_enc_onnx->get_output("e0", m_e0.data(), m_e0.size());
    m_enc_onnx->get_output("e1", m_e1.data(), m_e1.size());
    m_enc_onnx->get_output("e2", m_e2.data(), m_e2.size());
    m_enc_onnx->get_output("e3", m_e3.data(), m_e3.size());
    m_enc_onnx->get_output("c0", m_c0.data(), m_c0.size());

    m_erb_dec_onnx->set_input("emb", m_emb.data(), {1, 1, 512});
    m_erb_dec_onnx->set_input("e3", m_e3.data(), {1, 64, 1, 8});
    m_erb_dec_onnx->set_input("e2", m_e2.data(), {1, 64, 1, 8});
    m_erb_dec_onnx->set_input("e1", m_e1.data(), {1, 64, 1, 16});
    m_erb_dec_onnx->set_input("e0", m_e0.data(), {1, 64, 1, 32});
    m_erb_dec_onnx->run();
    m_erb_dec_onnx->get_output("m", m_erb_mask.data(), 32);

    if (m_df_enabled) {
        m_df_dec_onnx->set_input("emb", m_emb.data(), {1, 1, 512});
        m_df_dec_onnx->set_input("c0", m_c0.data(), {1, 64, 1, 96});
        m_df_dec_onnx->run();
        m_df_dec_onnx->get_output("coefs", m_df_coeffs.data(), 96 * 5 * 2);
    }

    // 4. Enhancement (Applied to frame t-2 due to lookahead)
    size_t lookahead_idx = (m_history_idx + 98) % 100;
    std::vector<std::complex<float>> enhanced_spec(481);

    // 4.1 Apply ERB Mask to delayed frame
    std::vector<std::complex<float>> spec_masked = m_spec_history[lookahead_idx];
    apply_mask(spec_masked.data(), m_erb_mask.data());

    // 4.2 Apply DF to Low Frequencies
    if (m_df_enabled) {
        std::vector<std::complex<float>> spec_df(96);
        compute_df_block(spec_df.data(), m_df_coeffs.data(), lookahead_idx);

        // 50/50 Blend as requested for better noise suppression while keeping quality
        for (size_t i = 0; i < 96; ++i) {
            enhanced_spec[i] = 0.5f * spec_df[i] + 0.5f * spec_masked[i];
        }
        std::copy(spec_masked.begin() + 96, spec_masked.end(), enhanced_spec.begin() + 96);
    } else {
        enhanced_spec = spec_masked;
    }

    // 5. ISTFT Synthesis (CPU)
    m_dsp->synthesize(enhanced_spec.data(), output);

    m_history_idx = (m_history_idx + 1) % 100;
}

void DirectMLAudioEngine::apply_mask(std::complex<float>* spec, const float* mask) {
    const auto& erb_bins = m_features->get_erb_bins();
    size_t spec_idx = 0;
    
    // More responsive smoothing (0.8) to catch sudden noises
    const float alpha = 0.8f;
    float effective_limit = std::min(m_attenuation_limit, 45.0f); // Allow up to 45dB
    float min_gain = std::pow(10.0f, -effective_limit / 20.0f);

    for (size_t b = 0; b < 32; ++b) {
        float raw_gain = std::clamp(mask[b], 0.0f, 1.0f);
        
        // Reduced voice floor from 0.05 to 0.01 to allow more noise removal
        raw_gain = 0.01f + 0.99f * raw_gain; 

        float gain = (alpha * raw_gain) + ((1.0f - alpha) * m_prev_erb_mask[b]);
        m_prev_erb_mask[b] = gain;

        gain = std::max(gain, min_gain);
        
        for (size_t i = 0; i < erb_bins[b]; ++i) {
            if (spec_idx < 481) {
                spec[spec_idx++] *= gain;
            }
        }
    }
}

void DirectMLAudioEngine::compute_df_block(std::complex<float>* spec_df_out, const float* df_coeffs, size_t ref_idx) {
    const size_t num_bins = 96;
    const size_t order = 5;
    const size_t imag_offset = order * num_bins;

    for (size_t k = 0; k < num_bins; ++k) {
        std::complex<float> sum(0, 0);
        for (size_t i = 0; i < order; ++i) {
            // coefs are [Complex, Order, Bins]
            size_t idx = i * num_bins + k;
            std::complex<float> coeff(df_coeffs[idx], df_coeffs[imag_offset + idx]);
            
            size_t h_idx = (ref_idx + 100 - i) % 100;
            std::complex<float> sample = m_spec_history[h_idx][k];
            
            sum += coeff * sample;
        }
        spec_df_out[k] = sum;
    }
}

void DirectMLAudioEngine::reset() {
    if (m_dsp) m_dsp->initialize();
    m_spec_history.assign(100, std::vector<std::complex<float>>(481, 0.0f));
    m_prev_erb_mask.assign(32, 1.0f);
    m_history_idx = 0;
}

} // namespace sa::infrastructure::directml_impl
