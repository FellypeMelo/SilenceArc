#include "silence_arc/infrastructure/feature_extractor.h"
#include <algorithm>
#include <numeric>
#include <iostream>

namespace sa::infrastructure::directml_impl {

FeatureExtractor::FeatureExtractor(size_t sr, size_t fft_size, size_t nb_erb)
    : m_sr(sr), m_fft_size(fft_size), m_nb_erb(nb_erb) {}

void FeatureExtractor::initialize() {
    init_erb_fb();
    init_norm_states();
}

float FeatureExtractor::freq2erb(float freq_hz) {
    return 9.265f * std::log1p(freq_hz / (24.7f * 9.265f));
}

float FeatureExtractor::erb2freq(float n_erb) {
    return 24.7f * 9.265f * (std::exp(n_erb / 9.265f) - 1.0f);
}

void FeatureExtractor::init_erb_fb() {
    float nyq_freq = m_sr / 2.0f;
    float freq_width = (float)m_sr / m_fft_size;
    float erb_low = freq2erb(0.0f);
    float erb_high = freq2erb(nyq_freq);
    
    m_erb_bins.assign(m_nb_erb, 0);
    float step = (erb_high - erb_low) / m_nb_erb;
    int min_nb_freqs = 2;
    int prev_freq = 0;
    int freq_over = 0;

    for (size_t i = 1; i <= m_nb_erb; ++i) {
        float f = erb2freq(erb_low + i * step);
        int fb = (int)std::round(f / freq_width);
        int nb_freqs = fb - prev_freq - freq_over;
        
        if (nb_freqs < min_nb_freqs) {
            freq_over = min_nb_freqs - nb_freqs;
            nb_freqs = min_nb_freqs;
        } else {
            freq_over = 0;
        }
        
        m_erb_bins[i - 1] = (size_t)nb_freqs;
        prev_freq = fb;
    }

    m_erb_bins[m_nb_erb - 1] += 1; // Last bin adjustment
    size_t total_bins = std::accumulate(m_erb_bins.begin(), m_erb_bins.end(), 0ULL);
    size_t target_bins = m_fft_size / 2 + 1;
    
    if (total_bins > target_bins) {
        m_erb_bins[m_nb_erb - 1] -= (total_bins - target_bins);
    }

    std::cout << "[DEBUG] ERB Filterbank initialized. Total frequency bins: " << target_bins << std::endl;
    std::cout << "  Bins per band: ";
    for (size_t b = 0; b < m_nb_erb; ++b) std::cout << m_erb_bins[b] << (b == m_nb_erb - 1 ? "" : ", ");
    std::cout << std::endl;
}

void FeatureExtractor::init_norm_states() {
    const float MEAN_NORM_INIT_MIN = -60.0f;
    const float MEAN_NORM_INIT_MAX = -90.0f;
    const float UNIT_NORM_INIT_MIN = 0.001f;
    const float UNIT_NORM_INIT_MAX = 0.0001f;

    m_mean_norm_state.resize(m_nb_erb);
    float step_mean = (MEAN_NORM_INIT_MAX - MEAN_NORM_INIT_MIN) / (m_nb_erb - 1);
    for (size_t i = 0; i < m_nb_erb; ++i) {
        m_mean_norm_state[i] = MEAN_NORM_INIT_MIN + i * step_mean;
    }

    size_t freq_size = m_fft_size / 2 + 1;
    m_unit_norm_state.resize(freq_size);
    float step_unit = (UNIT_NORM_INIT_MAX - UNIT_NORM_INIT_MIN) / (freq_size - 1);
    for (size_t i = 0; i < freq_size; ++i) {
        m_unit_norm_state[i] = UNIT_NORM_INIT_MIN + i * step_unit;
    }
}

void FeatureExtractor::compute_feat_erb(const std::complex<float>* spec, float* feat_erb_out) {
    size_t spec_idx = 0;
    for (size_t b = 0; b < m_nb_erb; ++b) {
        float band_energy = 0.0f;
        size_t band_size = m_erb_bins[b];
        for (size_t i = 0; i < band_size; ++i) {
            float mag_sq = std::norm(spec[spec_idx++]);
            band_energy += mag_sq;
        }
        band_energy /= (float)band_size;

        // Convert to dB-like scale
        float log_erb = 10.0f * std::log10(band_energy + 1e-10f);

        // Slow exponential moving average for mean normalization
        m_mean_norm_state[b] = (0.95f * m_mean_norm_state[b]) + (0.05f * log_erb);

        // DeepFilterNet3 typically expects features around -1.0 to 1.0 range.
        // We'll use a more standard 20dB range for the denominator.
        feat_erb_out[b] = (log_erb - m_mean_norm_state[b]) / 20.0f;
    }
}

void FeatureExtractor::compute_feat_spec(const std::complex<float>* spec, float* feat_spec_out) {
    size_t nb_df = 96;
    float* out_re = feat_spec_out;
    float* out_im = feat_spec_out + nb_df;

    for (size_t i = 0; i < nb_df; ++i) {
        float mag_sq = std::norm(spec[i]);
        // Per-bin unit normalization for the complex spectrum features
        m_unit_norm_state[i] = (0.95f * m_unit_norm_state[i]) + (0.05f * mag_sq);

        float scale = 1.0f / std::sqrt(m_unit_norm_state[i] + 1e-10f);

        out_re[i] = spec[i].real() * scale;
        out_im[i] = spec[i].imag() * scale;
    }
}
} // namespace sa::infrastructure::directml_impl
