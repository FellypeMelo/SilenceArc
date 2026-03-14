#pragma once

#include <vector>
#include <complex>
#include <cmath>

namespace sa::infrastructure::directml_impl {

/**
 * @brief Extract ERB and Complex features for DeepFilterNet3.
 * Ported from DeepFilterNet libDF (Rust).
 */
class FeatureExtractor {
public:
    FeatureExtractor(size_t sr = 48000, size_t fft_size = 960, size_t nb_erb = 32);

    void initialize();
    
    /**
     * @brief Compute log-ERB features with exponential mean normalization.
     */
    void compute_feat_erb(const std::complex<float>* spec, float* feat_erb_out);

    /**
     * @brief Compute complex features with unit normalization.
     */
    void compute_feat_spec(const std::complex<float>* spec, float* feat_spec_out);

    const std::vector<size_t>& get_erb_bins() const { return m_erb_bins; }

private:
    float freq2erb(float freq_hz);
    float erb2freq(float n_erb);
    void init_erb_fb();
    void init_norm_states();

    size_t m_sr;
    size_t m_fft_size;
    size_t m_nb_erb;
    
    std::vector<size_t> m_erb_bins;
    std::vector<float> m_mean_norm_state;
    std::vector<float> m_unit_norm_state;

    // Constantes de tempo baseadas no libDF (Rust)
    // alpha = exp(-hop_size / (sr * tau))
    // Para 480 hop, 48000 sr:
    // tau = 2.0s (mean) -> alpha = 0.995
    // tau = 0.5s (unit) -> alpha = 0.98
    const float m_alpha_mean = 0.995f; 
    const float m_alpha_unit = 0.98f;
};

} // namespace sa::infrastructure::directml_impl
