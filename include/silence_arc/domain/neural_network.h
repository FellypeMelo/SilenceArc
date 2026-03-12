#pragma once

#include <vector>
#include <string>

namespace sa::domain {

/**
 * @brief Domain interface for Neural Network inference.
 * Following Clean Architecture, this interface remains pure and decoupled from SYCL specifics.
 */
class NeuralNetworkModel {
public:
    virtual ~NeuralNetworkModel() = default;

    /**
     * @brief Initialize the model with weights.
     * @param weights_path Path to the weights directory.
     * @return true if successful, false otherwise.
     */
    virtual bool load_weights(const std::string& weights_path) = 0;

    /**
     * @brief Perform inference on ERB features (ERB mask only).
     * @param erb_features Input ERB features.
     * @param output_mask Output mask.
     */
    virtual void infer_erb(const float* erb_features, float* output_mask) = 0;

    /**
     * @brief Full inference: ERB mask + Deep Filtering coefficients.
     * @param erb_features Input ERB features (nb_erb floats).
     * @param df_features  Input complex bins (96 bins * 2 floats).
     * @param output_mask  Output ERB gain mask (nb_erb floats).
     * @param df_coefs     Output DF coefficients (get_df_coefs_count() floats).
     */
    virtual void infer(const float* erb_features,
                       const float* df_features,
                       float* output_mask,
                       float* df_coefs) = 0;

    /**
     * @brief Returns the number of float elements in the DF coefficients output.
     */
    virtual size_t get_df_coefs_count() const = 0;
};

} // namespace sa::domain
