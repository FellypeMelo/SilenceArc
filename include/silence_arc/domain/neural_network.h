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
     * @brief Perform inference on ERB features.
     * @param erb_features Input ERB features.
     * @param output_mask Output mask.
     */
    virtual void infer_erb(const float* erb_features, float* output_mask) = 0;
};

} // namespace sa::domain
