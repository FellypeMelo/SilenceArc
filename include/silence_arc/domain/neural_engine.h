#pragma once

#include <string>
#include <vector>

namespace sa::domain {

/**
 * @brief Abstract interface for Neural Network inference.
 * Decoupled from specific backend implementations (oneDNN, ONNX, etc.).
 */
class INeuralEngine {
public:
    virtual ~INeuralEngine() = default;

    /**
     * @brief Load weights from a file or directory.
     */
    virtual bool load_weights(const std::string& weights_path) = 0;

    /**
     * @brief Perform lightweight inference (e.g., ERB mask only).
     */
    virtual void infer_erb(const float* erb_features, float* output_mask) = 0;

    /**
     * @brief Perform full model inference.
     */
    virtual void infer(const float* erb_features,
                       const float* df_features,
                       float* output_mask,
                       float* df_coefs) = 0;

    /**
     * @brief Get the count of Deep Filtering coefficients expected by the model.
     */
    virtual size_t get_df_coefs_count() const = 0;

    /**
     * @brief Reset internal states (e.g., GRU hidden states).
     */
    virtual void reset() = 0;
};

} // namespace sa::domain
