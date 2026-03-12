#pragma once

#include "silence_arc/infrastructure/sycl/layer.h"
#include "silence_arc/infrastructure/sycl/sycl_memory_manager.h"
#include <vector>

namespace sa::infrastructure::sycl_impl {

/**
 * @brief Custom SYCL implementation of ERB log-scale normalization.
 * Matches DeepFilterNet3's internal feature normalization.
 */
class SyclNormalizationLayer : public Layer {
public:
    SyclNormalizationLayer(std::string name, sycl::queue& queue, SyclMemoryManager& mem_manager, size_t num_features);

    void forward(const Tensor& input, Tensor& output) override;

    /**
     * @brief Specialized forward pass with running mean/variance update.
     */
    void forward_with_update(const Tensor& input, Tensor& output, float alpha = 0.9f);

private:
    sycl::queue& m_sycl_queue;
    SyclMemoryManager& m_mem_manager;
    size_t m_num_features;

    float* m_running_mean;
    float* m_running_var;
};

} // namespace sa::infrastructure::sycl_impl
