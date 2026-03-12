#pragma once

#include "silence_arc/infrastructure/sycl/layer.h"
#include <dnnl.hpp>

namespace sa::infrastructure::sycl_impl {

/**
 * @brief oneDNN implementation of a Binary Add layer.
 * Performs element-wise addition of two tensors.
 */
class SyclBinaryAddLayer : public Layer {
public:
    SyclBinaryAddLayer(std::string name, dnnl::engine& engine, sycl::queue& queue, 
                       std::vector<size_t> shape);

    void forward(const Tensor& input_a, const Tensor& input_b, Tensor& output);
    
    // Default Layer interface implementation (not applicable for binary)
    void forward(const Tensor& input, Tensor& output) override {
        // Not used for binary add
    }

private:
    dnnl::engine& m_dnnl_engine;
    sycl::queue& m_sycl_queue;
    
    std::vector<size_t> m_shape;

    dnnl::binary::primitive_desc m_binary_pd;
    dnnl::binary m_binary_prim;
};

} // namespace sa::infrastructure::sycl_impl
