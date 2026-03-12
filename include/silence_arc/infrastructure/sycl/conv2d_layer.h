#pragma once

#include "silence_arc/infrastructure/sycl/layer.h"
#include <dnnl.hpp>

namespace sa::infrastructure::sycl_impl {

/**
 * @brief oneDNN implementation of a 2D Convolution layer.
 */
class SyclConv2dLayer : public Layer {
public:
    SyclConv2dLayer(std::string name, dnnl::engine& engine, sycl::queue& queue, 
                    std::vector<size_t> src_shape, 
                    std::vector<size_t> weights_shape,
                    std::vector<size_t> strides,
                    std::vector<size_t> padding,
                    float* weights, float* bias = nullptr);

    void forward(const Tensor& input, Tensor& output) override;

private:
    dnnl::engine& m_dnnl_engine;
    sycl::queue& m_sycl_queue;
    
    dnnl::convolution_forward::primitive_desc m_conv_pd;
    dnnl::convolution_forward m_conv_prim;
    
    dnnl::memory m_weight_mem;
    dnnl::memory m_bias_mem;
    bool m_has_bias;
};

} // namespace sa::infrastructure::sycl_impl
