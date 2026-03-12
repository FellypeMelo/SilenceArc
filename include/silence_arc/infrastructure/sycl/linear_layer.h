#pragma once

#include "silence_arc/infrastructure/sycl/layer.h"
#include <dnnl.hpp>

namespace sa::infrastructure::sycl_impl {

/**
 * @brief oneDNN implementation of a Linear (Fully Connected) layer.
 */
class SyclLinearLayer : public Layer {
public:
    SyclLinearLayer(std::string name, dnnl::engine& engine, sycl::queue& queue, 
                    size_t in_features, size_t out_features, 
                    float* weights, float* bias = nullptr);

    void forward(const Tensor& input, Tensor& output) override;

private:
    dnnl::engine& m_dnnl_engine;
    sycl::queue& m_sycl_queue;
    
    size_t m_in_features;
    size_t m_out_features;

    dnnl::inner_product_forward::primitive_desc m_ip_pd;
    dnnl::inner_product_forward m_ip_prim;
    
    dnnl::memory m_weight_mem;
    dnnl::memory m_bias_mem;
    bool m_has_bias;
};

} // namespace sa::infrastructure::sycl_impl
