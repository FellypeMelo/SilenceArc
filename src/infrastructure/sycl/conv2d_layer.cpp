#include "silence_arc/infrastructure/sycl/conv2d_layer.h"
#include <dnnl_sycl.hpp>
#include <iostream>

namespace sa::infrastructure::sycl_impl {

SyclConv2dLayer::SyclConv2dLayer(std::string name, dnnl::engine& engine, sycl::queue& queue, 
                                std::vector<size_t> src_shape, 
                                std::vector<size_t> weights_shape,
                                std::vector<size_t> strides,
                                std::vector<size_t> padding,
                                float* weights, float* bias)
    : Layer(std::move(name)), m_dnnl_engine(engine), m_sycl_queue(queue), m_has_bias(bias != nullptr) {

    using namespace dnnl;

    memory::dims src_dims;
    for (auto s : src_shape) src_dims.push_back(static_cast<memory::dim>(s));

    memory::dims weights_dims;
    for (auto s : weights_shape) weights_dims.push_back(static_cast<memory::dim>(s));

    memory::dims bias_dims = m_has_bias ? memory::dims{static_cast<memory::dim>(weights_shape[0])} : memory::dims{};

    memory::dims strd;
    for (auto s : strides) strd.push_back(static_cast<memory::dim>(s));

    memory::dims pad;
    for (auto s : padding) pad.push_back(static_cast<memory::dim>(s));

    // Calculate destination dimensions
    memory::dims dst_dims = {
        src_dims[0],
        weights_dims[0],
        (src_dims[2] + 2 * pad[0] - weights_dims[2]) / strd[0] + 1,
        (src_dims[3] + 2 * pad[1] - weights_dims[3]) / strd[1] + 1
    };

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
    auto bias_md = m_has_bias ? memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x) : memory::desc();
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    if (m_has_bias) {
        m_conv_pd = convolution_forward::primitive_desc(m_dnnl_engine, prop_kind::forward_inference,
                                                       algorithm::convolution_direct,
                                                       src_md, weights_md, bias_md, dst_md,
                                                       strd, pad, pad);
    } else {
        m_conv_pd = convolution_forward::primitive_desc(m_dnnl_engine, prop_kind::forward_inference,
                                                       algorithm::convolution_direct,
                                                       src_md, weights_md, dst_md,
                                                       strd, pad, pad);
    }

    m_weight_mem = sycl_interop::make_memory(weights_md, m_dnnl_engine, sycl_interop::memory_kind::usm, weights);
    if (m_has_bias) {
        m_bias_mem = sycl_interop::make_memory(bias_md, m_dnnl_engine, sycl_interop::memory_kind::usm, bias);
    }

    m_conv_prim = convolution_forward(m_conv_pd);
}

void SyclConv2dLayer::forward(const Tensor& input, Tensor& output) {
    using namespace dnnl;

    auto src_mem = sycl_interop::make_memory(m_conv_pd.src_desc(), m_dnnl_engine, sycl_interop::memory_kind::usm, const_cast<float*>(input.data()));
    auto dst_mem = sycl_interop::make_memory(m_conv_pd.dst_desc(), m_dnnl_engine, sycl_interop::memory_kind::usm, output.data());

    std::unordered_map<int, memory> args = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, m_weight_mem},
        {DNNL_ARG_DST, dst_mem}
    };
    if (m_has_bias) args[DNNL_ARG_BIAS] = m_bias_mem;

    stream s = sycl_interop::make_stream(m_dnnl_engine, m_sycl_queue);
    m_conv_prim.execute(s, args);
    s.wait();
}

} // namespace sa::infrastructure::sycl_impl
