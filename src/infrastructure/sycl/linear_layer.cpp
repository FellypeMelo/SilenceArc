#include "silence_arc/infrastructure/sycl/linear_layer.h"
#include <dnnl_sycl.hpp>
#include <iostream>

namespace sa::infrastructure::sycl_impl {

SyclLinearLayer::SyclLinearLayer(std::string name, dnnl::engine& engine, sycl::queue& queue, 
                                size_t in_features, size_t out_features, 
                                float* weights, float* bias)
    : Layer(std::move(name)), m_dnnl_engine(engine), m_sycl_queue(queue),
      m_in_features(in_features), m_out_features(out_features), m_has_bias(bias != nullptr) {

    using namespace dnnl;

    // 1. Create Memory Descriptors
    memory::dims src_dims = {1, static_cast<memory::dim>(in_features)};
    memory::dims weights_dims = {static_cast<memory::dim>(out_features), static_cast<memory::dim>(in_features)};
    memory::dims bias_dims = {static_cast<memory::dim>(out_features)};
    memory::dims dst_dims = {1, static_cast<memory::dim>(out_features)};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nc);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oi);
    auto bias_md = m_has_bias ? memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x) : memory::desc();
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nc);

    // 2. Create Primitive Descriptor
    if (m_has_bias) {
        m_ip_pd = inner_product_forward::primitive_desc(m_dnnl_engine, prop_kind::forward_inference,
                                                       src_md, weights_md, bias_md, dst_md);
    } else {
        m_ip_pd = inner_product_forward::primitive_desc(m_dnnl_engine, prop_kind::forward_inference,
                                                       src_md, weights_md, dst_md);
    }

    // 3. Create Memory Objects (Using USM pointers)
    m_weight_mem = sycl_interop::make_memory(weights_md, m_dnnl_engine, sycl_interop::memory_kind::usm, weights);
    if (m_has_bias) {
        m_bias_mem = sycl_interop::make_memory(bias_md, m_dnnl_engine, sycl_interop::memory_kind::usm, bias);
    }

    // 4. Create Primitive
    m_ip_prim = inner_product_forward(m_ip_pd);
}

void SyclLinearLayer::forward(const Tensor& input, Tensor& output) {
    using namespace dnnl;

    // Create temporary memory objects for input/output USM pointers
    // Cast const to void* for oneDNN interop
    auto src_mem = sycl_interop::make_memory(m_ip_pd.src_desc(), m_dnnl_engine, sycl_interop::memory_kind::usm, const_cast<float*>(input.data()));
    auto dst_mem = sycl_interop::make_memory(m_ip_pd.dst_desc(), m_dnnl_engine, sycl_interop::memory_kind::usm, output.data());

    std::unordered_map<int, memory> args = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, m_weight_mem},
        {DNNL_ARG_DST, dst_mem}
    };
    if (m_has_bias) args[DNNL_ARG_BIAS] = m_bias_mem;

    stream s = sycl_interop::make_stream(m_dnnl_engine, m_sycl_queue);
    m_ip_prim.execute(s, args);
    s.wait();
}

} // namespace sa::infrastructure::sycl_impl
