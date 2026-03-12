#include "silence_arc/infrastructure/sycl/grouped_linear_layer.h"
#include <dnnl_sycl.hpp>
#include <iostream>

namespace sa::infrastructure::sycl_impl {

SyclGroupedLinearLayer::SyclGroupedLinearLayer(std::string name, dnnl::engine& engine, sycl::queue& queue, 
                                               size_t in_features, size_t out_features, size_t groups,
                                               float* weights, float* bias)
    : Layer(std::move(name)), m_dnnl_engine(engine), m_sycl_queue(queue),
      m_groups(groups), m_in_feat_per_group(in_features / groups), m_out_feat_per_group(out_features / groups),
      m_has_bias(bias != nullptr) {

    using namespace dnnl;

    // oneDNN handles grouped inner product via multi-dimensional dimensions
    // Shape: {groups, 1, out_per_group} for destination
    // Shape: {groups, out_per_group, in_per_group} for weights
    
    memory::dims src_dims = {static_cast<memory::dim>(m_groups), 1, static_cast<memory::dim>(m_in_feat_per_group)};
    memory::dims weights_dims = {static_cast<memory::dim>(m_groups), static_cast<memory::dim>(m_out_feat_per_group), static_cast<memory::dim>(m_in_feat_per_group)};
    memory::dims bias_dims = {static_cast<memory::dim>(m_groups), static_cast<memory::dim>(m_out_feat_per_group)};
    memory::dims dst_dims = {static_cast<memory::dim>(m_groups), 1, static_cast<memory::dim>(m_out_feat_per_group)};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::abc);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::abc);
    auto bias_md = m_has_bias ? memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::ab) : memory::desc();
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::abc);

    if (m_has_bias) {
        m_ip_pd = inner_product_forward::primitive_desc(m_dnnl_engine, prop_kind::forward_inference,
                                                       src_md, weights_md, bias_md, dst_md);
    } else {
        m_ip_pd = inner_product_forward::primitive_desc(m_dnnl_engine, prop_kind::forward_inference,
                                                       src_md, weights_md, dst_md);
    }

    m_weight_mem = sycl_interop::make_memory(weights_md, m_dnnl_engine, sycl_interop::memory_kind::usm, weights);
    if (m_has_bias) {
        m_bias_mem = sycl_interop::make_memory(bias_md, m_dnnl_engine, sycl_interop::memory_kind::usm, bias);
    }

    m_ip_prim = inner_product_forward(m_ip_pd);
}

void SyclGroupedLinearLayer::forward(const Tensor& input, Tensor& output) {
    using namespace dnnl;

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
