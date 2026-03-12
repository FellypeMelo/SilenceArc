#include "silence_arc/infrastructure/sycl/binary_add_layer.h"
#include <dnnl_sycl.hpp>
#include <iostream>

namespace sa::infrastructure::sycl_impl {

SyclBinaryAddLayer::SyclBinaryAddLayer(std::string name, dnnl::engine& engine, sycl::queue& queue, 
                                       std::vector<size_t> shape)
    : Layer(std::move(name)), m_dnnl_engine(engine), m_sycl_queue(queue), m_shape(shape) {

    using namespace dnnl;

    memory::dims dims;
    for (auto s : shape) dims.push_back(static_cast<memory::dim>(s));

    auto md = memory::desc(dims, memory::data_type::f32, 
                           shape.size() == 4 ? memory::format_tag::nchw : 
                           (shape.size() == 2 ? memory::format_tag::nc : memory::format_tag::abc));

    m_binary_pd = binary::primitive_desc(m_dnnl_engine, algorithm::binary_add, md, md, md);
    m_binary_prim = binary(m_binary_pd);
}

void SyclBinaryAddLayer::forward(const Tensor& input_a, const Tensor& input_b, Tensor& output) {
    using namespace dnnl;

    auto src0_mem = sycl_interop::make_memory(m_binary_pd.src0_desc(), m_dnnl_engine, sycl_interop::memory_kind::usm, const_cast<float*>(input_a.data()));
    auto src1_mem = sycl_interop::make_memory(m_binary_pd.src1_desc(), m_dnnl_engine, sycl_interop::memory_kind::usm, const_cast<float*>(input_b.data()));
    auto dst_mem = sycl_interop::make_memory(m_binary_pd.dst_desc(), m_dnnl_engine, sycl_interop::memory_kind::usm, output.data());

    std::unordered_map<int, memory> args = {
        {DNNL_ARG_SRC_0, src0_mem},
        {DNNL_ARG_SRC_1, src1_mem},
        {DNNL_ARG_DST, dst_mem}
    };

    stream s = sycl_interop::make_stream(m_dnnl_engine, m_sycl_queue);
    m_binary_prim.execute(s, args);
    s.wait();
}

} // namespace sa::infrastructure::sycl_impl
