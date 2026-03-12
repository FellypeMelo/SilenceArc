#include "silence_arc/infrastructure/sycl/normalization_layer.h"
#include <cmath>

namespace sa::infrastructure::sycl_impl {

SyclNormalizationLayer::SyclNormalizationLayer(std::string name, sycl::queue& queue, SyclMemoryManager& mem_manager, size_t num_features)
    : Layer(std::move(name)), m_sycl_queue(queue), m_mem_manager(mem_manager), m_num_features(num_features) {
    
    m_running_mean = m_mem_manager.allocate_device<float>(num_features);
    m_running_var = m_mem_manager.allocate_device<float>(num_features);

    // Initial stats (match DeepFilterNet initialization)
    std::vector<float> h_mean(num_features, -60.0f);
    std::vector<float> h_var(num_features, 0.001f);
    
    m_sycl_queue.memcpy(m_running_mean, h_mean.data(), num_features * sizeof(float));
    m_sycl_queue.memcpy(m_running_var, h_var.data(), num_features * sizeof(float)).wait();
}

void SyclNormalizationLayer::forward(const Tensor& input, Tensor& output) {
    forward_with_update(input, output, 1.0f); // No update (alpha=1.0)
}

void SyclNormalizationLayer::forward_with_update(const Tensor& input, Tensor& output, float alpha) {
    float* in_ptr = const_cast<float*>(input.data());
    float* out_ptr = output.data();
    float* mean_ptr = m_running_mean;
    float* var_ptr = m_running_var;
    size_t n = m_num_features;

    m_sycl_queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            float val = in_ptr[i];
            
            // 1. Log scale
            float lp = std::log10(val + 1e-10f) * 10.0f;
            
            // 2. Update running stats
            float m = mean_ptr[i];
            m = lp * (1.0f - alpha) + m * alpha;
            mean_ptr[i] = m;
            
            // 3. Normalize
            out_ptr[i] = (lp - m) / 20.0f;
        });
    }).wait();
}

} // namespace sa::infrastructure::sycl_impl
