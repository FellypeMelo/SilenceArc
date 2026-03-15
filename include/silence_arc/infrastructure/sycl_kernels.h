#pragma once

#include <complex>
#include <vector>

namespace sa::infrastructure::sycl_impl {

/**
 * @brief Low-level SYCL-based optimizations for Intel Hardware.
 * These kernels provide hardware acceleration for DSP tasks that remain on the CPU in the base DirectML engine.
 */
class SyclKernels {
public:
    SyclKernels();
    ~SyclKernels();

    bool initialize();

    /**
     * @brief Parallel ERB Mask Application on GPU.
     */
    void apply_mask(std::complex<float>* spec, const float* mask, const std::vector<size_t>& erb_bins, float attenuation_limit);

    /**
     * @brief Parallel Deep Filtering (Complex MAC) on GPU.
     */
    void compute_df_block(std::complex<float>* spec_df_out, 
                         const float* df_coeffs, 
                         const std::vector<std::vector<std::complex<float>>>& spec_history,
                         size_t history_idx);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace sa::infrastructure::sycl_impl
