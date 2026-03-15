#include "silence_arc/infrastructure/sycl_kernels.h"
#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>

namespace sa::infrastructure::sycl_impl {

struct SyclKernels::Impl {
    sycl::queue q;
    bool initialized = false;

    Impl() : q(sycl::default_selector_v) {}
};

SyclKernels::SyclKernels() : m_impl(std::make_unique<Impl>()) {}
SyclKernels::~SyclKernels() = default;

bool SyclKernels::initialize() {
    try {
        std::cout << "[INFO] SYCL Device: " << m_impl->q.get_device().get_info<sycl::info::device::name>() << std::endl;
        m_impl->initialized = true;
        return true;
    } catch (const sycl::exception& e) {
        std::cerr << "[ERROR] SYCL Init failed: " << e.what() << std::endl;
        return false;
    }
}

void SyclKernels::apply_mask(std::complex<float>* spec, const float* mask, const std::vector<size_t>& erb_bins, float attenuation_limit) {
    if (!m_impl->initialized) return;

    size_t num_bins = 481;
    size_t num_erb = 32;
    float min_gain = std::pow(10.0f, -attenuation_limit / 20.0f);

    {
        sycl::buffer<std::complex<float>, 1> b_spec(spec, sycl::range<1>(num_bins));
        sycl::buffer<float, 1> b_mask(mask, sycl::range<1>(num_erb));
        sycl::buffer<size_t, 1> b_erb_bins(erb_bins.data(), sycl::range<1>(num_erb));

        m_impl->q.submit([&](sycl::handler& h) {
            auto acc_spec = b_spec.get_access<sycl::access::mode::read_write>(h);
            auto acc_mask = b_mask.get_access<sycl::access::mode::read>(h);
            auto acc_erb_bins = b_erb_bins.get_access<sycl::access::mode::read>(h);

            h.parallel_for(sycl::range<1>(num_bins), [=](sycl::id<1> idx) {
                size_t k = idx[0];
                
                // Find which ERB bin this linear bin k belongs to
                size_t current_erb = 0;
                size_t accumulated_bins = 0;
                for (size_t b = 0; b < num_erb; ++b) {
                    accumulated_bins += acc_erb_bins[b];
                    if (k < accumulated_bins) {
                        current_erb = b;
                        break;
                    }
                }

                float gain = acc_mask[current_erb];
                if (gain < 0.0f) gain = 0.0f;
                if (gain > 1.0f) gain = 1.0f;
                if (gain < min_gain) gain = min_gain;

                acc_spec[k] *= gain;
            });
        });
    }
    m_impl->q.wait();
}

void SyclKernels::compute_df_block(std::complex<float>* spec_df_out, 
                                 const float* df_coeffs, 
                                 const std::vector<std::vector<std::complex<float>>>& spec_history,
                                 size_t history_idx) {
    if (!m_impl->initialized) return;

    size_t num_df_bins = 96;
    size_t order = 5;

    // Flatten history for buffer transfer
    std::vector<std::complex<float>> flat_history(order * 481);
    for (size_t i = 0; i < order; ++i) {
        size_t h_idx = (history_idx + order - i) % order;
        std::copy(spec_history[h_idx].begin(), spec_history[h_idx].end(), flat_history.begin() + i * 481);
    }

    {
        sycl::buffer<std::complex<float>, 1> b_out(spec_df_out, sycl::range<1>(num_df_bins));
        sycl::buffer<float, 1> b_coeffs(df_coeffs, sycl::range<1>(num_df_bins * order * 2));
        sycl::buffer<std::complex<float>, 1> b_history(flat_history.data(), sycl::range<1>(order * 481));

        m_impl->q.submit([&](sycl::handler& h) {
            auto acc_out = b_out.get_access<sycl::access::mode::write>(h);
            auto acc_coeffs = b_coeffs.get_access<sycl::access::mode::read>(h);
            auto acc_history = b_history.get_access<sycl::access::mode::read>(h);

            h.parallel_for(sycl::range<1>(num_df_bins), [=](sycl::id<1> idx) {
                size_t k = idx[0];
                std::complex<float> sum(0, 0);
                
                for (size_t i = 0; i < order; ++i) {
                    // coeffs are (96, order, 2)
                    size_t c_base = (k * order * 2) + (i * 2);
                    std::complex<float> coeff(acc_coeffs[c_base], acc_coeffs[c_base + 1]);
                    
                    // history is (order, 481)
                    std::complex<float> sample = acc_history[i * 481 + k];
                    
                    sum += coeff * sample;
                }
                acc_out[k] = sum;
            });
        });
    }
    m_impl->q.wait();
}

} // namespace sa::infrastructure::sycl_impl
