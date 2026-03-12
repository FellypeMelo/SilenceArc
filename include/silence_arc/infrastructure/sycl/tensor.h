#pragma once

#include <vector>
#include <string>
#include <sycl/sycl.hpp>

namespace sa::infrastructure::sycl_impl {

/**
 * @brief Represents a multi-dimensional tensor in SYCL memory.
 */
class Tensor {
public:
    Tensor(std::vector<size_t> shape, float* data, bool is_device = true)
        : m_shape(shape), m_data(data), m_is_device(is_device) {
        m_total_elements = 1;
        for (auto s : shape) m_total_elements *= s;
    }

    const std::vector<size_t>& shape() const { return m_shape; }
    float* data() { return m_data; }
    const float* data() const { return m_data; }
    size_t total_elements() const { return m_total_elements; }
    bool is_device() const { return m_is_device; }

    size_t rank() const { return m_shape.size(); }

private:
    std::vector<size_t> m_shape;
    float* m_data;
    size_t m_total_elements;
    bool m_is_device;
};

} // namespace sa::infrastructure::sycl_impl
