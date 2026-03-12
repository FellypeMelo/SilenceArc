#pragma once

#include "silence_arc/infrastructure/sycl/tensor.h"
#include <string>
#include <memory>

namespace sa::infrastructure::sycl_impl {

/**
 * @brief Abstract base class for a neural network layer in SYCL.
 */
class Layer {
public:
    explicit Layer(std::string name) : m_name(std::move(name)) {}
    virtual ~Layer() = default;

    /**
     * @brief Executes the forward pass of the layer.
     * @param input Input tensor.
     * @param output Output tensor.
     */
    virtual void forward(const Tensor& input, Tensor& output) = 0;

    const std::string& name() const { return m_name; }

protected:
    std::string m_name;
};

} // namespace sa::infrastructure::sycl_impl
