#include "silence_arc/infrastructure/sycl/sycl_graph_builder.h"
#include "silence_arc/infrastructure/sycl/linear_layer.h"
#include <fstream>
#include <iostream>

namespace sa::infrastructure::sycl_impl {

SyclGraphBuilder::SyclGraphBuilder(sycl::queue& queue, dnnl::engine& engine, SyclMemoryManager& mem_manager)
    : m_queue(queue), m_engine(engine), m_mem_manager(mem_manager) {}

bool SyclGraphBuilder::load_weights(const std::string& directory_path) {
    namespace fs = std::filesystem;
    if (!fs::exists(directory_path)) {
        std::cerr << "[ERROR] Weights directory not found: " << directory_path << std::endl;
        return false;
    }

    std::cout << "[INFO] Loading weights from: " << directory_path << std::endl;

    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.path().extension() == ".bin") {
            std::string name = entry.path().stem().string();
            size_t size = fs::file_size(entry.path());
            size_t num_elements = size / sizeof(float);

            float* dev_ptr = m_mem_manager.allocate_device<float>(num_elements);
            
            std::ifstream file(entry.path(), std::ios::binary);
            std::vector<float> host_buffer(num_elements);
            file.read(reinterpret_cast<char*>(host_buffer.data()), size);
            
            m_queue.memcpy(dev_ptr, host_buffer.data(), size).wait();

            m_weights[name] = dev_ptr;
            m_weight_sizes[name] = num_elements;
        }
    }

    std::cout << "[SUCCESS] Loaded " << m_weights.size() << " weight tensors." << std::endl;
    return true;
}

float* SyclGraphBuilder::get_weight(const std::string& name) {
    auto it = m_weights.find(name);
    return (it != m_weights.end()) ? it->second : nullptr;
}

std::unique_ptr<Layer> SyclGraphBuilder::build_linear(const std::string& name, size_t in_features, size_t out_features) {
    float* weight_ptr = get_weight(name + "_weight");
    float* bias_ptr = get_weight(name + "_bias");

    if (!weight_ptr) {
        std::cerr << "[ERROR] Weight not found for linear layer: " << name << std::endl;
        return nullptr;
    }

    return std::make_unique<SyclLinearLayer>(name, m_engine, m_queue, in_features, out_features, weight_ptr, bias_ptr);
}

} // namespace sa::infrastructure::sycl_impl
