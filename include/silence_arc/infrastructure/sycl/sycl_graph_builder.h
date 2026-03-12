#pragma once

#include "silence_arc/infrastructure/sycl/sycl_memory_manager.h"
#include "silence_arc/infrastructure/sycl/layer.h"
#include <string>
#include <map>
#include <filesystem>
#include <dnnl.hpp>

namespace sa::infrastructure::sycl_impl {

/**
 * @brief Automates the construction of the neural network graph from weight files.
 * Maps weight file naming conventions to oneDNN primitives.
 */
class SyclGraphBuilder {
public:
    SyclGraphBuilder(sycl::queue& queue, dnnl::engine& engine, SyclMemoryManager& mem_manager);
    ~SyclGraphBuilder() = default;

    /**
     * @brief Scans a directory for .bin weight files and loads them into memory.
     */
    bool load_weights(const std::string& directory_path);

    /**
     * @brief Creates a Linear layer wrapper using oneDNN.
     * @param name Prefix of the weight files (e.g., "enc_df_fc_emb_0").
     */
    std::unique_ptr<Layer> build_linear(const std::string& name, size_t in_features, size_t out_features);

    /**
     * @brief Retrieves a loaded weight buffer by name.
     */
    float* get_weight(const std::string& name);

private:
    sycl::queue& m_queue;
    dnnl::engine& m_engine;
    SyclMemoryManager& m_mem_manager;

    std::map<std::string, float*> m_weights;
    std::map<std::string, size_t> m_weight_sizes;
};

} // namespace sa::infrastructure::sycl_impl
