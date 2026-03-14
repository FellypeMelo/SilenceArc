#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace sa::infrastructure {

/**
 * @brief Adapter for ONNX Runtime with DirectML support.
 * Replaces OpenVinoAdapter for better stability on Intel Arc Windows drivers.
 */
class OnnxAdapter {
public:
    OnnxAdapter();
    ~OnnxAdapter();

    /**
     * @brief Initialize the ONNX session.
     * @param model_path Path to .onnx file.
     * @param use_gpu If true, tries to use DirectML.
     */
    bool initialize(const std::string& model_path, bool use_gpu = true);

    void set_input(const std::string& name, const float* data, const std::vector<int64_t>& shape);
    void get_output(const std::string& name, float* data, size_t size);
    size_t get_output_size(const std::string& name);
    void run();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace sa::infrastructure
