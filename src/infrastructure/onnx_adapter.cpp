#include "silence_arc/infrastructure/onnx_adapter.h"
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <iostream>
#include <algorithm>
#include <map>

namespace sa::infrastructure {

struct OnnxAdapter::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "SilenceArc"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_names_ptr;
    std::vector<const char*> output_names_ptr;

    // Use a map to hold input values, and a vector for the final pointers
    std::map<std::string, Ort::Value> input_values;
    std::vector<Ort::Value> output_tensors;
};

OnnxAdapter::OnnxAdapter() : m_impl(std::make_unique<Impl>()) {}

OnnxAdapter::~OnnxAdapter() {}

bool OnnxAdapter::initialize(const std::string& model_path, bool use_gpu) {
    try {
        if (use_gpu) {
            m_impl->session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            m_impl->session_options.DisableMemPattern();
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(m_impl->session_options, 0));
        }

        std::wstring model_path_w(model_path.begin(), model_path.end());
        m_impl->session = std::make_unique<Ort::Session>(m_impl->env, model_path_w.c_str(), m_impl->session_options);

        // Verify active providers
        auto providers = Ort::GetAvailableProviders();
        bool dml_active = false;
        std::cout << "[DEBUG] Available Providers: ";
        for (const auto& p : providers) {
            std::cout << p << " ";
            if (p == "DmlExecutionProvider") dml_active = true;
        }
        std::cout << std::endl;

        if (use_gpu && !dml_active) {
            std::cerr << "[ERROR] DirectML requested but not available!" << std::endl;
            return false;
        }

        Ort::AllocatorWithDefaultOptions allocator;
        std::cout << "[DEBUG] Model: " << model_path << std::endl;
        for (size_t i = 0; i < m_impl->session->GetInputCount(); i++) {
            auto name = m_impl->session->GetInputNameAllocated(i, allocator);
            m_impl->input_names.push_back(name.get());
            
            auto type_info = m_impl->session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            
            std::cout << "  Input " << i << ": " << name.get() << " Shape: [";
            for (size_t j = 0; j < shape.size(); j++) std::cout << shape[j] << (j == shape.size() - 1 ? "" : ", ");
            std::cout << "]" << std::endl;
        }
        for (size_t i = 0; i < m_impl->session->GetOutputCount(); i++) {
            auto name = m_impl->session->GetOutputNameAllocated(i, allocator);
            m_impl->output_names.push_back(name.get());

            auto type_info = m_impl->session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();

            std::cout << "  Output " << i << ": " << name.get() << " Shape: [";
            for (size_t j = 0; j < shape.size(); j++) std::cout << shape[j] << (j == shape.size() - 1 ? "" : ", ");
            std::cout << "]" << std::endl;
        }

        for (const auto& name : m_impl->input_names) m_impl->input_names_ptr.push_back(name.c_str());
        for (const auto& name : m_impl->output_names) m_impl->output_names_ptr.push_back(name.c_str());

        // Initial dummy run or just query shapes? 
        // Actually, we can just return the counts for allocation.
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] ONNX Adapter Init Error: " << e.what() << std::endl;
        return false;
    }
}

void OnnxAdapter::set_input(const std::string& name, const float* data, const std::vector<int64_t>& shape) {
    size_t size = 1;
    for (auto d : shape) size *= d;
    
    // Create tensor referencing external data
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        m_impl->memory_info, const_cast<float*>(data), size, shape.data(), shape.size());
    
    m_impl->input_values.insert_or_assign(name, std::move(tensor));
}

size_t OnnxAdapter::get_output_size(const std::string& name) {
    for (size_t i = 0; i < m_impl->session->GetOutputCount(); ++i) {
        if (m_impl->output_names[i] == name) {
            auto type_info = m_impl->session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            size_t count = 1;
            for (auto d : shape) {
                if (d > 0) count *= d;
            }
            return count;
        }
    }
    return 0;
}

void OnnxAdapter::run() {
    std::vector<Ort::Value> run_inputs;
    for (const auto& name : m_impl->input_names) {
        auto it = m_impl->input_values.find(name);
        if (it == m_impl->input_values.end()) throw std::runtime_error("Input not set: " + name);
        
        // We must pass the Ort::Value by reference or move. 
        // In ORT C++ API, we usually build a vector of Ort::Value.
        // Since Ort::Value move constructor is efficient, we can do this:
        // (But we need to keep the original values alive, which input_values map does)
    }

    // Direct pointers to values in the map are not available, so we build a temporary vector of Ort::Value
    // that wraps the pointers to the same underlying tensors.
    std::vector<Ort::Value> input_tensors_to_run;
    for (const auto& name : m_impl->input_names) {
        auto& val = m_impl->input_values.at(name);
        // Create a new Ort::Value that shares ownership if possible, 
        // or just move it if we don't mind rebuilding the map.
        // Simplest: reconstruct the list from the map every time.
        input_tensors_to_run.push_back(std::move(m_impl->input_values.at(name)));
    }

    m_impl->output_tensors = m_impl->session->Run(
        Ort::RunOptions{nullptr},
        m_impl->input_names_ptr.data(),
        input_tensors_to_run.data(),
        input_tensors_to_run.size(),
        m_impl->output_names_ptr.data(),
        m_impl->output_names_ptr.size()
    );
    
    // Clear map after run as tensors were moved
    m_impl->input_values.clear();
}

void OnnxAdapter::get_output(const std::string& name, float* data, size_t size) {
    for (size_t i = 0; i < m_impl->output_names.size(); ++i) {
        if (m_impl->output_names[i] == name) {
            const float* out_data = m_impl->output_tensors[i].GetTensorData<float>();
            std::copy(out_data, out_data + size, data);
            return;
        }
    }
}

} // namespace sa::infrastructure
