#include "silence_arc/infrastructure/sycl/native_sycl_engine.h"
#include <dnnl_sycl.hpp>
#include <iostream>
#include <filesystem>

namespace sa::infrastructure::sycl_impl {

NativeSyclEngine::NativeSyclEngine() {}

NativeSyclEngine::~NativeSyclEngine() {}

bool NativeSyclEngine::initialize() {
    if (m_initialized) return true;

    try {
        // 1. SYCL Queue (Explicitly select Intel Arc if possible)
        sycl::device device;
        bool found = false;
        auto platforms = sycl::platform::get_platforms();
        for (auto& platform : platforms) {
            auto devices = platform.get_devices();
            for (auto& dev : devices) {
                std::string name = dev.get_info<sycl::info::device::name>();
                if (dev.is_gpu() && name.find("Arc") != std::string::npos) {
                    device = dev;
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        if (!found) device = sycl::device(sycl::default_selector_v);

        m_queue = std::make_unique<sycl::queue>(device, sycl::property::queue::in_order());
        m_dnnl_engine = std::make_unique<dnnl::engine>(dnnl::sycl_interop::make_engine(m_queue->get_device(), m_queue->get_context()));
        
        // 2. Core Managers
        m_mem_manager = std::make_unique<SyclMemoryManager>(*m_queue);
        m_dsp = std::make_unique<SyclDspEngine>(*m_queue, *m_mem_manager);
        m_dsp->initialize();
        
        m_graph_builder = std::make_unique<SyclGraphBuilder>(*m_queue, *m_dnnl_engine, *m_mem_manager);

        // 3. Load Weights
        auto path = std::filesystem::current_path();
        if (path.filename() == "build") path = path.parent_path();
        auto weights_dir = path / "models" / "df3_weights";
        
        if (!m_graph_builder->load_weights(weights_dir.string())) {
            return false;
        }

        // 4. Build Graph
        build_graph();

        m_initialized = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] NativeSyclEngine init failed: " << e.what() << std::endl;
        return false;
    }
}

std::string NativeSyclEngine::get_device_name() const {
    if (!m_queue) return "Not Initialized";
    return m_queue->get_device().get_info<sycl::info::device::name>();
}

void NativeSyclEngine::build_graph() {
    // Phase 4 will implement the full path here
    std::cout << "[INFO] Building Neural Graph (Encoder + Decoders)..." << std::endl;
}

void NativeSyclEngine::process_frame(const float* input, float* output, size_t size) {
    if (!m_initialized || size != 480) return;

    // TODO: Phase 4 Inference Pipeline
    // 1. m_dsp->analyze(input, freq_buffer)
    // 2. m_encoder->forward(...)
    // 3. m_decoders->forward(...)
    // 4. m_dsp->synthesize(freq_buffer, output)
    
    std::copy(input, input + size, output); // Bypass for now
}

void NativeSyclEngine::reset() {
    if (m_dsp) m_dsp->initialize(); // Clear OLA buffers
}

} // namespace sa::infrastructure::sycl_impl
