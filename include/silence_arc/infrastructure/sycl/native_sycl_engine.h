#pragma once

#include "silence_arc/domain/audio_processor.h"
#include "silence_arc/infrastructure/sycl/sycl_memory_manager.h"
#include "silence_arc/infrastructure/sycl/sycl_dsp_engine.h"
#include "silence_arc/infrastructure/sycl/sycl_graph_builder.h"
#include <memory>
#include <vector>

namespace sa::infrastructure::sycl_impl {

/**
 * @brief Full native SYCL/oneDNN inference engine for DeepFilterNet3.
 * Replaces the Rust adapter with highly optimized Intel-specific code.
 */
class NativeSyclEngine : public domain::IAudioProcessor {
public:
    NativeSyclEngine();
    ~NativeSyclEngine() override;

    // IAudioProcessor Implementation
    bool initialize() override;
    std::string get_device_name() const override;
    void process_frame(const float* input, float* output, size_t size) override;
    size_t get_frame_size() const override { return 480; }
    size_t get_latency() const override { return 960; } // 2 frames lookahead
    void set_deep_filtering_enabled(bool enabled) override { m_df_enabled = enabled; }
    void set_attenuation_limit(float limit_db) override { m_attenuation_limit = limit_db; }
    void reset() override;

private:
    void build_graph();
    
    // Core SYCL/oneDNN Resources
    std::unique_ptr<sycl::queue> m_queue;
    std::unique_ptr<dnnl::engine> m_dnnl_engine;
    std::unique_ptr<SyclMemoryManager> m_mem_manager;
    std::unique_ptr<SyclDspEngine> m_dsp;
    std::unique_ptr<SyclGraphBuilder> m_graph_builder;

    // State
    bool m_initialized = false;
    bool m_df_enabled = true;
    float m_attenuation_limit = 40.0f;

    // Neural Layers (Modules)
    std::vector<std::unique_ptr<Layer>> m_encoder_layers;
    std::vector<std::unique_ptr<Layer>> m_erb_decoder_layers;
    std::vector<std::unique_ptr<Layer>> m_df_decoder_layers;

    // Intermediate Tensors (Managed by USM)
    // ...
};

} // namespace sa::infrastructure::sycl_impl
