#pragma once

#include "silence_arc/domain/neural_network.h"
#include <dnnl.hpp>
#include <sycl/sycl.hpp>
#include <memory>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include <functional>

namespace sa::infrastructure {

struct OneDNNLayer {
    dnnl::primitive prim;
    std::unordered_map<int, dnnl::memory> args;
    std::string name;
    std::function<void()> custom_exec = nullptr;
};

class OneDNNInferenceEngine : public domain::NeuralNetworkModel {
public:
    OneDNNInferenceEngine(sycl::queue& queue, dnnl::engine& engine, dnnl::stream& stream);
    ~OneDNNInferenceEngine() override = default;

    bool load_weights(const std::string& weights_path) override;
    void infer_erb(const float* erb_features, float* output_mask) override;

    // Test helpers
    void test_conv2d_mapping();
    void test_batchnorm_mapping();
    void test_gru_mapping();
    void test_linear_mapping();

private:
    void setup_encoder();
    void setup_erb_decoder();
    void setup_df_decoder();
    
    // Builders
    void add_conv2d(std::vector<OneDNNLayer>& sequence,
                    const std::string& weight_name,
                    dnnl::memory input, dnnl::memory& output,
                    int out_channels, int kh, int kw, 
                    int pl, int pr, int pt, int pb,
                    int sh, int sw, int groups = 1);

    void add_batchnorm(std::vector<OneDNNLayer>& sequence,
                       const std::string& bn_name,
                       dnnl::memory input, dnnl::memory output);

    void add_relu(std::vector<OneDNNLayer>& sequence,
                  dnnl::memory input, dnnl::memory output);

    void add_sigmoid(std::vector<OneDNNLayer>& sequence,
                     dnnl::memory input, dnnl::memory output);

    void add_gru(std::vector<OneDNNLayer>& sequence,
                 const std::string& gru_name,
                 dnnl::memory input, dnnl::memory& output,
                 int hidden_size, int num_layers = 1);

    void add_linear(std::vector<OneDNNLayer>& sequence,
                    const std::string& weight_name,
                    const std::string& bias_name,
                    dnnl::memory input, dnnl::memory& output,
                    int out_features);

    void add_conv_transpose2d(std::vector<OneDNNLayer>& sequence,
                              const std::string& weight_name,
                              dnnl::memory input, dnnl::memory& output,
                              int out_channels, int kh, int kw,
                              int pl, int pr, int pt, int pb,
                              int sh, int sw, int groups = 1);

    void add_grouped_linear(std::vector<OneDNNLayer>& sequence,
                            const std::string& weight_name,
                            dnnl::memory input, dnnl::memory& output,
                            int groups);

    void add_binary_add(std::vector<OneDNNLayer>& sequence,
                        dnnl::memory input_a, dnnl::memory input_b,
                        dnnl::memory output);

    void add_concat(std::vector<OneDNNLayer>& sequence,
                    dnnl::memory input_a, dnnl::memory input_b,
                    dnnl::memory& output, int concat_dim = 1);

    void add_squeezed_gru(std::vector<OneDNNLayer>& sequence,
                          const std::string& name,
                          dnnl::memory input, dnnl::memory& output,
                          int hidden_size, int out_size,
                          int groups, bool skip = false, int num_layers = 1);

    void add_sycl_reorder_tnc_to_nchw(std::vector<OneDNNLayer>& sequence,
                                      const std::string& name,
                                      dnnl::memory input_tnc, dnnl::memory& output_nchw);

    sycl::queue& m_queue;
    dnnl::engine& m_engine;
    dnnl::stream& m_stream;

    std::map<std::string, std::vector<float>> m_weights;
    std::map<std::string, dnnl::memory> m_persistent_mems;
    std::vector<OneDNNLayer> m_encoder_layers;
    std::vector<OneDNNLayer> m_erb_decoder_layers;
    std::vector<OneDNNLayer> m_df_decoder_layers;
};

} // namespace sa::infrastructure
