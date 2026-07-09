#include "silence_arc/infrastructure/onednn_inference_engine.h"
#include "silence_arc/infrastructure/log.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cstring>

namespace silence_arc::infrastructure {

using json = nlohmann::json;
using namespace dnnl;

OneDNNInferenceEngine::OneDNNInferenceEngine(sycl::queue& queue, dnnl::engine& engine, dnnl::stream& stream)
    : m_queue(queue), m_engine(engine), m_stream(stream) {}

bool OneDNNInferenceEngine::load_weights(const std::string& weights_path) {
    SA_LOG_INFO("[INFO] Loading weights from: " << weights_path);
    std::string metadata_path = weights_path + "/metadata.json";
    std::ifstream f(metadata_path);
    if (!f.is_open()) return false;

    try {
        json metadata = json::parse(f);
        for (auto& [name, info] : metadata.items()) {
            std::string filename = info["file"];
            std::vector<int> shape = info["shape"];
            size_t num_elements = 1;
            for (int s : shape) num_elements *= s;
            
            std::ifstream bin_f(weights_path + "/" + filename, std::ios::binary);
            if (!bin_f.is_open()) continue;
            
            std::vector<float> buffer(num_elements);
            bin_f.read(reinterpret_cast<char*>(buffer.data()), num_elements * sizeof(float));
            m_weights[name] = std::move(buffer);
        }
        SA_LOG_INFO("[SUCCESS] Loaded " << m_weights.size() << " weight tensors.");
        setup_encoder();
        setup_erb_decoder();
        setup_df_decoder();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Weights load failed: " << e.what() << std::endl;
        return false;
    }
}

template<typename Map>
auto safe_at(const Map& m, const std::string& key) -> decltype(m.at(key)) {
    auto it = m.find(key);
    if (it == m.end()) {
        throw std::runtime_error("Key not found in map: " + key);
    }
    return it->second;
}

void OneDNNInferenceEngine::add_conv2d(std::vector<OneDNNLayer>& sequence,
                                     const std::string& weight_name,
                                     memory input, memory& output,
                                     int out_channels, int kh, int kw, 
                                     int pl, int pr, int pt, int pb,
                                     int sh, int sw, int groups) {
    if (m_weights.find(weight_name) == m_weights.end()) {
        throw std::runtime_error("Weight not found: " + weight_name);
    }

    auto src_md = input.get_desc();
    auto src_dims = src_md.get_dims();
    int in_channels = src_dims[1];
    
    memory::dims weights_dims;
    memory::format_tag weights_tag;
    if (groups > 1) {
        weights_dims = {groups, out_channels/groups, in_channels/groups, kh, kw};
        weights_tag = memory::format_tag::goihw;
    } else {
        weights_dims = {out_channels, in_channels, kh, kw};
        weights_tag = memory::format_tag::oihw;
    }

    memory::dims dst_dims = {
        src_dims[0],
        out_channels,
        (src_dims[2] + pt + pb - kh) / sh + 1,
        (src_dims[3] + pl + pr - kw) / sw + 1
    };

    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, weights_tag);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    output = memory(dst_md, m_engine);
    m_persistent_mems[weight_name + "_out"] = output;

    auto weights_mem = memory(weights_md, m_engine);
    m_queue.memcpy(weights_mem.get_data_handle(), safe_at(m_weights, weight_name).data(), safe_at(m_weights, weight_name).size() * sizeof(float)).wait();
    m_persistent_mems[weight_name] = weights_mem;

    try {
        auto conv_pd = convolution_forward::primitive_desc(m_engine,
            prop_kind::forward_inference, algorithm::convolution_direct,
            src_md, weights_md, dst_md, {sh, sw}, {pt, pl}, {pb, pr});

        OneDNNLayer layer;
        layer.prim = convolution_forward(conv_pd);
        layer.args = {{DNNL_ARG_SRC, input}, {DNNL_ARG_WEIGHTS, weights_mem}, {DNNL_ARG_DST, output}};
        layer.name = "Conv2d(" + weight_name + ")";
        sequence.push_back(layer);
    } catch (const dnnl::error& e) {
        std::cerr << "[ERROR] Conv2d creation failed for " << weight_name << ": " << e.what() << std::endl;
        throw;
    }
}

void OneDNNInferenceEngine::add_batchnorm(std::vector<OneDNNLayer>& sequence,
                                        const std::string& bn_name,
                                        memory input, memory output) {
    auto data_md = input.get_desc();
    int channels = data_md.get_dims()[1];

    auto bn_pd = batch_normalization_forward::primitive_desc(m_engine,
        prop_kind::forward_inference, data_md, data_md, 1e-5f,
        normalization_flags::use_global_stats | normalization_flags::use_scale | normalization_flags::use_shift);

    auto mean_mem = memory(bn_pd.mean_desc(), m_engine);
    auto var_mem = memory(bn_pd.variance_desc(), m_engine);
    
    auto scale_md = memory::desc({channels}, memory::data_type::f32, memory::format_tag::x);
    auto shift_md = memory::desc({channels}, memory::data_type::f32, memory::format_tag::x);
    auto scale_mem = memory(scale_md, m_engine);
    auto shift_mem = memory(shift_md, m_engine);

    m_queue.memcpy(mean_mem.get_data_handle(), safe_at(m_weights, bn_name + ".running_mean").data(), channels * sizeof(float)).wait();
    m_queue.memcpy(var_mem.get_data_handle(), safe_at(m_weights, bn_name + ".running_var").data(), channels * sizeof(float)).wait();
    m_queue.memcpy(scale_mem.get_data_handle(), safe_at(m_weights, bn_name + ".weight").data(), channels * sizeof(float)).wait();
    m_queue.memcpy(shift_mem.get_data_handle(), safe_at(m_weights, bn_name + ".bias").data(), channels * sizeof(float)).wait();

    OneDNNLayer layer;
    layer.prim = batch_normalization_forward(bn_pd);
    layer.args = {
        {DNNL_ARG_SRC, input}, {DNNL_ARG_DST, output},
        {DNNL_ARG_MEAN, mean_mem}, {DNNL_ARG_VARIANCE, var_mem},
        {DNNL_ARG_SCALE, scale_mem}, {DNNL_ARG_SHIFT, shift_mem}
    };
    layer.name = "BatchNorm(" + bn_name + ")";
    sequence.push_back(layer);

    m_persistent_mems[bn_name + ".mean"] = mean_mem;
    m_persistent_mems[bn_name + ".var"] = var_mem;
    m_persistent_mems[bn_name + ".scale"] = scale_mem;
    m_persistent_mems[bn_name + ".shift"] = shift_mem;
}

void OneDNNInferenceEngine::add_relu(std::vector<OneDNNLayer>& sequence, memory input, memory output) {
    auto md = input.get_desc();
    auto relu_pd = eltwise_forward::primitive_desc(m_engine, prop_kind::forward_inference, algorithm::eltwise_relu, md, md, 0.0f);
    
    OneDNNLayer layer;
    layer.prim = eltwise_forward(relu_pd);
    layer.args = {{DNNL_ARG_SRC, input}, {DNNL_ARG_DST, output}};
    layer.name = "ReLU";
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::add_sigmoid(std::vector<OneDNNLayer>& sequence, memory input, memory output) {
    auto md = input.get_desc();
    auto sigmoid_pd = eltwise_forward::primitive_desc(m_engine, prop_kind::forward_inference, algorithm::eltwise_logistic, md, md, 0.0f, 0.0f);
    
    OneDNNLayer layer;
    layer.prim = eltwise_forward(sigmoid_pd);
    layer.args = {{DNNL_ARG_SRC, input}, {DNNL_ARG_DST, output}};
    layer.name = "Sigmoid";
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::add_sycl_reorder_nchw_to_tnc(std::vector<OneDNNLayer>& sequence,
                                      const std::string& name,
                                      dnnl::memory input_nchw, dnnl::memory& output_tnc) {
    auto in_dims = input_nchw.get_desc().get_dims(); // [N, C, T, 1]
    int n_dim = in_dims[0];
    int c_dim = in_dims[1];
    int t_dim = in_dims[2];

    auto dst_md = memory::desc({t_dim, n_dim, c_dim}, memory::data_type::f32, memory::format_tag::tnc);
    output_tnc = memory(dst_md, m_engine);
    m_persistent_mems[name + "_sycl_reorder_out"] = output_tnc;

    float* in_ptr = static_cast<float*>(input_nchw.get_data_handle());
    float* out_ptr = static_cast<float*>(output_tnc.get_data_handle());

    OneDNNLayer layer;
    layer.name = "SYCL_Reorder_NchwToTnc(" + name + ")";
    sycl::queue q = m_queue;
    layer.custom_exec = [q, in_ptr, out_ptr, t_dim, n_dim, c_dim]() mutable {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class NchwToTncReorder>(sycl::range<3>(t_dim, n_dim, c_dim), [=](sycl::id<3> id) {
                int t = id[0];
                int n = id[1];
                int c = id[2];
                // NCHW: [n][c][t][1] -> n*(C*T) + c*T + t
                int in_idx = n * (c_dim * t_dim) + c * t_dim + t;
                // TNC: [t][n][c] -> t*(N*C) + n*C + c
                int out_idx = t * (n_dim * c_dim) + n * c_dim + c;
                out_ptr[out_idx] = in_ptr[in_idx];
            });
        });
        // No wait: this kernel is on the shared in-order queue, so the next
        // layer (and the terminal output copy) chain after it automatically.
    };
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::add_gru(std::vector<OneDNNLayer>& sequence,
                                   const std::string& gru_name,
                                   memory input, memory& output,
                                   int hidden_size, int num_layers) {
    auto src_md = input.get_desc();
    auto src_dims = src_md.get_dims(); 
    
    int batch, time, input_size;
    memory gru_src = input;

    if (src_dims.size() == 4) {
        batch = src_dims[0];
        time = src_dims[2];
        input_size = src_dims[1];
        add_sycl_reorder_nchw_to_tnc(sequence, gru_name + "_permute", input, gru_src);
    } else {
        batch = src_dims[0];
        time = src_dims[1];
        input_size = src_dims[2];
    }

    memory::dims weights_layer_dims = {num_layers, 1, input_size, 3, hidden_size};
    memory::dims weights_iter_dims = {num_layers, 1, hidden_size, 3, hidden_size};
    // lbr_gru requires 4 bias gates [b_u, b_r, b_cx, b_ch] -- see dnnl_lbr_gru docs:
    // the candidate gate's input-side and hidden-side biases must stay separate
    // because only the hidden-side term is scaled by the reset gate:
    // c_t = tanh(W_c*x_t + b_cx + r_t*(U_c*h_{t-1} + b_ch))
    memory::dims bias_dims = {num_layers, 1, 4, hidden_size};
    memory::dims dst_dims = {time, batch, hidden_size};
    memory::dims state_dims = {num_layers, 1, batch, hidden_size};

    auto weights_layer_md = memory::desc(weights_layer_dims, memory::data_type::f32, memory::format_tag::ldigo);
    auto weights_iter_md = memory::desc(weights_iter_dims, memory::data_type::f32, memory::format_tag::ldigo);
    auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::ldgo);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::tnc);
    auto state_md = memory::desc(state_dims, memory::data_type::f32, memory::format_tag::ldnc);

    output = memory(dst_md, m_engine);
    m_persistent_mems[gru_name + "_out"] = output;
    
    // Create/Retrieve persistent state
    if (m_gru_states.find(gru_name) == m_gru_states.end()) {
        m_gru_states[gru_name] = memory(state_md, m_engine);
        m_queue.fill(m_gru_states[gru_name].get_data_handle(), 0.0f, num_layers * batch * hidden_size).wait();
    }
    memory state_mem = m_gru_states[gru_name];

    auto weights_layer_mem = memory(weights_layer_md, m_engine);
    auto weights_iter_mem = memory(weights_iter_md, m_engine);
    auto bias_mem = memory(bias_md, m_engine);

    for (int l = 0; l < num_layers; ++l) {
        std::string layer_gru_name = gru_name + ".gru";
        std::string l_suffix = "_l" + std::to_string(l);
        
        auto& w_ih = safe_at(m_weights, layer_gru_name + ".weight_ih" + l_suffix);
        auto& w_hh = safe_at(m_weights, layer_gru_name + ".weight_hh" + l_suffix);
        auto& b_ih = safe_at(m_weights, layer_gru_name + ".bias_ih" + l_suffix);
        auto& b_hh = safe_at(m_weights, layer_gru_name + ".bias_hh" + l_suffix);

        std::vector<float> w_ih_reordered(input_size * 3 * hidden_size);
        for(int i=0; i<input_size; ++i) {
            for(int j=0; j<hidden_size; ++j) w_ih_reordered[i*3*hidden_size + 0*hidden_size + j] = w_ih[(1*hidden_size + j)*input_size + i];
            for(int j=0; j<hidden_size; ++j) w_ih_reordered[i*3*hidden_size + 1*hidden_size + j] = w_ih[(0*hidden_size + j)*input_size + i];
            for(int j=0; j<hidden_size; ++j) w_ih_reordered[i*3*hidden_size + 2*hidden_size + j] = w_ih[(2*hidden_size + j)*input_size + i];
        }
        
        std::vector<float> w_hh_reordered(hidden_size * 3 * hidden_size);
        for(int i=0; i<hidden_size; ++i) {
            for(int j=0; j<hidden_size; ++j) w_hh_reordered[i*3*hidden_size + 0*hidden_size + j] = w_hh[(1*hidden_size + j)*hidden_size + i];
            for(int j=0; j<hidden_size; ++j) w_hh_reordered[i*3*hidden_size + 1*hidden_size + j] = w_hh[(0*hidden_size + j)*hidden_size + i];
            for(int j=0; j<hidden_size; ++j) w_hh_reordered[i*3*hidden_size + 2*hidden_size + j] = w_hh[(2*hidden_size + j)*hidden_size + i];
        }

        // PyTorch weight_ih/weight_hh/bias_ih/bias_hh gate order is [r, z, n].
        // oneDNN lbr_gru gate order is [u(=z), r, candidate]; bias needs 4 slots:
        // [b_u, b_r, b_cx, b_ch] with b_cx/b_ch kept separate (see note above) --
        // summing them (as a plain vanilla_gru bias would) silently breaks the
        // reset-gate semantics and was the root cause of the GRU state slowly
        // diverging instead of converging.
        std::vector<float> b_reordered(4 * hidden_size);
        for(int j=0; j<hidden_size; ++j) b_reordered[0*hidden_size + j] = b_ih[1*hidden_size + j] + b_hh[1*hidden_size + j]; // b_u
        for(int j=0; j<hidden_size; ++j) b_reordered[1*hidden_size + j] = b_ih[0*hidden_size + j] + b_hh[0*hidden_size + j]; // b_r
        for(int j=0; j<hidden_size; ++j) b_reordered[2*hidden_size + j] = b_ih[2*hidden_size + j];                          // b_cx (input-side only)
        for(int j=0; j<hidden_size; ++j) b_reordered[3*hidden_size + j] = b_hh[2*hidden_size + j];                          // b_ch (hidden-side only)

        // Each layer occupies its own slice of the num_layers-dimensioned buffer;
        // writing all layers to offset 0 (as before) made layer l overwrite l-1.
        auto* layer_base = static_cast<char*>(weights_layer_mem.get_data_handle());
        auto* iter_base = static_cast<char*>(weights_iter_mem.get_data_handle());
        auto* bias_base = static_cast<char*>(bias_mem.get_data_handle());
        m_queue.memcpy(layer_base + l * w_ih_reordered.size() * sizeof(float), w_ih_reordered.data(), w_ih_reordered.size() * sizeof(float)).wait();
        m_queue.memcpy(iter_base + l * w_hh_reordered.size() * sizeof(float), w_hh_reordered.data(), w_hh_reordered.size() * sizeof(float)).wait();
        m_queue.memcpy(bias_base + l * b_reordered.size() * sizeof(float), b_reordered.data(), b_reordered.size() * sizeof(float)).wait();
    }

    auto gru_pd = lbr_gru_forward::primitive_desc(m_engine,
        prop_kind::forward_inference, rnn_direction::unidirectional_left2right,
        gru_src.get_desc(), state_md, weights_layer_md, weights_iter_md, bias_md, dst_md, state_md);

    OneDNNLayer layer;
    layer.prim = lbr_gru_forward(gru_pd);
    layer.args = {
        {DNNL_ARG_SRC, gru_src},
        {DNNL_ARG_SRC_ITER, state_mem},
        {DNNL_ARG_WEIGHTS_LAYER, weights_layer_mem},
        {DNNL_ARG_WEIGHTS_ITER, weights_iter_mem},
        {DNNL_ARG_BIAS, bias_mem},
        {DNNL_ARG_DST, output},
        {DNNL_ARG_DST_ITER, state_mem}
    };
    layer.name = "GRU(" + gru_name + ")";
    sequence.push_back(layer);

    m_persistent_mems[gru_name + ".weights_layer"] = weights_layer_mem;
    m_persistent_mems[gru_name + ".weights_iter"] = weights_iter_mem;
    m_persistent_mems[gru_name + ".bias"] = bias_mem;
}

void OneDNNInferenceEngine::add_linear(std::vector<OneDNNLayer>& sequence,
                                     const std::string& weight_name,
                                     const std::string& bias_name,
                                     memory input, memory& output,
                                     int out_features) {
    auto src_md = input.get_desc();
    auto src_dims = src_md.get_dims();
    int in_features = src_dims[src_dims.size()-1];

    if (src_dims.size() == 4) {
        add_conv2d(sequence, weight_name, input, output, out_features, 1, 1, 0, 0, 0, 0, 1, 1);
        return;
    }

    memory::dims weights_dims = {out_features, in_features};
    memory::dims bias_dims = {out_features};
    memory::dims dst_dims = src_dims;
    dst_dims[dst_dims.size()-1] = out_features;

    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oi);
    auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::any);

    memory::format_tag dst_tag;
    switch (dst_dims.size()) {
        case 1: dst_tag = memory::format_tag::a; break;
        case 2: dst_tag = memory::format_tag::ab; break;
        case 3: dst_tag = memory::format_tag::abc; break;
        case 4: dst_tag = memory::format_tag::abcd; break;
        default: dst_tag = memory::format_tag::any; break;
    }

    auto actual_dst_md = memory::desc(dst_dims, memory::data_type::f32, dst_tag);
    output = memory(actual_dst_md, m_engine);
    m_persistent_mems[weight_name + "_out"] = output;

    auto weights_mem = memory(weights_md, m_engine);
    m_queue.memcpy(weights_mem.get_data_handle(), safe_at(m_weights, weight_name).data(), safe_at(m_weights, weight_name).size() * sizeof(float)).wait();
    m_persistent_mems[weight_name] = weights_mem;

    auto bias_mem = memory(bias_md, m_engine);
    if (!bias_name.empty()) {
        m_queue.memcpy(bias_mem.get_data_handle(), safe_at(m_weights, bias_name).data(), safe_at(m_weights, bias_name).size() * sizeof(float)).wait();
        m_persistent_mems[bias_name] = bias_mem;
    }

    auto ip_pd = inner_product_forward::primitive_desc(m_engine,
        prop_kind::forward_inference, src_md, weights_md, bias_md, dst_md);

    OneDNNLayer layer;
    layer.prim = inner_product_forward(ip_pd);
    layer.args = {{DNNL_ARG_SRC, input}, {DNNL_ARG_WEIGHTS, weights_mem}, {DNNL_ARG_BIAS, bias_mem}, {DNNL_ARG_DST, output}};
    layer.name = "Linear(" + weight_name + ")";
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::add_conv_transpose2d(std::vector<OneDNNLayer>& sequence,
                                               const std::string& weight_name,
                                               memory input, memory& output,
                                               int out_channels, int kh, int kw,
                                               int pl, int pr, int pt, int pb,
                                               int sh, int sw, int groups) {
    auto src_md = input.get_desc();
    auto src_dims = src_md.get_dims();
    int in_channels = src_dims[1];

    memory::dims weights_dims;
    memory::format_tag weights_tag;
    if (groups > 1) {
        weights_dims = {groups, in_channels/groups, out_channels/groups, kh, kw};
        weights_tag = memory::format_tag::goihw;
    } else {
        weights_dims = {in_channels, out_channels, kh, kw};
        weights_tag = memory::format_tag::iohw;
    }

    memory::dims dst_dims = {
        src_dims[0],
        out_channels,
        (src_dims[2] - 1) * sh + kh - pt - pb + (sh == 2 ? 1 : 0),
        (src_dims[3] - 1) * sw + kw - pl - pr + (sw == 2 ? 1 : 0)
    };

    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, weights_tag);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    output = memory(dst_md, m_engine);
    m_persistent_mems[weight_name + "_out"] = output;

    auto weights_mem = memory(weights_md, m_engine);
    m_queue.memcpy(weights_mem.get_data_handle(), safe_at(m_weights, weight_name).data(), safe_at(m_weights, weight_name).size() * sizeof(float)).wait();
    m_persistent_mems[weight_name] = weights_mem;

    auto deconv_pd = deconvolution_forward::primitive_desc(m_engine,
        prop_kind::forward_inference, algorithm::deconvolution_direct,
        src_md, weights_md, dst_md, {sh, sw}, {pt, pl}, {pb, pr});

    OneDNNLayer layer;
    layer.prim = deconvolution_forward(deconv_pd);
    layer.args = {{DNNL_ARG_SRC, input}, {DNNL_ARG_WEIGHTS, weights_mem}, {DNNL_ARG_DST, output}};
    layer.name = "ConvTranspose2d(" + weight_name + ")";
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::add_grouped_linear(std::vector<OneDNNLayer>& sequence,
                                             const std::string& weight_name,
                                             memory input, memory& output,
                                             int groups) {
    auto src_md = input.get_desc();
    auto src_dims = src_md.get_dims();
    auto& w_data = safe_at(m_weights, weight_name);
    
    int in_features;
    if (src_dims.size() == 3) {
        in_features = src_dims[2];
    } else {
        in_features = src_dims[1] * (src_dims.size() > 3 ? src_dims[3] : 1);
    }

    int in_per_group = in_features / groups;
    int out_per_group = w_data.size() / (groups * in_per_group);
    int out_channels = groups * out_per_group;

    SA_LOG_DEBUG("GroupedLinear(" << weight_name << "): in_feat=" << in_features << ", in_pg=" << in_per_group << ", out_pg=" << out_per_group << ", out_channels=" << out_channels << ", dims=" << src_dims.size());

    try {
        memory conv_src = input;
        auto current_src_dims = input.get_desc().get_dims();
        int current_time = (current_src_dims.size() == 3 ? current_src_dims[0] : current_src_dims[2]);
        auto src_md_for_conv = memory::desc({src_dims[0], in_features, current_time, 1}, memory::data_type::f32, memory::format_tag::nchw);

        if (current_src_dims.size() == 3) {
            add_sycl_reorder_tnc_to_nchw(sequence, weight_name + "_permute", input, conv_src);
        } else if (current_src_dims.size() == 4 && current_src_dims != src_md_for_conv.get_dims()) {
            conv_src = memory(src_md_for_conv, m_engine, input.get_data_handle());
            m_persistent_mems[weight_name + "_flattened_src"] = conv_src;
        }

        memory::dims weights_dims = {groups, out_per_group, in_per_group, 1, 1};
        auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::goihw);
        memory::dims dst_dims = {src_dims[0], out_channels, current_time, 1};
        auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

        output = memory(dst_md, m_engine);
        m_persistent_mems[weight_name + "_out"] = output;

        auto weights_mem = memory(weights_md, m_engine);
        std::vector<float> transposed(w_data.size());
        for(int g=0; g<groups; ++g) {
            for(int i=0; i<in_per_group; ++i) {
                for(int o=0; o<out_per_group; ++o) {
                    transposed[g*in_per_group*out_per_group + o*in_per_group + i] = w_data[g*in_per_group*out_per_group + i*out_per_group + o];
                }
            }
        }
        m_queue.memcpy(weights_mem.get_data_handle(), transposed.data(), transposed.size() * sizeof(float)).wait();
        m_persistent_mems[weight_name] = weights_mem;

        auto conv_pd = convolution_forward::primitive_desc(m_engine,
            prop_kind::forward_inference, algorithm::convolution_direct,
            src_md_for_conv, weights_md, dst_md, {1, 1}, {0, 0}, {0, 0});

        OneDNNLayer layer;
        layer.prim = convolution_forward(conv_pd);
        layer.args = {{DNNL_ARG_SRC, conv_src}, {DNNL_ARG_WEIGHTS, weights_mem}, {DNNL_ARG_DST, output}};
        layer.name = "GroupedLinear(" + weight_name + ")";
        sequence.push_back(layer);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] GroupedLinear creation failed: " << e.what() << std::endl;
        throw;
    }
}

void OneDNNInferenceEngine::add_binary_add(std::vector<OneDNNLayer>& sequence,
                                         memory input_a, memory input_b,
                                         memory output) {
    auto a_md = input_a.get_desc();
    auto b_md = input_b.get_desc();
    auto a_dims = a_md.get_dims();
    auto b_dims = b_md.get_dims();

    if (verbose_logging_enabled()) {
        std::cout << "[DEBUG] BinaryAdd: A=";
        for(auto d : a_dims) std::cout << d << " ";
        std::cout << "| B=";
        for(auto d : b_dims) std::cout << d << " ";
        std::cout << std::endl;
    }
    auto dims = a_md.get_dims();

    auto get_std_tag = [](size_t n) {
        if (n == 1) return memory::format_tag::a;
        if (n == 2) return memory::format_tag::ab;
        if (n == 3) return memory::format_tag::abc;
        return memory::format_tag::abcd;
    };

    auto std_md = memory::desc(dims, memory::data_type::f32, get_std_tag(dims.size()));
    
    memory a_std = input_a;
    if (a_md != std_md) {
        a_std = memory(std_md, m_engine);
        m_persistent_mems["add_a_reorder_" + std::to_string(m_persistent_mems.size())] = a_std;
        auto re_pd = reorder::primitive_desc(m_engine, a_md, m_engine, std_md);
        sequence.push_back({reorder(re_pd), {{DNNL_ARG_FROM, input_a}, {DNNL_ARG_TO, a_std}}, "ReorderA", nullptr});
    }
    
    memory b_std = input_b;
    if (b_md != std_md) {
        b_std = memory(std_md, m_engine);
        m_persistent_mems["add_b_reorder_" + std::to_string(m_persistent_mems.size())] = b_std;
        auto re_pd = reorder::primitive_desc(m_engine, b_md, m_engine, std_md);
        sequence.push_back({reorder(re_pd), {{DNNL_ARG_FROM, input_b}, {DNNL_ARG_TO, b_std}}, "ReorderB", nullptr});
    }

    auto bin_pd = binary::primitive_desc(m_engine, algorithm::binary_add, std_md, std_md, std_md);
    
    OneDNNLayer layer;
    layer.prim = binary(bin_pd);
    layer.args = {{DNNL_ARG_SRC_0, a_std}, {DNNL_ARG_SRC_1, b_std}, {DNNL_ARG_DST, output}};
    layer.name = "Add";
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::add_concat(std::vector<OneDNNLayer>& sequence,
                                     memory input_a, memory input_b,
                                     memory& output, int concat_dim) {
    auto a_dims = input_a.get_desc().get_dims();
    auto b_dims = input_b.get_desc().get_dims();
    auto dims_size = a_dims.size();

    auto get_std_tag = [](size_t n) {
        if (n == 1) return memory::format_tag::a;
        if (n == 2) return memory::format_tag::ab;
        if (n == 3) return memory::format_tag::abc;
        return memory::format_tag::abcd;
    };

    auto a_std_md = memory::desc(a_dims, memory::data_type::f32, get_std_tag(dims_size));
    auto b_std_md = memory::desc(b_dims, memory::data_type::f32, get_std_tag(dims_size));
    
    memory a_std = input_a;
    if (input_a.get_desc() != a_std_md) {
        a_std = memory(a_std_md, m_engine);
        m_persistent_mems["concat_a_reorder_" + std::to_string(m_persistent_mems.size())] = a_std;
        auto re_pd = reorder::primitive_desc(m_engine, input_a.get_desc(), m_engine, a_std_md);
        sequence.push_back({reorder(re_pd), {{DNNL_ARG_FROM, input_a}, {DNNL_ARG_TO, a_std}}, "ReorderA_Concat", nullptr});
    }
    
    memory b_std = input_b;
    if (input_b.get_desc() != b_std_md) {
        b_std = memory(b_std_md, m_engine);
        m_persistent_mems["concat_b_reorder_" + std::to_string(m_persistent_mems.size())] = b_std;
        auto re_pd = reorder::primitive_desc(m_engine, input_b.get_desc(), m_engine, b_std_md);
        sequence.push_back({reorder(re_pd), {{DNNL_ARG_FROM, input_b}, {DNNL_ARG_TO, b_std}}, "ReorderB_Concat", nullptr});
    }

    memory::dims dst_dims = a_dims;
    dst_dims[concat_dim] += b_dims[concat_dim];
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, get_std_tag(dims_size));

    output = memory(dst_md, m_engine);
    m_persistent_mems["concat_" + std::to_string(m_persistent_mems.size()) + "_out"] = output;

    auto concat_pd = concat::primitive_desc(m_engine, concat_dim, {a_std_md, b_std_md});
    
    OneDNNLayer layer;
    layer.prim = concat(concat_pd);
    layer.args = {{DNNL_ARG_MULTIPLE_SRC, a_std}, {DNNL_ARG_MULTIPLE_SRC + 1, b_std}, {DNNL_ARG_DST, output}};
    layer.name = "Concat";
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::add_sycl_reorder_tnc_to_nchw(std::vector<OneDNNLayer>& sequence,
                                      const std::string& name,
                                      dnnl::memory input_tnc, dnnl::memory& output_nchw) {
    auto in_dims = input_tnc.get_desc().get_dims();
    int t_dim = in_dims[0];
    int n_dim = in_dims[1];
    int c_dim = in_dims[2];

    auto dst_md = memory::desc({n_dim, c_dim, t_dim, 1}, memory::data_type::f32, memory::format_tag::nchw);
    output_nchw = memory(dst_md, m_engine);
    m_persistent_mems[name + "_sycl_reorder_out"] = output_nchw;

    float* in_ptr = static_cast<float*>(input_tnc.get_data_handle());
    float* out_ptr = static_cast<float*>(output_nchw.get_data_handle());

    OneDNNLayer layer;
    layer.name = "SYCL_Reorder(" + name + ")";
    
    // Explicitly copy queue and dims to local variables for safe capture
    sycl::queue q = m_queue;
    layer.custom_exec = [q, in_ptr, out_ptr, t_dim, n_dim, c_dim]() mutable {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class TncToNchwReorder>(sycl::range<3>(t_dim, n_dim, c_dim), [=](sycl::id<3> id) {
                int t = id[0];
                int n = id[1];
                int c = id[2];
                int in_idx = t * (n_dim * c_dim) + n * c_dim + c;
                int out_idx = n * (c_dim * t_dim) + c * t_dim + t;
                out_ptr[out_idx] = in_ptr[in_idx];
            });
        });
        // No wait: this kernel is on the shared in-order queue, so the next
        // layer (and the terminal output copy) chain after it automatically.
    };
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::add_squeezed_gru(std::vector<OneDNNLayer>& sequence,
                                           const std::string& name,
                                           memory input, memory& output,
                                           int hidden_size, int out_size,
                                           int groups, bool skip, int num_layers) {
    memory current = input;
    memory lin_in_out;
    add_grouped_linear(sequence, name + ".linear_in.0.weight", current, lin_in_out, groups);
    add_relu(sequence, lin_in_out, lin_in_out);
    memory gru_out;
    add_gru(sequence, name, lin_in_out, gru_out, hidden_size, num_layers);
    if (out_size > 0) {
        add_grouped_linear(sequence, name + ".linear_out.0.weight", gru_out, output, groups);
        add_relu(sequence, output, output);
    } else {
        output = gru_out;
    }
    if (skip) {
        auto standard_md = memory::desc(output.get_desc().get_dims(), output.get_desc().get_data_type(), memory::format_tag::nchw);
        memory clean_output = memory(standard_md, m_engine);
        m_persistent_mems[name + "_skip_out_clean"] = clean_output;
        
        auto re_pd = reorder::primitive_desc(m_engine, output.get_desc(), m_engine, standard_md);
        sequence.push_back({reorder(re_pd), {{DNNL_ARG_FROM, output}, {DNNL_ARG_TO, clean_output}}, "CleanOutput", nullptr});
        
        memory clean_skip_input;
        if (input.get_desc().get_dims().size() == 3) {
            add_sycl_reorder_tnc_to_nchw(sequence, name + "_skip_in_clean", input, clean_skip_input);
        } else {
            clean_skip_input = input;
        }
        add_binary_add(sequence, clean_output, clean_skip_input, clean_output);
        output = clean_output;
    }
}

void OneDNNInferenceEngine::add_flatten_to_nchw(std::vector<OneDNNLayer>& sequence,
                                              const std::string& name,
                                              memory input, memory& output,
                                              int out_channels) {
    auto in_dims = input.get_desc().get_dims();
    memory::dims flat_dims = {in_dims[0], out_channels, 1, 1};
    auto dst_md = memory::desc(flat_dims, memory::data_type::f32, memory::format_tag::nchw);
    
    // Reshape in oneDNN is done by creating a new memory object sharing the same handle
    output = memory(dst_md, m_engine, input.get_data_handle());
    m_persistent_mems[name + "_flat_out"] = output;

    // No primitive needed for a simple reshape/view change
    OneDNNLayer layer;
    layer.name = "Reshape(" + name + ")";
    layer.custom_exec = []() { /* Nothing to do, it's just a view */ };
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::add_sycl_permute_nchw_to_flat_fmajor(std::vector<OneDNNLayer>& sequence,
                                          const std::string& name,
                                          dnnl::memory input_nchw, dnnl::memory& output_flat) {
    auto in_dims = input_nchw.get_desc().get_dims(); // [N, C, 1, F]
    int n_dim = in_dims[0];
    int c_dim = in_dims[1];
    int f_dim = in_dims[3];

    auto dst_md = memory::desc({n_dim, c_dim * f_dim, 1, 1}, memory::data_type::f32, memory::format_tag::nchw);
    output_flat = memory(dst_md, m_engine);
    m_persistent_mems[name + "_permute_out"] = output_flat;

    float* in_ptr = static_cast<float*>(input_nchw.get_data_handle());
    float* out_ptr = static_cast<float*>(output_flat.get_data_handle());

    OneDNNLayer layer;
    layer.name = "SYCL_Permute_NchwToFlatFmajor(" + name + ")";
    sycl::queue q = m_queue;
    layer.custom_exec = [q, in_ptr, out_ptr, n_dim, c_dim, f_dim]() mutable {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class NchwToFlatFmajor>(sycl::range<3>(n_dim, c_dim, f_dim), [=](sycl::id<3> id) {
                int n = id[0];
                int c = id[1];
                int f = id[2];
                int in_idx = n * (c_dim * f_dim) + c * f_dim + f;   // NCHW: c*F+f
                int out_idx = n * (c_dim * f_dim) + f * c_dim + c;  // flat: f*C+c
                out_ptr[out_idx] = in_ptr[in_idx];
            });
        });
        // No wait: this kernel is on the shared in-order queue, so the next
        // layer (and the terminal output copy) chain after it automatically.
    };
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::add_sycl_permute_flat_fmajor_to_nchw(std::vector<OneDNNLayer>& sequence,
                                          const std::string& name,
                                          dnnl::memory input_flat, dnnl::memory& output_nchw,
                                          int out_channels) {
    auto in_dims = input_flat.get_desc().get_dims();
    int n_dim = in_dims[0];
    int flat_size = in_dims[1];
    int c_dim = out_channels;
    int f_dim = flat_size / c_dim;

    auto dst_md = memory::desc({n_dim, c_dim, 1, f_dim}, memory::data_type::f32, memory::format_tag::nchw);
    output_nchw = memory(dst_md, m_engine);
    m_persistent_mems[name + "_permute_out"] = output_nchw;

    float* in_ptr = static_cast<float*>(input_flat.get_data_handle());
    float* out_ptr = static_cast<float*>(output_nchw.get_data_handle());

    OneDNNLayer layer;
    layer.name = "SYCL_Permute_FlatFmajorToNchw(" + name + ")";
    sycl::queue q = m_queue;
    layer.custom_exec = [q, in_ptr, out_ptr, n_dim, c_dim, f_dim]() mutable {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class FlatFmajorToNchw>(sycl::range<3>(n_dim, c_dim, f_dim), [=](sycl::id<3> id) {
                int n = id[0];
                int c = id[1];
                int f = id[2];
                int in_idx = n * (c_dim * f_dim) + f * c_dim + c;   // flat: f*C+c
                int out_idx = n * (c_dim * f_dim) + c * f_dim + f;  // NCHW: c*F+f
                out_ptr[out_idx] = in_ptr[in_idx];
            });
        });
        // No wait: this kernel is on the shared in-order queue, so the next
        // layer (and the terminal output copy) chain after it automatically.
    };
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::shift_time_history(dnnl::memory& history_mem, const float* new_frame_host,
                                              int channels, int history_len, int width) {
    size_t total = static_cast<size_t>(channels) * history_len * width;
    std::vector<float> host_buf(total);
    m_queue.memcpy(host_buf.data(), history_mem.get_data_handle(), total * sizeof(float)).wait();
    for (int c = 0; c < channels; ++c) {
        float* channel_ptr = host_buf.data() + static_cast<size_t>(c) * history_len * width;
        std::memmove(channel_ptr, channel_ptr + width, static_cast<size_t>(history_len - 1) * width * sizeof(float));
        std::memcpy(channel_ptr + static_cast<size_t>(history_len - 1) * width,
                    new_frame_host + static_cast<size_t>(c) * width, width * sizeof(float));
    }
    m_queue.memcpy(history_mem.get_data_handle(), host_buf.data(), total * sizeof(float)).wait();
}

void OneDNNInferenceEngine::setup_encoder() {
    SA_LOG_INFO("[INFO] Building Full Encoder...");
    // H holds a rolling 3-frame causal window (t-2,t-1,t), not just the current
    // frame: erb_conv0 has kernel_size=(3,3) with the time axis fully causal
    // (pad top=2,bottom=0 in the reference), which requires 2 real past frames,
    // not zero-padding, feeding tap 2 (the most recent) with real data every call.
    auto erb_input_md = memory::desc({1, 1, 3, 32}, memory::data_type::f32, memory::format_tag::nchw);
    auto erb_input_mem = memory(erb_input_md, m_engine);
    m_queue.fill(erb_input_mem.get_data_handle(), 0.0f, 1 * 1 * 3 * 32).wait();
    m_persistent_mems["encoder_erb_input"] = erb_input_mem;
    memory e_curr = erb_input_mem;
    add_conv2d(m_encoder_layers, "enc.erb_conv0.1.weight", e_curr, e_curr, 64, 3, 3, 1, 1, 0, 0, 1, 1);
    add_batchnorm(m_encoder_layers, "enc.erb_conv0.2", e_curr, e_curr);
    add_relu(m_encoder_layers, e_curr, e_curr);
    m_persistent_mems["enc.erb_block0_out"] = e_curr;
    add_conv2d(m_encoder_layers, "enc.erb_conv1.0.weight", e_curr, e_curr, 64, 1, 3, 1, 1, 0, 0, 1, 2, 64);
    add_conv2d(m_encoder_layers, "enc.erb_conv1.1.weight", e_curr, e_curr, 64, 1, 1, 0, 0, 0, 0, 1, 1, 1);
    add_batchnorm(m_encoder_layers, "enc.erb_conv1.2", e_curr, e_curr);
    add_relu(m_encoder_layers, e_curr, e_curr);
    m_persistent_mems["enc.erb_block1_out"] = e_curr;
    add_conv2d(m_encoder_layers, "enc.erb_conv2.0.weight", e_curr, e_curr, 64, 1, 3, 1, 1, 0, 0, 1, 2, 64);
    add_conv2d(m_encoder_layers, "enc.erb_conv2.1.weight", e_curr, e_curr, 64, 1, 1, 0, 0, 0, 0, 1, 1, 1);
    add_batchnorm(m_encoder_layers, "enc.erb_conv2.2", e_curr, e_curr);
    add_relu(m_encoder_layers, e_curr, e_curr);
    m_persistent_mems["enc.erb_block2_out"] = e_curr;
    add_conv2d(m_encoder_layers, "enc.erb_conv3.0.weight", e_curr, e_curr, 64, 1, 3, 1, 1, 0, 0, 1, 1, 64);
    add_conv2d(m_encoder_layers, "enc.erb_conv3.1.weight", e_curr, e_curr, 64, 1, 1, 0, 0, 0, 0, 1, 1, 1);
    add_batchnorm(m_encoder_layers, "enc.erb_conv3.2", e_curr, e_curr);
    add_relu(m_encoder_layers, e_curr, e_curr);
    m_persistent_mems["enc.erb_block3_out"] = e_curr;
    // Same rolling 3-frame causal window as encoder_erb_input, see comment above.
    auto df_input_md = memory::desc({1, 2, 3, 96}, memory::data_type::f32, memory::format_tag::nchw);
    auto df_input_mem = memory(df_input_md, m_engine);
    m_queue.fill(df_input_mem.get_data_handle(), 0.0f, 1 * 2 * 3 * 96).wait();
    m_persistent_mems["encoder_df_input"] = df_input_mem;
    memory c_curr = df_input_mem;
    add_conv2d(m_encoder_layers, "enc.df_conv0.1.weight", c_curr, c_curr, 64, 3, 3, 1, 1, 0, 0, 1, 1, 2);
    add_conv2d(m_encoder_layers, "enc.df_conv0.2.weight", c_curr, c_curr, 64, 1, 1, 0, 0, 0, 0, 1, 1, 1);
    add_batchnorm(m_encoder_layers, "enc.df_conv0.3", c_curr, c_curr);
    add_relu(m_encoder_layers, c_curr, c_curr);
    m_persistent_mems["enc.df_block0_out"] = c_curr;
    add_conv2d(m_encoder_layers, "enc.df_conv1.0.weight", c_curr, c_curr, 64, 1, 3, 1, 1, 0, 0, 1, 2, 64);
    add_conv2d(m_encoder_layers, "enc.df_conv1.1.weight", c_curr, c_curr, 64, 1, 1, 0, 0, 0, 0, 1, 1, 1);
    add_batchnorm(m_encoder_layers, "enc.df_conv1.2", c_curr, c_curr);
    add_relu(m_encoder_layers, c_curr, c_curr);
    m_persistent_mems["enc.df_block1_out"] = c_curr;
    // PyTorch feeds df_fc_emb with c1.permute(0,2,3,1).flatten(2) (frequency-major,
    // channel-minor) -- must actually transpose the NCHW buffer, not just relabel it.
    memory c_curr_fmajor;
    add_sycl_permute_nchw_to_flat_fmajor(m_encoder_layers, "enc_df_fc_emb_permute", c_curr, c_curr_fmajor);
    memory cemb;
    add_grouped_linear(m_encoder_layers, "enc.df_fc_emb.0.weight", c_curr_fmajor, cemb, 32);
    add_relu(m_encoder_layers, cemb, cemb);

    // Align and Merge ERB and Complex embeddings -- same frequency-major flatten
    // is required here (e3.permute(0,2,3,1).flatten(2) in the reference).
    memory erb_flat;
    add_sycl_permute_nchw_to_flat_fmajor(m_encoder_layers, "enc_erb_merge", e_curr, erb_flat);
    add_binary_add(m_encoder_layers, erb_flat, cemb, erb_flat);
    e_curr = erb_flat;
    
    memory emb_out;
    add_squeezed_gru(m_encoder_layers, "enc.emb_gru", e_curr, emb_out, 256, 512, 16);
    m_persistent_mems["encoder_emb_out"] = emb_out;
    memory lsnr;
    add_linear(m_encoder_layers, "enc.lsnr_fc.0.weight", "enc.lsnr_fc.0.bias", emb_out, lsnr, 1);
    add_sigmoid(m_encoder_layers, lsnr, lsnr);
    SA_LOG_INFO("[SUCCESS] Full Encoder ready.");
}

void OneDNNInferenceEngine::setup_erb_decoder() {
    SA_LOG_INFO("[INFO] Building ERB Decoder...");
    memory emb = safe_at(m_persistent_mems, "encoder_emb_out");
    memory dec_emb_out;
    // emb_num_layers=3 in config.ini -> ErbDecoder's GRU needs num_layers=2 (emb_num_layers-1);
    // checkpoint has real weight_ih_l1/weight_hh_l1 tensors that were being silently skipped.
    // emb_gru_skip=none in config.ini (no gru_skip.* weights exist) -> no residual add here.
    add_squeezed_gru(m_erb_decoder_layers, "erb_dec.emb_gru", emb, dec_emb_out, 256, 512, 16, false, 2);
    memory current = dec_emb_out;
    memory e3 = safe_at(m_persistent_mems, "enc.erb_block3_out");
    memory p3;
    add_conv2d(m_erb_decoder_layers, "erb_dec.conv3p.0.weight", e3, p3, 64, 1, 1, 0, 0, 0, 0, 1, 1, 64);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.conv3p.1", p3, p3);
    add_relu(m_erb_decoder_layers, p3, p3);
    
    // Reshape embedding to [1, 64, 1, 8] -- PyTorch does
    // emb.view(b,t,f8,-1).permute(0,3,1,2) (frequency-major flat -> NCHW), the
    // inverse of the permute used when the embedding was first flattened.
    memory reshaped_current;
    add_sycl_permute_flat_fmajor_to_nchw(m_erb_decoder_layers, "erb_dec_emb_reshape", current, reshaped_current, 64);
    m_persistent_mems["erb_dec_emb_reshaped"] = reshaped_current;
    current = reshaped_current;

    add_binary_add(m_erb_decoder_layers, p3, current, current);
    add_conv2d(m_erb_decoder_layers, "erb_dec.convt3.0.weight", current, current, 64, 1, 3, 1, 1, 0, 0, 1, 1, 64);
    add_conv2d(m_erb_decoder_layers, "erb_dec.convt3.1.weight", current, current, 64, 1, 1, 0, 0, 0, 0, 1, 1);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.convt3.2", current, current);
    add_relu(m_erb_decoder_layers, current, current);
    memory e2 = safe_at(m_persistent_mems, "enc.erb_block2_out");
    memory p2;
    add_conv2d(m_erb_decoder_layers, "erb_dec.conv2p.0.weight", e2, p2, 64, 1, 1, 0, 0, 0, 0, 1, 1, 64);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.conv2p.1", p2, p2);
    add_relu(m_erb_decoder_layers, p2, p2);
    add_binary_add(m_erb_decoder_layers, p2, current, current);
    add_conv_transpose2d(m_erb_decoder_layers, "erb_dec.convt2.0.weight", current, current, 64, 1, 3, 1, 1, 0, 0, 1, 2, 64);
    add_conv2d(m_erb_decoder_layers, "erb_dec.convt2.1.weight", current, current, 64, 1, 1, 0, 0, 0, 0, 1, 1);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.convt2.2", current, current);
    add_relu(m_erb_decoder_layers, current, current);
    memory e1 = safe_at(m_persistent_mems, "enc.erb_block1_out");
    memory p1;
    add_conv2d(m_erb_decoder_layers, "erb_dec.conv1p.0.weight", e1, p1, 64, 1, 1, 0, 0, 0, 0, 1, 1, 64);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.conv1p.1", p1, p1);
    add_relu(m_erb_decoder_layers, p1, p1);
    add_binary_add(m_erb_decoder_layers, p1, current, current);
    add_conv_transpose2d(m_erb_decoder_layers, "erb_dec.convt1.0.weight", current, current, 64, 1, 3, 1, 1, 0, 0, 1, 2, 64);
    add_conv2d(m_erb_decoder_layers, "erb_dec.convt1.1.weight", current, current, 64, 1, 1, 0, 0, 0, 0, 1, 1);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.convt1.2", current, current);
    add_relu(m_erb_decoder_layers, current, current);
    memory e0 = safe_at(m_persistent_mems, "enc.erb_block0_out");
    memory p0;
    add_conv2d(m_erb_decoder_layers, "erb_dec.conv0p.0.weight", e0, p0, 64, 1, 1, 0, 0, 0, 0, 1, 1, 64);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.conv0p.1", p0, p0);
    add_relu(m_erb_decoder_layers, p0, p0);
    add_binary_add(m_erb_decoder_layers, p0, current, current);
    add_conv2d(m_erb_decoder_layers, "erb_dec.conv0_out.0.weight", current, current, 1, 1, 3, 1, 1, 0, 0, 1, 1);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.conv0_out.1", current, current);
    add_sigmoid(m_erb_decoder_layers, current, current);
    m_persistent_mems["erb_mask_out"] = current;
    SA_LOG_INFO("[SUCCESS] ERB Decoder ready.");
}

void OneDNNInferenceEngine::setup_df_decoder() {
    SA_LOG_INFO("[INFO] Building DF Decoder...");
    memory emb = safe_at(m_persistent_mems, "encoder_emb_out");
    memory dec_emb_out;
    
    // 1. GRU Path
    add_squeezed_gru(m_df_decoder_layers, "df_dec.df_gru", emb, dec_emb_out, 256, -1, 8, false, 2);
    
    // 2. DF Skip Connection (Embedding Path)
    memory skip_out;
    add_grouped_linear(m_df_decoder_layers, "df_dec.df_skip.weight", emb, skip_out, 16);
    // df_skip is a bare GroupedLinearEinsum in PyTorch (deepfilternet3.py) -- no activation.
    
    // Sum GRU + Skip
    // Need to handle TNC vs NCHW for the sum
    memory gru_out_nchw;
    add_sycl_reorder_tnc_to_nchw(m_df_decoder_layers, "df_dec_gru_res", dec_emb_out, gru_out_nchw);
    add_binary_add(m_df_decoder_layers, gru_out_nchw, skip_out, gru_out_nchw);
    
    // 3. Complex Pathway (c0 from encoder). df_convp.1 has kernel_size_t=5 with
    // fully causal padding (top=4,bottom=0) in the reference -- feed it a real
    // rolling 5-frame window (kept current in infer(), see shift_time_history
    // call there) instead of zero-padding a single frame.
    memory c0_history_init = safe_at(m_persistent_mems, "enc.df_block0_out"); // for shape only
    auto c0h_dims = c0_history_init.get_desc().get_dims(); // [1,64,1,96]
    auto c0_history_md = memory::desc({c0h_dims[0], c0h_dims[1], 5, c0h_dims[3]}, memory::data_type::f32, memory::format_tag::nchw);
    memory c0_history(c0_history_md, m_engine);
    m_queue.fill(c0_history.get_data_handle(), 0.0f, c0h_dims[0] * c0h_dims[1] * 5 * c0h_dims[3]).wait();
    m_persistent_mems["c0_history"] = c0_history;
    memory df_p;
    // df_convp.1 is grouped [groups=2, in=64, out=10] -> weight [10, 32, 5, 1]
    add_conv2d(m_df_decoder_layers, "df_dec.df_convp.1.weight", c0_history, df_p, 10, 5, 1, 0, 0, 0, 0, 1, 1, 2);
    // df_convp.2 is pointwise [in=10, out=10]
    add_conv2d(m_df_decoder_layers, "df_dec.df_convp.2.weight", df_p, df_p, 10, 1, 1, 0, 0, 0, 0, 1, 1);
    add_batchnorm(m_df_decoder_layers, "df_dec.df_convp.3", df_p, df_p);
    add_relu(m_df_decoder_layers, df_p, df_p);
    
    // 4. Grouped Linear to final Coefficients
    memory df_linear_out;
    add_grouped_linear(m_df_decoder_layers, "df_dec.df_out.0.weight", gru_out_nchw, df_linear_out, 16);
    
    // 5. Final Sum: df_linear_out + df_convp
    // The sum is done in SYCL to handle the [F, O*2] vs [F*O*2] mapping correctly
    // (df_p is [1, 10, 1, 96] -> [order*2][bin]; df_linear_out is flat [960]).
    OneDNNLayer final_sum;
    final_sum.name = "DF_Final_Sum";
    float* p_ptr = static_cast<float*>(df_p.get_data_handle());
    float* l_ptr = static_cast<float*>(df_linear_out.get_data_handle());
    
    sycl::queue q = m_queue;
    final_sum.custom_exec = [q, p_ptr, l_ptr]() mutable {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class DfFinalSum>(sycl::range<1>(960), [=](sycl::id<1> id) {
                int i = id[0];
                // df_p is [1, 10, 1, 96] -> [order*2][bin]
                int bin = i / 10;
                int ch = i % 10;
                int p_idx = ch * 96 + bin;
                l_ptr[i] = std::tanh(l_ptr[i]) + p_ptr[p_idx];
            });
        });
        // No wait: this kernel is on the shared in-order queue, so the next
        // layer (and the terminal output copy) chain after it automatically.
    };
    m_df_decoder_layers.push_back(final_sum);
    
    m_persistent_mems["df_coefs_out"] = df_linear_out;
    SA_LOG_INFO("[SUCCESS] DF Decoder ready.");
}

void OneDNNInferenceEngine::infer_erb(const float* erb_features, float* output_mask) {
    std::vector<float> dummy_df(96 * 2, 0.0f);
    std::vector<float> dummy_coefs(960);
    infer(erb_features, dummy_df.data(), output_mask, dummy_coefs.data());
}

void OneDNNInferenceEngine::infer(const float* erb_features, const float* df_features, float* output_mask, float* df_coefs) {
    if (m_encoder_layers.empty()) return;
    try {
        auto erb_mem = safe_at(m_persistent_mems, "encoder_erb_input");
        shift_time_history(erb_mem, erb_features, /*channels=*/1, /*history_len=*/3, /*width=*/32);
        auto df_input_mem = safe_at(m_persistent_mems, "encoder_df_input");
        std::vector<float> df_reordered(2 * 96);
        for (int i = 0; i < 96; ++i) {
            df_reordered[i] = df_features[i * 2 + 0];
            df_reordered[96 + i] = df_features[i * 2 + 1];
        }
        shift_time_history(df_input_mem, df_reordered.data(), /*channels=*/2, /*history_len=*/3, /*width=*/96);
        for (auto& layer : m_encoder_layers) {
            if (layer.custom_exec) layer.custom_exec();
            else layer.prim.execute(m_stream, layer.args);
        }
        {
            // df_convp.1 (kh=5) needs a real 5-frame causal window of c0, not
            // just the current frame -- roll it forward here, after the encoder
            // has produced this frame's fresh c0, before the df decoder reads it.
            auto c0_mem = safe_at(m_persistent_mems, "enc.df_block0_out");
            std::vector<float> c0_host(64 * 96);
            m_queue.memcpy(c0_host.data(), c0_mem.get_data_handle(), c0_host.size() * sizeof(float)).wait();
            auto c0_history_mem = safe_at(m_persistent_mems, "c0_history");
            shift_time_history(c0_history_mem, c0_host.data(), /*channels=*/64, /*history_len=*/5, /*width=*/96);
        }
        for (auto& layer : m_erb_decoder_layers) {
            if (layer.custom_exec) layer.custom_exec();
            else layer.prim.execute(m_stream, layer.args);
        }
        for (auto& layer : m_df_decoder_layers) {
            if (layer.custom_exec) layer.custom_exec();
            else layer.prim.execute(m_stream, layer.args);
        }
        // No m_stream.wait() here: the mask copy below is on the same in-order
        // queue and its wait() flushes the whole decoder chain (and surfaces any
        // async errors) in one sync instead of two.
        auto mask_mem = safe_at(m_persistent_mems, "erb_mask_out");
        m_queue.memcpy(output_mask, mask_mem.get_data_handle(), 32 * sizeof(float)).wait();
        auto df_mem = safe_at(m_persistent_mems, "df_coefs_out");
        m_queue.memcpy(df_coefs, df_mem.get_data_handle(), 960 * sizeof(float)).wait();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] oneDNN inference failed: " << e.what() << std::endl;
        throw;
    }
}

size_t OneDNNInferenceEngine::get_df_coefs_count() const {
    return 96 * 5 * 2;
}

void OneDNNInferenceEngine::reset() {
    for (auto& [name, mem] : m_gru_states) {
        auto desc = mem.get_desc();
        auto dims = desc.get_dims();
        size_t size = 1;
        for (auto d : dims) size *= d;
        m_queue.fill(mem.get_data_handle(), 0.0f, size).wait();
    }
    for (const char* history_name : {"encoder_erb_input", "encoder_df_input", "c0_history"}) {
        auto it = m_persistent_mems.find(history_name);
        if (it == m_persistent_mems.end()) continue;
        auto dims = it->second.get_desc().get_dims();
        size_t size = 1;
        for (auto d : dims) size *= d;
        m_queue.fill(it->second.get_data_handle(), 0.0f, size).wait();
    }
}

} // namespace silence_arc::infrastructure
