#include "silence_arc/infrastructure/onednn_inference_engine.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace sa::infrastructure {

using json = nlohmann::json;
using namespace dnnl;

OneDNNInferenceEngine::OneDNNInferenceEngine(sycl::queue& queue, dnnl::engine& engine, dnnl::stream& stream)
    : m_queue(queue), m_engine(engine), m_stream(stream) {}

bool OneDNNInferenceEngine::load_weights(const std::string& weights_path) {
    std::cout << "[INFO] Loading weights from: " << weights_path << std::endl;
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
        std::cout << "[SUCCESS] Loaded " << m_weights.size() << " weight tensors." << std::endl;
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

    auto conv_pd = convolution_forward::primitive_desc(m_engine,
        prop_kind::forward_inference, algorithm::convolution_direct,
        src_md, weights_md, dst_md, {sh, sw}, {pt, pl}, {pb, pr});

    OneDNNLayer layer;
    layer.prim = convolution_forward(conv_pd);
    layer.args = {{DNNL_ARG_SRC, input}, {DNNL_ARG_WEIGHTS, weights_mem}, {DNNL_ARG_DST, output}};
    layer.name = "Conv2d(" + weight_name + ")";
    sequence.push_back(layer);
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

void OneDNNInferenceEngine::add_gru(std::vector<OneDNNLayer>& sequence,
                                   const std::string& gru_name,
                                   memory input, memory& output,
                                   int hidden_size, int num_layers) {
    auto src_md = input.get_desc();
    auto src_dims = src_md.get_dims(); 
    
    int batch, time, input_size;
    memory gru_src = input;

    if (src_dims.size() == 4) {
        // NCHW -> [Batch, Channels, Time, Freq=1]
        // We want TNC: [Time, Batch, Channels]
        batch = src_dims[0];
        time = src_dims[2];
        input_size = src_dims[1];
        
        // oneDNN reorder with permutation
        // src_md is NCHW. We want to permute to TNC.
        // TNC: dim 0=time (idx 2), dim 1=batch (idx 0), dim 2=channels (idx 1)
        
        // Create source descriptor that "views" input as TNC
        // Input is [N, C, T, 1] in NCHW order.
        // Strides for [N, C, T, 1]: [C*T*1, T*1, 1, 1]
        // We want to map this to logical [T, N, C]
        // Logical T (dim 0) -> physical stride 1 (from index 2)
        // Logical N (dim 1) -> physical stride C*T (from index 0)
        // Logical C (dim 2) -> physical stride T (from index 1)
        
        memory::dims tnc_dims = {time, batch, input_size};
        memory::dims tnc_strides = {1, input_size * time, time};
        auto src_view_md = memory::desc(tnc_dims, memory::data_type::f32, tnc_strides);
        
        auto dst_tnc_md = memory::desc(tnc_dims, memory::data_type::f32, memory::format_tag::tnc);
        gru_src = memory(dst_tnc_md, m_engine);
        m_persistent_mems[gru_name + "_permuted_input"] = gru_src;

        auto reorder_pd = reorder::primitive_desc(m_engine, src_view_md, m_engine, dst_tnc_md);
        OneDNNLayer p_layer;
        p_layer.prim = reorder(reorder_pd);
        // We need to wrap 'input' memory with the new descriptor 'src_view_md'
        auto input_view = memory(src_view_md, m_engine, input.get_data_handle());
        p_layer.args = {{DNNL_ARG_FROM, input_view}, {DNNL_ARG_TO, gru_src}};
        p_layer.name = "Permute(NCHW->TNC)";
        sequence.push_back(p_layer);
    } else {
        batch = src_dims[0];
        time = src_dims[1];
        input_size = src_dims[2];
    }

    // oneDNN RNN descriptors
    memory::dims weights_layer_dims = {num_layers, 1, input_size, 3, hidden_size};
    memory::dims weights_iter_dims = {num_layers, 1, hidden_size, 3, hidden_size};
    memory::dims bias_dims = {num_layers, 1, 3, hidden_size};
    memory::dims dst_dims = {time, batch, hidden_size};

    auto weights_layer_md = memory::desc(weights_layer_dims, memory::data_type::f32, memory::format_tag::ldigo);
    auto weights_iter_md = memory::desc(weights_iter_dims, memory::data_type::f32, memory::format_tag::ldigo);
    auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::ldgo);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::tnc);

    output = memory(dst_md, m_engine);
    m_persistent_mems[gru_name + "_out"] = output;
    
    auto actual_src_md = gru_src.get_desc();

    auto weights_layer_mem = memory(weights_layer_md, m_engine);
    auto weights_iter_mem = memory(weights_iter_md, m_engine);
    auto bias_mem = memory(bias_md, m_engine);

    // Reorder PyTorch (r, z, n) to oneDNN (z, r, h)
    auto reorder_gates = [&](std::vector<float>& dst, const float* src, int rows, int hs) {
        // src is [3*hs, rows]
        for (int i = 0; i < rows; ++i) {
            // z gate (oneDNN 0, PyTorch 1)
            std::copy(src + hs + i * hs, src + 2 * hs + i * hs, dst.data() + i * 3 * hs + 0 * hs);
            // r gate (oneDNN 1, PyTorch 0)
            std::copy(src + 0 * hs + i * hs, src + 1 * hs + i * hs, dst.data() + i * 3 * hs + 1 * hs);
            // h gate (oneDNN 2, PyTorch 2)
            std::copy(src + 2 * hs + i * hs, src + 3 * hs + i * hs, dst.data() + i * 3 * hs + 2 * hs);
        }
    };

    // Wait, PyTorch weights are [3*hs, input_size]
    // oneDNN ldigo for weights_layer is [layers, dirs, input_size, gates, hs]
    // So we need to transpose PyTorch weights if they are [3*hs, input_size] to [input_size, 3*hs] then reorder.
    
    for (int l = 0; l < num_layers; ++l) {
        std::string layer_gru_name = gru_name + ".gru";
        std::string l_suffix = "_l" + std::to_string(l);
        
        auto& w_ih = safe_at(m_weights, layer_gru_name + ".weight_ih" + l_suffix);
        auto& w_hh = safe_at(m_weights, layer_gru_name + ".weight_hh" + l_suffix);
        auto& b_ih = safe_at(m_weights, layer_gru_name + ".bias_ih" + l_suffix);
        auto& b_hh = safe_at(m_weights, layer_gru_name + ".bias_hh" + l_suffix);

        std::vector<float> w_ih_reordered(input_size * 3 * hidden_size);
        // PyTorch w_ih is [3*hs, input_size]
        // We want [input_size, 3*hs] with gates reordered
        for(int i=0; i<input_size; ++i) {
            // z gate (PT 1)
            for(int j=0; j<hidden_size; ++j) w_ih_reordered[i*3*hidden_size + 0*hidden_size + j] = w_ih[(1*hidden_size + j)*input_size + i];
            // r gate (PT 0)
            for(int j=0; j<hidden_size; ++j) w_ih_reordered[i*3*hidden_size + 1*hidden_size + j] = w_ih[(0*hidden_size + j)*input_size + i];
            // h gate (PT 2)
            for(int j=0; j<hidden_size; ++j) w_ih_reordered[i*3*hidden_size + 2*hidden_size + j] = w_ih[(2*hidden_size + j)*input_size + i];
        }
        
        std::vector<float> w_hh_reordered(hidden_size * 3 * hidden_size);
        for(int i=0; i<hidden_size; ++i) {
            for(int j=0; j<hidden_size; ++j) w_hh_reordered[i*3*hidden_size + 0*hidden_size + j] = w_hh[(1*hidden_size + j)*hidden_size + i];
            for(int j=0; j<hidden_size; ++j) w_hh_reordered[i*3*hidden_size + 1*hidden_size + j] = w_hh[(0*hidden_size + j)*hidden_size + i];
            for(int j=0; j<hidden_size; ++j) w_hh_reordered[i*3*hidden_size + 2*hidden_size + j] = w_hh[(2*hidden_size + j)*hidden_size + i];
        }

        std::vector<float> b_reordered(3 * hidden_size);
        for(int j=0; j<hidden_size; ++j) b_reordered[0*hidden_size + j] = b_ih[1*hidden_size + j] + b_hh[1*hidden_size + j];
        for(int j=0; j<hidden_size; ++j) b_reordered[1*hidden_size + j] = b_ih[0*hidden_size + j] + b_hh[0*hidden_size + j];
        for(int j=0; j<hidden_size; ++j) b_reordered[2*hidden_size + j] = b_ih[2*hidden_size + j] + b_hh[2*hidden_size + j];

        // Copy to GPU (assuming 1 layer for now, need to handle multiple layers in ldigo)
        m_queue.memcpy(weights_layer_mem.get_data_handle(), w_ih_reordered.data(), w_ih_reordered.size() * sizeof(float)).wait();
        m_queue.memcpy(weights_iter_mem.get_data_handle(), w_hh_reordered.data(), w_hh_reordered.size() * sizeof(float)).wait();
        m_queue.memcpy(bias_mem.get_data_handle(), b_reordered.data(), b_reordered.size() * sizeof(float)).wait();
    }

    auto src_iter_md = memory::desc();
    auto dst_iter_md = memory::desc();

    auto gru_pd = gru_forward::primitive_desc(m_engine,
        prop_kind::forward_inference, rnn_direction::unidirectional_left2right,
        actual_src_md, src_iter_md, weights_layer_md, weights_iter_md, bias_md, dst_md, dst_iter_md);

    OneDNNLayer layer;
    layer.prim = gru_forward(gru_pd);
    layer.args = {
        {DNNL_ARG_SRC, gru_src},
        {DNNL_ARG_WEIGHTS_LAYER, weights_layer_mem},
        {DNNL_ARG_WEIGHTS_ITER, weights_iter_mem},
        {DNNL_ARG_BIAS, bias_mem},
        {DNNL_ARG_DST, output}
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
        // Use 1x1 Conv instead of Inner Product for 4D inputs
        add_conv2d(sequence, weight_name, input, output, out_features, 1, 1, 0, 0, 0, 0, 1, 1);
        // TODO: Handle bias for Conv2d if needed (add_conv2d current impl doesn't take bias_name)
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
    auto src_dims = src_md.get_dims(); // [B, T, C] or [B, C, T, 1] or [T, B, C]
    
    std::cout << "[DEBUG] GroupedLinear(" << weight_name << "): src_dims=[";
    for(size_t i=0; i<src_dims.size(); ++i) std::cout << src_dims[i] << (i==src_dims.size()-1 ? "" : ",");
    std::cout << "], groups=" << groups << std::endl;

    // We treat it as a 1x1 grouped convolution on [B, C, T, 1]
    // Weights in metadata are [G, I/G, O/G]
    auto& w_data = safe_at(m_weights, weight_name);
    
    int in_features;
    if (src_dims.size() == 3) {
        // TNC: [Time, Batch, Channels]
        in_features = src_dims[2];
    } else {
        // NCHW: [Batch, Channels, Time, Freq]
        in_features = src_dims[1] * (src_dims.size() > 3 ? src_dims[3] : 1);
    }

    int in_per_group = in_features / groups;
    int out_per_group = w_data.size() / (groups * in_per_group);
    int out_channels = groups * out_per_group;

    std::cout << "[DEBUG] GroupedLinear(" << weight_name << "): in_feat=" << in_features << ", in_pg=" << in_per_group << ", out_pg=" << out_per_group << ", out_channels=" << out_channels << std::endl;

    try {
        memory conv_src = input;
        auto current_src_md = input.get_desc();
        auto current_src_dims = current_src_md.get_dims();

        if (current_src_dims.size() == 3) {
            // TNC -> NCHW [Batch, Channels, Time, 1]
            int time = current_src_dims[0];
            int batch = current_src_dims[1];
            int channels = current_src_dims[2];
            
            // Physical TNC: [Time, Batch, Channels]
            // Time is dim 0, Batch is dim 1, Channels is dim 2.
            // Logical [N, C, T, 1] maps to physical [T, N, C]
            // N (logical 0) -> physical Batch (dim 1) -> stride Channels (current_src_dims[2])
            // C (logical 1) -> physical Channels (dim 2) -> stride 1
            // T (logical 2) -> physical Time (dim 0) -> stride Batch * Channels (current_src_dims[1] * current_src_dims[2])
            // W (logical 3) -> broadcast -> stride batch * channels * time
            
            memory::dims nchw_dims = {batch, channels, time, 1};
            memory::dims nchw_strides = {channels, 1, batch * channels, batch * channels * time}; 
            auto src_view_md = memory::desc(nchw_dims, memory::data_type::f32, nchw_strides);
            
            auto dst_nchw_md = memory::desc(nchw_dims, memory::data_type::f32, memory::format_tag::nchw);
            conv_src = memory(dst_nchw_md, m_engine);
            m_persistent_mems[weight_name + "_reshaped_input"] = conv_src;

            auto reorder_pd = reorder::primitive_desc(m_engine, src_view_md, m_engine, dst_nchw_md);
            OneDNNLayer p_layer;
            p_layer.prim = reorder(reorder_pd);
            auto input_view = memory(src_view_md, m_engine, input.get_data_handle());
            p_layer.args = {{DNNL_ARG_FROM, input_view}, {DNNL_ARG_TO, conv_src}};
            p_layer.name = "Permute(TNC->NCHW)";
            sequence.push_back(p_layer);
        }

        // Now conv_src is NCHW [B, C, T, 1] where C is in_features
        int current_time = (current_src_dims.size() == 3 ? current_src_dims[0] : src_dims[2]);
        auto src_md_for_conv = memory::desc({src_dims[0], in_features, current_time, 1}, memory::data_type::f32, memory::format_tag::nchw);
        
        memory::dims weights_dims = {groups, out_per_group, in_per_group, 1, 1};
        auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::goihw);
        
        memory::dims dst_dims = {src_dims[0], out_channels, current_time, 1};
        auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

        output = memory(dst_md, m_engine);
        m_persistent_mems[weight_name + "_out"] = output;

        auto weights_mem = memory(weights_md, m_engine);
        // PyTorch GroupedLinearEinsum weight is [G, I/G, O/G]
        // oneDNN goihw is [G, O/G, I/G, 1, 1]
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
    auto dst_md = output.get_desc();

    std::cout << "[DEBUG] BinaryAdd: src0_dims=[";
    for(size_t i=0; i<a_md.get_dims().size(); ++i) std::cout << a_md.get_dims()[i] << (i==a_md.get_dims().size()-1 ? "" : ",");
    std::cout << "], src1_dims=[";
    for(size_t i=0; i<b_md.get_dims().size(); ++i) std::cout << b_md.get_dims()[i] << (i==b_md.get_dims().size()-1 ? "" : ",");
    std::cout << "]" << std::endl;

    auto get_tag = [](int dims) {
        if (dims == 1) return memory::format_tag::a;
        if (dims == 2) return memory::format_tag::ab;
        if (dims == 3) return memory::format_tag::abc;
        return memory::format_tag::abcd;
    };

    auto bin_pd = binary::primitive_desc(m_engine, algorithm::binary_add, 
        memory::desc(a_md.get_dims(), a_md.get_data_type(), get_tag(a_md.get_dims().size())),
        memory::desc(b_md.get_dims(), b_md.get_data_type(), get_tag(b_md.get_dims().size())),
        memory::desc(dst_md.get_dims(), dst_md.get_data_type(), get_tag(dst_md.get_dims().size())));
    
    OneDNNLayer layer;
    layer.prim = binary(bin_pd);
    layer.args = {{DNNL_ARG_SRC_0, input_a}, {DNNL_ARG_SRC_1, input_b}, {DNNL_ARG_DST, output}};
    layer.name = "Add";
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::add_concat(std::vector<OneDNNLayer>& sequence,
                                     memory input_a, memory input_b,
                                     memory& output, int concat_dim) {
    auto a_md = input_a.get_desc();
    auto b_md = input_b.get_desc();
    
    auto a_dims = a_md.get_dims();
    auto b_dims = b_md.get_dims();
    
    memory::dims dst_dims = a_dims;
    dst_dims[concat_dim] += b_dims[concat_dim];

    auto get_tag = [](int dims) {
        if (dims == 1) return memory::format_tag::a;
        if (dims == 2) return memory::format_tag::ab;
        if (dims == 3) return memory::format_tag::abc;
        return memory::format_tag::abcd;
    };

    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, get_tag(dst_dims.size()));
    output = memory(dst_md, m_engine);
    m_persistent_mems["concat_out_" + std::to_string(sequence.size())] = output;

    auto concat_pd = concat::primitive_desc(m_engine, concat_dim, {a_md, b_md});
    
    OneDNNLayer layer;
    layer.prim = concat(concat_pd);
    layer.args = {
        {DNNL_ARG_MULTIPLE_SRC + 0, input_a},
        {DNNL_ARG_MULTIPLE_SRC + 1, input_b},
        {DNNL_ARG_DST, output}
    };
    layer.name = "Concat";
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::add_sycl_reorder_tnc_to_nchw(std::vector<OneDNNLayer>& sequence,
                                      const std::string& name,
                                      dnnl::memory input_tnc, dnnl::memory& output_nchw) {
    auto in_dims = input_tnc.get_desc().get_dims();
    if (in_dims.size() != 3) {
        throw std::runtime_error("add_sycl_reorder_tnc_to_nchw expects 3D input (TNC)");
    }
    
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
    layer.custom_exec = [this, in_ptr, out_ptr, t_dim, n_dim, c_dim]() {
        // Physical TNC index: t * (N * C) + n * C + c
        // Physical NCHW index: n * (C * T * 1) + c * (T * 1) + t * 1 + 0
        m_queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class TncToNchwReorder>(sycl::range<3>(t_dim, n_dim, c_dim), [=](sycl::id<3> id) {
                int t = id[0];
                int n = id[1];
                int c = id[2];

                int in_idx = t * (n_dim * c_dim) + n * c_dim + c;
                int out_idx = n * (c_dim * t_dim) + c * t_dim + t;

                out_ptr[out_idx] = in_ptr[in_idx];
            });
        });
        m_queue.wait_and_throw();
    };
    sequence.push_back(layer);
}

void OneDNNInferenceEngine::add_squeezed_gru(std::vector<OneDNNLayer>& sequence,
                                           const std::string& name,
                                           memory input, memory& output,
                                           int hidden_size, int out_size,
                                           int groups, bool skip, int num_layers) {
    memory current = input;
    
    // 1. linear_in
    memory lin_in_out;
    add_grouped_linear(sequence, name + ".linear_in.0.weight", current, lin_in_out, groups);
    add_relu(sequence, lin_in_out, lin_in_out);
    
    // 2. GRU
    memory gru_out;
    add_gru(sequence, name, lin_in_out, gru_out, hidden_size, num_layers);
    
    // 3. linear_out (Optional)
    memory lin_out_out;
    if (out_size > 0) {
        add_grouped_linear(sequence, name + ".linear_out.0.weight", gru_out, lin_out_out, groups);
        add_relu(sequence, lin_out_out, lin_out_out);
    } else {
        lin_out_out = gru_out;
    }
    
    output = lin_out_out;

    // 4. skip
    if (skip) {
        // Force a clean reorder of both inputs to standard NCHW to ensure Binary Add succeeds
        auto out_md = output.get_desc();
        auto out_dims = out_md.get_dims();
        auto standard_md = memory::desc(out_dims, out_md.get_data_type(), memory::format_tag::nchw);
        
        // Reorder current output to standard NCHW
        memory clean_output = memory(standard_md, m_engine);
        m_persistent_mems[name + "_skip_out_clean"] = clean_output;
        sequence.push_back({reorder(output, clean_output), {{DNNL_ARG_FROM, output}, {DNNL_ARG_TO, clean_output}}, "CleanOutput", nullptr});

        memory clean_skip_input;
        auto in_dims = input.get_desc().get_dims();
        if (in_dims.size() == 3) {
            // TNC -> NCHW via custom SYCL kernel
            add_sycl_reorder_tnc_to_nchw(sequence, name + "_skip_in_clean", input, clean_skip_input);
        } else {
            clean_skip_input = input;
        }
        
        add_binary_add(sequence, clean_output, clean_skip_input, clean_output);
        output = clean_output;
    }
}

void OneDNNInferenceEngine::setup_encoder() {
    std::cout << "[INFO] Building Full Encoder..." << std::endl;
    
    // 1. ERB Path
    // ERB Input: [1, 1, 1, 32] (Batch, Ch, Time, Freq)
    auto erb_input_md = memory::desc({1, 1, 1, 32}, memory::data_type::f32, memory::format_tag::nchw);
    auto erb_input_mem = memory(erb_input_md, m_engine);
    m_persistent_mems["encoder_erb_input"] = erb_input_mem;

    memory e_curr = erb_input_mem;
    // ERB Block 0: 3x3 Conv (1->64), symmetric pad 1
    add_conv2d(m_encoder_layers, "enc.erb_conv0.1.weight", e_curr, e_curr, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    add_batchnorm(m_encoder_layers, "enc.erb_conv0.2", e_curr, e_curr);
    add_relu(m_encoder_layers, e_curr, e_curr);
    m_persistent_mems["enc.erb_block0_out"] = e_curr;

    // ERB Block 1: 1x1 Depthwise (64, groups=64), 1x1 Pointwise (64->64), fstride=2
    add_conv2d(m_encoder_layers, "enc.erb_conv1.0.weight", e_curr, e_curr, 64, 1, 3, 1, 1, 0, 0, 1, 2, 64);
    add_conv2d(m_encoder_layers, "enc.erb_conv1.1.weight", e_curr, e_curr, 64, 1, 1, 0, 0, 0, 0, 1, 1, 1);
    add_batchnorm(m_encoder_layers, "enc.erb_conv1.2", e_curr, e_curr);
    add_relu(m_encoder_layers, e_curr, e_curr);
    m_persistent_mems["enc.erb_block1_out"] = e_curr;

    // ERB Block 2: same as 1, fstride=2
    add_conv2d(m_encoder_layers, "enc.erb_conv2.0.weight", e_curr, e_curr, 64, 1, 3, 1, 1, 0, 0, 1, 2, 64);
    add_conv2d(m_encoder_layers, "enc.erb_conv2.1.weight", e_curr, e_curr, 64, 1, 1, 0, 0, 0, 0, 1, 1, 1);
    add_batchnorm(m_encoder_layers, "enc.erb_conv2.2", e_curr, e_curr);
    add_relu(m_encoder_layers, e_curr, e_curr);
    m_persistent_mems["enc.erb_block2_out"] = e_curr;

    // ERB Block 3: same as 1, fstride=1
    add_conv2d(m_encoder_layers, "enc.erb_conv3.0.weight", e_curr, e_curr, 64, 1, 3, 1, 1, 0, 0, 1, 1, 64);
    add_conv2d(m_encoder_layers, "enc.erb_conv3.1.weight", e_curr, e_curr, 64, 1, 1, 0, 0, 0, 0, 1, 1, 1);
    add_batchnorm(m_encoder_layers, "enc.erb_conv3.2", e_curr, e_curr);
    add_relu(m_encoder_layers, e_curr, e_curr);
    m_persistent_mems["enc.erb_block3_out"] = e_curr;

    // 2. DF Path
    // DF Input: [1, 2, 1, 96] (Batch, ComplexCh, Time, Freq)
    auto df_input_md = memory::desc({1, 2, 1, 96}, memory::data_type::f32, memory::format_tag::nchw);
    auto df_input_mem = memory(df_input_md, m_engine);
    m_persistent_mems["encoder_df_input"] = df_input_mem;

    memory c_curr = df_input_mem;
    // DF Block 0: 3x3 Conv (2->64)
    add_conv2d(m_encoder_layers, "enc.df_conv0.1.weight", c_curr, c_curr, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    add_conv2d(m_encoder_layers, "enc.df_conv0.2.weight", c_curr, c_curr, 64, 1, 1, 0, 0, 0, 0, 1, 1, 1);
    add_batchnorm(m_encoder_layers, "enc.df_conv0.3", c_curr, c_curr);
    add_relu(m_encoder_layers, c_curr, c_curr);
    m_persistent_mems["enc.df_block0_out"] = c_curr;

    // DF Block 1: 1x3 Conv, fstride=2
    add_conv2d(m_encoder_layers, "enc.df_conv1.0.weight", c_curr, c_curr, 64, 1, 3, 1, 1, 0, 0, 1, 2, 64);
    add_conv2d(m_encoder_layers, "enc.df_conv1.1.weight", c_curr, c_curr, 64, 1, 1, 0, 0, 0, 0, 1, 1, 1);
    add_batchnorm(m_encoder_layers, "enc.df_conv1.2", c_curr, c_curr);
    add_relu(m_encoder_layers, c_curr, c_curr);
    m_persistent_mems["enc.df_block1_out"] = c_curr;

    // 3. Fusion & GRU
    // DF Path -> Linear -> cemb [B, T, 512]
    // GroupedLinear(3072 -> 512, groups=16)
    memory cemb;
    add_grouped_linear(m_encoder_layers, "enc.df_fc_emb.0.weight", c_curr, cemb, 16);
    add_relu(m_encoder_layers, cemb, cemb);

    // TODO: Broadcast cemb to match ERB path or vice versa.
    // For now, let's just skip binary_add if shapes mismatch to see if rest builds.
    // add_binary_add(m_encoder_layers, e_curr, cemb, e_curr);

    // Squeezed GRU
    memory emb_out;
    add_squeezed_gru(m_encoder_layers, "enc.emb_gru", e_curr, emb_out, 256, 512, 16);
    m_persistent_mems["encoder_emb_out"] = emb_out;

    // LSNR FC
    memory lsnr;
    add_linear(m_encoder_layers, "enc.lsnr_fc.0.weight", "enc.lsnr_fc.0.bias", emb_out, lsnr, 1);
    add_sigmoid(m_encoder_layers, lsnr, lsnr);

    std::cout << "[SUCCESS] Full Encoder ready with " << m_encoder_layers.size() << " primitives." << std::endl;
}

void OneDNNInferenceEngine::setup_erb_decoder() {
    std::cout << "[INFO] Building ERB Decoder..." << std::endl;
    
    // 1. Squeezed GRU
    memory emb = safe_at(m_persistent_mems, "encoder_emb_out");
    memory dec_emb_out;
    add_squeezed_gru(m_erb_decoder_layers, "erb_dec.emb_gru", emb, dec_emb_out, 256, 512, 16, true);

    // 2. Decoder Pathway (Transposed Convs + Pathway Additions)
    memory current = dec_emb_out;
    
    // Pathway 3
    memory e3 = safe_at(m_persistent_mems, "enc.erb_block3_out");
    memory p3;
    add_conv2d(m_erb_decoder_layers, "erb_dec.conv3p.0.weight", e3, p3, 64, 1, 1, 0, 0, 0, 0, 1, 1);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.conv3p.1", p3, p3);

    // Reshape and permute current (dec_emb_out) from [1, 512, 1, 1] to match p3 [1, 64, 1, 8]
    // PyTorch: view(b, t, f8=8, ch=64) -> permute(0, 3, 1, 2) -> [b, ch, t, f8]
    // This means logical NCHW is [b, 64, 1, 8].
    // Original physical memory is flat 512.
    // Logical idx (n, c, t, w) -> maps to (n, t, w, c) in original view
    // Offset = n*(1*8*64) + t*(8*64) + w*(64) + c*(1)
    // Strides for [N, C, T, W] -> N:512, C:1, T:512, W:64
    auto p3_dims = p3.get_desc().get_dims(); // [1, 64, 1, 8]
    memory::dims permuted_dims = {p3_dims[0], p3_dims[1], p3_dims[2], p3_dims[3]};
    memory::dims permuted_strides = {p3_dims[1]*p3_dims[2]*p3_dims[3], 1, p3_dims[1]*p3_dims[3], p3_dims[1]};
    
    auto src_view_md = memory::desc(permuted_dims, memory::data_type::f32, permuted_strides);
    auto dst_nchw_md = memory::desc(permuted_dims, memory::data_type::f32, memory::format_tag::nchw);
    
    memory reshaped_current = memory(dst_nchw_md, m_engine);
    m_persistent_mems["erb_dec_emb_reshaped"] = reshaped_current;

    auto reorder_pd = reorder::primitive_desc(m_engine, src_view_md, m_engine, dst_nchw_md);
    OneDNNLayer r_layer;
    r_layer.prim = reorder(reorder_pd);
    auto input_view = memory(src_view_md, m_engine, current.get_data_handle());
    r_layer.args = {{DNNL_ARG_FROM, input_view}, {DNNL_ARG_TO, reshaped_current}};
    r_layer.name = "Permute(EmbView->NCHW)";
    m_erb_decoder_layers.push_back(r_layer);

    current = reshaped_current;

    add_binary_add(m_erb_decoder_layers, p3, current, current);
    
    // convt3 is just a Conv2dNormAct (separable) in df3
    add_conv2d(m_erb_decoder_layers, "erb_dec.convt3.0.weight", current, current, 64, 1, 3, 1, 1, 0, 0, 1, 1, 64);
    add_conv2d(m_erb_decoder_layers, "erb_dec.convt3.1.weight", current, current, 64, 1, 1, 0, 0, 0, 0, 1, 1);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.convt3.2", current, current);
    add_relu(m_erb_decoder_layers, current, current);

    // Pathway 2
    memory e2 = safe_at(m_persistent_mems, "enc.erb_block2_out");
    memory p2;
    add_conv2d(m_erb_decoder_layers, "erb_dec.conv2p.0.weight", e2, p2, 64, 1, 1, 0, 0, 0, 0, 1, 1);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.conv2p.1", p2, p2);
    
    add_binary_add(m_erb_decoder_layers, p2, current, current);
    
    // convt2 is ConvTranspose2dNormAct (separable)
    add_conv_transpose2d(m_erb_decoder_layers, "erb_dec.convt2.0.weight", current, current, 64, 1, 3, 1, 1, 0, 0, 1, 2, 64);
    add_conv2d(m_erb_decoder_layers, "erb_dec.convt2.1.weight", current, current, 64, 1, 1, 0, 0, 0, 0, 1, 1);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.convt2.2", current, current);
    add_relu(m_erb_decoder_layers, current, current);

    // Pathway 1
    memory e1 = safe_at(m_persistent_mems, "enc.erb_block1_out");
    memory p1;
    add_conv2d(m_erb_decoder_layers, "erb_dec.conv1p.0.weight", e1, p1, 64, 1, 1, 0, 0, 0, 0, 1, 1);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.conv1p.1", p1, p1);
    
    add_binary_add(m_erb_decoder_layers, p1, current, current);
    
    // convt1 is ConvTranspose2dNormAct (separable)
    add_conv_transpose2d(m_erb_decoder_layers, "erb_dec.convt1.0.weight", current, current, 64, 1, 3, 1, 1, 0, 0, 1, 2, 64);
    add_conv2d(m_erb_decoder_layers, "erb_dec.convt1.1.weight", current, current, 64, 1, 1, 0, 0, 0, 0, 1, 1);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.convt1.2", current, current);
    add_relu(m_erb_decoder_layers, current, current);

    // Pathway 0
    memory e0 = safe_at(m_persistent_mems, "enc.erb_block0_out");
    memory p0;
    add_conv2d(m_erb_decoder_layers, "erb_dec.conv0p.0.weight", e0, p0, 64, 1, 1, 0, 0, 0, 0, 1, 1);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.conv0p.1", p0, p0);
    
    add_binary_add(m_erb_decoder_layers, p0, current, current);
    
    // Output
    add_conv2d(m_erb_decoder_layers, "erb_dec.conv0_out.0.weight", current, current, 1, 1, 3, 1, 1, 0, 0, 1, 1);
    add_batchnorm(m_erb_decoder_layers, "erb_dec.conv0_out.1", current, current);
    add_sigmoid(m_erb_decoder_layers, current, current);

    m_persistent_mems["erb_mask_out"] = current;
    std::cout << "[SUCCESS] ERB Decoder ready." << std::endl;
}

void OneDNNInferenceEngine::setup_df_decoder() {
    std::cout << "[INFO] Building DF Decoder..." << std::endl;
    
    // 1. Squeezed GRU
    memory emb = safe_at(m_persistent_mems, "encoder_emb_out");
    memory dec_emb_out;
    // DF GRU has 2 layers, NO linear_out (out_size=-1), and 8 groups
    add_squeezed_gru(m_df_decoder_layers, "df_dec.df_gru", emb, dec_emb_out, 256, -1, 8, false, 2);

    // 2. Pathway from c0 (encoder)
    memory c0 = safe_at(m_persistent_mems, "enc.df_block0_out");
    memory df_p;
    // Conv2dNormAct(64 -> 10, kt=5, kh=1) -> kt is kernel time
    // Metadata: df_dec.df_convp.1.weight is [10, 32, 5, 1]
    // Wait, why 32? Groups?
    // Let's re-read metadata for df_dec.df_convp.1.weight: shape [10, 32, 5, 1]
    // Input c0 is [1, 64, T, 96]. If groups=2, in_ch=64/2=32. YES.
    add_conv2d(m_df_decoder_layers, "df_dec.df_convp.1.weight", c0, df_p, 10, 5, 1, 0, 0, 2, 2, 1, 1, 2);
    add_conv2d(m_df_decoder_layers, "df_dec.df_convp.2.weight", df_p, df_p, 10, 1, 1, 0, 0, 0, 0, 1, 1);
    add_batchnorm(m_df_decoder_layers, "df_dec.df_convp.3", df_p, df_p);
    
    // 3. DF Out
    memory df_out;
    add_grouped_linear(m_df_decoder_layers, "df_dec.df_out.0.weight", dec_emb_out, df_out, 1);

    m_persistent_mems["df_coefs_out"] = df_out;
    std::cout << "[SUCCESS] DF Decoder ready." << std::endl;
}

void OneDNNInferenceEngine::infer_erb(const float* erb_features, float* output_mask) {
    if (m_encoder_layers.empty()) return;
    
    try {
        auto input_mem = safe_at(m_persistent_mems, "encoder_erb_input");
        m_queue.memcpy(input_mem.get_data_handle(), erb_features, 32 * sizeof(float)).wait();

        for (size_t i = 0; i < m_encoder_layers.size(); ++i) {
            auto& layer = m_encoder_layers[i];
            std::cout << "[DEBUG] Executing encoder layer " << i << ": " << layer.name << std::endl;
            if (layer.custom_exec) {
                layer.custom_exec();
            } else {
                layer.prim.execute(m_stream, layer.args);
            }
        }

        for (size_t i = 0; i < m_erb_decoder_layers.size(); ++i) {
            auto& layer = m_erb_decoder_layers[i];
            std::cout << "[DEBUG] Executing erb_dec layer " << i << ": " << layer.name << std::endl;
            if (layer.custom_exec) {
                layer.custom_exec();
            } else {
                layer.prim.execute(m_stream, layer.args);
            }
        }

        for (size_t i = 0; i < m_df_decoder_layers.size(); ++i) {
            auto& layer = m_df_decoder_layers[i];
            std::cout << "[DEBUG] Executing df_dec layer " << i << ": " << layer.name << std::endl;
            if (layer.custom_exec) {
                layer.custom_exec();
            } else {
                layer.prim.execute(m_stream, layer.args);
            }
        }

        m_stream.wait();
    } catch (const dnnl::error& e) {
        std::cerr << "[ERROR] oneDNN error in infer_erb: " << e.message << std::endl;
        throw;
    }
}

void OneDNNInferenceEngine::test_conv2d_mapping() {}
void OneDNNInferenceEngine::test_batchnorm_mapping() {}
void OneDNNInferenceEngine::test_gru_mapping() {
    std::cout << "[TEST] Testing GRU Mapping..." << std::endl;
    std::vector<OneDNNLayer> test_seq;
    
    // Input: [Batch=1, Time=1, InputSize=256]
    auto input_md = memory::desc({1, 1, 256}, memory::data_type::f32, memory::format_tag::tnc);
    auto input_mem = memory(input_md, m_engine);
    memory output_mem;

    try {
        add_gru(test_seq, "enc.emb_gru", input_mem, output_mem, 256, 1);
        std::cout << "[PASS] GRU Primitive created successfully." << std::endl;
        
        std::vector<float> dummy_in(256, 0.5f);
        m_queue.memcpy(input_mem.get_data_handle(), dummy_in.data(), dummy_in.size() * sizeof(float)).wait();
        
        test_seq[0].prim.execute(m_stream, test_seq[0].args);
        m_stream.wait();
        std::cout << "[PASS] GRU Primitive executed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] GRU Test failed: " << e.what() << std::endl;
        throw;
    }
}
void OneDNNInferenceEngine::test_linear_mapping() {}

} // namespace sa::infrastructure
