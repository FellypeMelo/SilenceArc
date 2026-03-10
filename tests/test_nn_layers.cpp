#include "silence_arc/infrastructure/sycl_accelerator.h"
#include "silence_arc/infrastructure/onednn_inference_engine.h"
#include <vector>
#include <iostream>

using namespace sa::infrastructure;

int main() {
    try {
        std::cout << "[RUN] Basic Encoder Execution Test" << std::endl;
        SYCLAccelerator accel;
        if (!accel.initialize()) {
            std::cerr << "Init failed" << std::endl;
            return 1;
        }
        
        std::cout << "[INFO] Initializing OneDNNInferenceEngine..." << std::endl;
        OneDNNInferenceEngine engine(accel.get_queue(), accel.get_dnnl_engine(), accel.get_dnnl_stream());
        
        std::cout << "[INFO] Loading weights..." << std::endl;
        if (!engine.load_weights("models/df3_weights")) {
            std::cerr << "Weights load failed" << std::endl;
            return 1;
        }

        engine.test_gru_mapping();
        
        std::vector<float> dummy_erb(32, 1.0f);
        std::vector<float> output_mask(32, 0.0f);
        
        std::cout << "[INFO] Calling infer_erb..." << std::endl;
        engine.infer_erb(dummy_erb.data(), output_mask.data());
        std::cout << "[PASS] Encoder inference finished." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[FATAL] Unknown exception" << std::endl;
        return 1;
    }
    
    return 0;
}
