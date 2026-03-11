#include "silence_arc/infrastructure/deep_filter_adapter.h"
#include "deep_filter.h"
#include <stdexcept>
#include <filesystem>

namespace silence_arc {
namespace infrastructure {

struct DeepFilterAdapter::Impl {
    DFState* state = nullptr;
    size_t frame_length = 0;
};

DeepFilterAdapter::DeepFilterAdapter() : impl_(std::make_unique<Impl>()) {}

DeepFilterAdapter::~DeepFilterAdapter() {
    if (impl_->state) {
        df_free(impl_->state);
    }
}

bool DeepFilterAdapter::Init(const std::string& model_path) {
    if (impl_->state) {
        df_free(impl_->state);
        impl_->state = nullptr;
    }

    if (!std::filesystem::exists(model_path)) {
        return false;
    }

    // Default attenuation limit 40.0 for better noise removal.
    // Note: df_create might still panic if the file is not a valid model.
    impl_->state = df_create(model_path.c_str(), 40.0f, nullptr);
    if (!impl_->state) {
        return false;
    }

    impl_->frame_length = df_get_frame_length(impl_->state);
    return true;
}

size_t DeepFilterAdapter::GetFrameLength() const {
    return impl_->frame_length;
}

float DeepFilterAdapter::ProcessFrame(const float* input, float* output) {
    if (!impl_->state) {
        return -100.0f; // Error code or default SNR
    }
    // We cast to float* because the C API expects a mutable pointer for the input 
    // (though it's usually treated as const internally if not used for in-place)
    // Looking at capi.rs: input: *mut c_float
    return df_process_frame(impl_->state, const_cast<float*>(input), output);
}

void DeepFilterAdapter::SetAttenuationLimit(float limit_db) {
    if (impl_->state) {
        df_set_atten_lim(impl_->state, limit_db);
    }
}

} // namespace infrastructure
} // namespace silence_arc
