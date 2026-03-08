#ifndef SILENCE_ARC_INFRASTRUCTURE_DEEP_FILTER_ADAPTER_H_
#define SILENCE_ARC_INFRASTRUCTURE_DEEP_FILTER_ADAPTER_H_

#include "silence_arc/domain/noise_suppressor.h"
#include <string>
#include <memory>

namespace silence_arc {
namespace infrastructure {

class DeepFilterAdapter : public domain::INoiseSuppressor {
public:
    DeepFilterAdapter();
    ~DeepFilterAdapter() override;

    bool Init(const std::string& model_path) override;
    size_t GetFrameLength() const override;
    float ProcessFrame(const float* input, float* output) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace infrastructure
} // namespace silence_arc

#endif // SILENCE_ARC_INFRASTRUCTURE_DEEP_FILTER_ADAPTER_H_
