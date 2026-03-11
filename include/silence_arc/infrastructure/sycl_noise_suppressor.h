#ifndef SILENCE_ARC_INFRASTRUCTURE_SYCL_NOISE_SUPPRESSOR_H_
#define SILENCE_ARC_INFRASTRUCTURE_SYCL_NOISE_SUPPRESSOR_H_

#include "silence_arc/domain/noise_suppressor.h"
#include "silence_arc/infrastructure/sycl_accelerator.h"
#include <memory>

namespace silence_arc {
namespace infrastructure {

class SyclNoiseSuppressor : public domain::INoiseSuppressor {
public:
    SyclNoiseSuppressor();
    ~SyclNoiseSuppressor() override = default;

    bool Init(const std::string& model_path) override;
    size_t GetFrameLength() const override;
    float ProcessFrame(const float* input, float* output) override;
    void SetAttenuationLimit(float limit_db) override;
    void SetDeepFilteringEnabled(bool enabled) override;

private:
};

} // namespace infrastructure
} // namespace silence_arc

#endif // SILENCE_ARC_INFRASTRUCTURE_SYCL_NOISE_SUPPRESSOR_H_
