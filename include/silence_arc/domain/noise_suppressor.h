#ifndef SILENCE_ARC_DOMAIN_NOISE_SUPPRESSOR_H_
#define SILENCE_ARC_DOMAIN_NOISE_SUPPRESSOR_H_

#include <vector>
#include <string>

namespace silence_arc {
namespace domain {

class INoiseSuppressor {
public:
    virtual ~INoiseSuppressor() = default;

    virtual bool Init(const std::string& model_path) = 0;
    virtual size_t GetFrameLength() const = 0;
    virtual float ProcessFrame(const float* input, float* output) = 0;
    virtual void SetAttenuationLimit(float limit_db) = 0;
};

} // namespace domain
} // namespace silence_arc

#endif // SILENCE_ARC_DOMAIN_NOISE_SUPPRESSOR_H_
