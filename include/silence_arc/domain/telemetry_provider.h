#pragma once

namespace sa::domain {

/**
 * @brief Interface for hardware monitoring and telemetry data.
 */
class ITelemetryProvider {
public:
    virtual ~ITelemetryProvider() = default;

    virtual float get_gpu_load() = 0;
    virtual float get_vram_usage() = 0;
};

} // namespace sa::domain
