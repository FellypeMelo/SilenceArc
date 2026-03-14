#include "silence_arc/infrastructure/directml_telemetry_provider.h"
#include <iostream>

namespace sa::infrastructure {

struct DirectMLTelemetryProvider::Impl {
    // Stub implementation for now
};

DirectMLTelemetryProvider::DirectMLTelemetryProvider() : m_impl(std::make_unique<Impl>()) {
    std::cout << "[INFO] DirectML Telemetry initialized." << std::endl;
}

DirectMLTelemetryProvider::~DirectMLTelemetryProvider() = default;

TelemetryData DirectMLTelemetryProvider::GetLatestData() {
    std::lock_guard<std::mutex> lock(m_mutex);
    TelemetryData data;
    data.gpu_load = 0.0f;
    data.vram_usage_mb = 0.0f;
    data.processing_latency_ms = m_latency_ms;
    data.device_name = "Intel Arc B580 (DirectML)";
    return data;
}

float DirectMLTelemetryProvider::get_gpu_load() {
    return 0.0f;
}

float DirectMLTelemetryProvider::get_vram_usage() {
    return 0.0f;
}

void DirectMLTelemetryProvider::SetProcessingLatency(float ms) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_latency_ms = ms;
}

} // namespace sa::infrastructure
