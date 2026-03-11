#include "silence_arc/infrastructure/sycl_telemetry_provider.h"
#include <level_zero/ze_api.h>
#include <iostream>
#include <algorithm>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#else
#include <stdlib.h>
#endif

namespace silence_arc {
namespace infrastructure {

SyclTelemetryProvider::SyclTelemetryProvider() {
    // Ensure Sysman is enabled
#ifdef _WIN32
    SetEnvironmentVariableA("ZES_ENABLE_SYSMAN", "1");
#else
    setenv("ZES_ENABLE_SYSMAN", "1", 1);
#endif

    InitializeSysman();

    // Start background polling
    run_polling_ = true;
    worker_thread_ = std::thread(&SyclTelemetryProvider::PollingLoop, this);
}

SyclTelemetryProvider::~SyclTelemetryProvider() {
    run_polling_ = false;
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void SyclTelemetryProvider::InitializeSysman() {
    try {
        // 1. Get SYCL device (Intel Arc preferred)
        sycl::device device;
        bool found = false;
        
        // We look for platforms explicitly
        auto platforms = sycl::platform::get_platforms();
        for (auto& platform : platforms) {
            // Check if backend is Level Zero
            if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
            
            auto devices = platform.get_devices();
            for (auto& dev : devices) {
                if (dev.is_gpu() && dev.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) {
                    device = dev;
                    found = true;
                    break;
                }
            }
            if (found) break;
        }

        if (!found) {
            // Fallback to any Intel GPU on Level Zero
            for (auto& platform : platforms) {
                if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
                auto devices = platform.get_devices();
                for (auto& dev : devices) {
                    if (dev.is_gpu() && dev.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos) {
                        device = dev;
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
        }

        if (!found) {
            std::cerr << "[Telemetry] No Intel GPU with Level Zero backend found." << std::endl;
            return;
        }

        // 2. Get native handle
        ze_device_handle_t hDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
        if (!hDevice) {
             std::cerr << "[Telemetry] Failed to get native Level Zero device handle." << std::endl;
             return;
        }
        hSysmanDevice = reinterpret_cast<zes_device_handle_t>(hDevice);

        // 3. Enumerate Engine Groups for GPU Load
        uint32_t numEngines = 0;
        if (zesDeviceEnumEngineGroups(hSysmanDevice, &numEngines, nullptr) == ZE_RESULT_SUCCESS && numEngines > 0) {
            std::vector<zes_engine_handle_t> hEngines(numEngines);
            if (zesDeviceEnumEngineGroups(hSysmanDevice, &numEngines, hEngines.data()) == ZE_RESULT_SUCCESS) {
                for (auto hEngine : hEngines) {
                    zes_engine_properties_t props = {ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES};
                    if (zesEngineGetProperties(hEngine, &props) == ZE_RESULT_SUCCESS) {
                        if (props.type == ZES_ENGINE_GROUP_ALL) {
                            hEngineAll = hEngine;
                            break;
                        }
                    }
                }
                // Fallback to first engine if "ALL" not found
                if (!hEngineAll && !hEngines.empty()) hEngineAll = hEngines[0];
            }
        }

        // 4. Enumerate Memory Modules for VRAM
        uint32_t numMem = 0;
        if (zesDeviceEnumMemoryModules(hSysmanDevice, &numMem, nullptr) == ZE_RESULT_SUCCESS && numMem > 0) {
            std::vector<zes_mem_handle_t> hMems(numMem);
            if (zesDeviceEnumMemoryModules(hSysmanDevice, &numMem, hMems.data()) == ZE_RESULT_SUCCESS) {
                hMainMemory = hMems[0]; // Assume first one is the main discrete VRAM
            }
        }

        sysman_initialized_ = true;
        std::cout << "[Telemetry] Sysman initialized for: " << device.get_info<sycl::info::device::name>() << std::endl;

    } catch (const sycl::exception& e) {
        std::cerr << "[Telemetry] SYCL Exception during initialization: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Telemetry] Initialization error: " << e.what() << std::endl;
    }
}

domain::TelemetryData SyclTelemetryProvider::GetLatestData() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return data_;
}

void SyclTelemetryProvider::Update() {
    if (!sysman_initialized_ || !hSysmanDevice) return;

    domain::TelemetryData updated_data;

    // 1. GPU Utilization
    if (hEngineAll) {
        zes_engine_stats_t stats = {0};
        if (zesEngineGetActivity(hEngineAll, &stats) == ZE_RESULT_SUCCESS) {
            if (last_timestamp_ != 0 && stats.timestamp > last_timestamp_) {
                uint64_t delta_active = stats.activeTime - last_engine_stats_.activeTime;
                uint64_t delta_total = stats.timestamp - last_timestamp_;
                
                if (delta_total > 0) {
                    float utilization = static_cast<float>(delta_active) / static_cast<float>(delta_total);
                    updated_data.gpu_utilization = std::clamp(utilization, 0.0f, 1.0f);
                }
            }
            last_engine_stats_ = stats;
            last_timestamp_ = stats.timestamp;
        }
    }

    // 2. VRAM Usage
    if (hMainMemory) {
        zes_mem_state_t state = {ZES_STRUCTURE_TYPE_MEM_STATE};
        if (zesMemoryGetState(hMainMemory, &state) == ZE_RESULT_SUCCESS) {
            if (state.size > 0) {
                uint64_t used = state.size - state.free;
                updated_data.memory_footprint_mb = static_cast<float>(used) / (1024.0f * 1024.0f);
            }
        }
    }

    // 3. Audio Latency (from external setter)
    updated_data.processing_latency_ms = external_latency_ms_.load();

    // Atomic update of public data
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        data_ = updated_data;
    }
}

void SyclTelemetryProvider::PollingLoop() {
    while (run_polling_) {
        auto start = std::chrono::steady_clock::now();
        
        try {
            Update();
        } catch (...) {
            // Silently catch all exceptions in polling loop to prevent crashes
        }
        
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Target ~30Hz (33ms) instead of 60Hz to reduce overhead and potential for race conditions
        auto sleep_time = std::chrono::milliseconds(33) - elapsed;
        if (sleep_time > std::chrono::milliseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        }
    }
}

void SyclTelemetryProvider::SetProcessingLatency(float latency_ms) {
    external_latency_ms_.store(latency_ms);
}

} // namespace infrastructure
} // namespace silence_arc
