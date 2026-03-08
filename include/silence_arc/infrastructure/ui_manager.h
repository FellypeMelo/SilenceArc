#ifndef SILENCE_ARC_INFRASTRUCTURE_UI_MANAGER_H_
#define SILENCE_ARC_INFRASTRUCTURE_UI_MANAGER_H_

#include "silence_arc/domain/ui_state.h"
#include <memory>
#include <string>

namespace silence_arc {
namespace infrastructure {

class UIManager {
public:
    UIManager();
    ~UIManager();

    bool Init(const std::string& window_title, int width, int height);
    void Shutdown();

    void BeginFrame();
    void Render();
    void EndFrame();

    bool ShouldClose() const;
    void SetTransparency(float alpha);

    bool IsInitialized() const { return is_initialized_; }
    domain::UIState& GetState() { return state_; }

    void UpdateTelemetry(domain::TelemetryData data);

    void ShowWindow(bool show);
    bool IsMinimizedToTray() const { return is_minimized_to_tray_; }

private:
    bool CreateDeviceD3D();
    void CleanupDeviceD3D();
    void CreateRenderTarget();
    void CleanupRenderTarget();

    void CreateTrayIcon();
    void DestroyTrayIcon();

    bool is_initialized_ = false;
    bool is_minimized_to_tray_ = false;
    domain::UIState state_;
    void* hwnd_ = nullptr;
    void* device_ = nullptr;
    void* device_context_ = nullptr;
    void* swap_chain_ = nullptr;
    void* render_target_view_ = nullptr;
    
    int width_ = 1280;
    int height_ = 800;
    float transparency_ = 1.0f;
};

} // namespace infrastructure
} // namespace silence_arc

#endif // SILENCE_ARC_INFRASTRUCTURE_UI_MANAGER_H_
