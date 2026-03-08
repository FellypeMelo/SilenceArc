#include "silence_arc/infrastructure/ui_manager.h"

#include <d3d11.h>
#include <dwmapi.h>
#include <shellapi.h>
#include <tchar.h>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "shell32.lib")

// Tray icon constants
#define WM_TRAYICON (WM_USER + 1)
#define ID_TRAY_APP_ICON 1001
#define ID_TRAY_EXIT 1002
#define ID_TRAY_RESTORE 1003

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace silence_arc {
namespace infrastructure {

static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) {
    return true;
  }

  UIManager* manager = reinterpret_cast<UIManager*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

  switch (msg) {
    case WM_SIZE:
      if (wParam == SIZE_MINIMIZED) {
        if (manager) {
          manager->ShowWindow(false);
        }
        return 0;
      }
      break;
    case WM_SYSCOMMAND:
      if ((wParam & 0xfff0) == SC_KEYMENU) { // Disable ALT application menu
        return 0;
      }
      break;
    case WM_TRAYICON:
      if (LOWORD(lParam) == WM_LBUTTONUP) {
        if (manager) {
          manager->ShowWindow(true);
        }
      } else if (LOWORD(lParam) == WM_RBUTTONUP) {
        POINT curPoint;
        GetCursorPos(&curPoint);
        HMENU hMenu = CreatePopupMenu();
        InsertMenu(hMenu, 0, MF_BYPOSITION | MF_STRING, ID_TRAY_RESTORE, L"Restore");
        InsertMenu(hMenu, 1, MF_BYPOSITION | MF_STRING, ID_TRAY_EXIT, L"Exit");
        SetForegroundWindow(hWnd);
        TrackPopupMenu(hMenu, TPM_BOTTOMALIGN | TPM_LEFTALIGN, curPoint.x, curPoint.y, 0, hWnd, NULL);
        DestroyMenu(hMenu);
      }
      break;
    case WM_COMMAND:
      if (LOWORD(wParam) == ID_TRAY_EXIT) {
        ::PostQuitMessage(0);
      } else if (LOWORD(wParam) == ID_TRAY_RESTORE) {
        if (manager) {
          manager->ShowWindow(true);
        }
      }
      break;
    case WM_DESTROY:
      ::PostQuitMessage(0);
      return 0;
  }
  return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}

UIManager::UIManager() {}

UIManager::~UIManager() {
  Shutdown();
}

bool UIManager::Init(const std::string& window_title, int width, int height) {
  if (is_initialized_) {
    return true;
  }

  width_ = width;
  height_ = height;

  // 1. Create Window
  ImGui_ImplWin32_EnableDpiAwareness();
  WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, L"SilenceArcGUI", nullptr };
  ::RegisterClassExW(&wc);
  
  std::wstring w_title(window_title.begin(), window_title.end());
  hwnd_ = ::CreateWindowW(wc.lpszClassName, w_title.c_str(), WS_OVERLAPPEDWINDOW, 100, 100, width_, height_, nullptr, nullptr, wc.hInstance, nullptr);

  if (!hwnd_) {
    return false;
  }

  // Store instance pointer for WndProc
  SetWindowLongPtr(static_cast<HWND>(hwnd_), GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

  // 2. Init D3D
  if (!CreateDeviceD3D()) {
    CleanupDeviceD3D();
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return false;
  }

  // 3. Show Window
  ::ShowWindow(static_cast<HWND>(hwnd_), SW_SHOWDEFAULT);
  ::UpdateWindow(static_cast<HWND>(hwnd_));

  // 4. Setup ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO(); (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  // --- Intel Arc Branded Styling ---
  ImGuiStyle& style = ImGui::GetStyle();
  ImVec4* colors = style.Colors;

  style.WindowRounding = 0.0f;
  style.ChildRounding = 0.0f;
  style.FrameRounding = 2.0f;
  style.GrabRounding = 2.0f;
  style.WindowBorderSize = 1.0f;

  // Colors - Arc Blue / Silver theme
  colors[ImGuiCol_Text]                   = ImVec4(0.95f, 0.96f, 0.98f, 1.00f);
  colors[ImGuiCol_WindowBg]               = ImVec4(0.06f, 0.06f, 0.08f, 0.94f);
  colors[ImGuiCol_Border]                 = ImVec4(0.12f, 0.45f, 0.85f, 0.50f); // Arc Blue Border
  colors[ImGuiCol_FrameBg]                = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
  colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.15f, 0.15f, 0.18f, 1.00f);
  colors[ImGuiCol_FrameBgActive]          = ImVec4(0.20f, 0.20f, 0.25f, 1.00f);
  colors[ImGuiCol_TitleBg]                = ImVec4(0.04f, 0.04f, 0.04f, 1.00f);
  colors[ImGuiCol_TitleBgActive]          = ImVec4(0.00f, 0.40f, 0.80f, 1.00f); // Arc Blue Title
  colors[ImGuiCol_CheckMark]              = ImVec4(0.12f, 0.45f, 0.85f, 1.00f); // Arc Blue Check
  colors[ImGuiCol_SliderGrab]             = ImVec4(0.12f, 0.45f, 0.85f, 1.00f);
  colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.20f, 0.55f, 0.95f, 1.00f);
  colors[ImGuiCol_Button]                 = ImVec4(0.12f, 0.45f, 0.85f, 0.40f);
  colors[ImGuiCol_ButtonHovered]          = ImVec4(0.12f, 0.45f, 0.85f, 1.00f);
  colors[ImGuiCol_ButtonActive]           = ImVec4(0.00f, 0.35f, 0.75f, 1.00f);
  colors[ImGuiCol_Header]                 = ImVec4(0.12f, 0.45f, 0.85f, 0.31f);
  colors[ImGuiCol_HeaderHovered]          = ImVec4(0.12f, 0.45f, 0.85f, 0.80f);
  colors[ImGuiCol_HeaderActive]           = ImVec4(0.12f, 0.45f, 0.85f, 1.00f);
  colors[ImGuiCol_PlotHistogram]          = ImVec4(0.12f, 0.45f, 0.85f, 1.00f); // Arc Blue Bars
  colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(0.20f, 0.55f, 0.95f, 1.00f);

  ImGui_ImplWin32_Init(static_cast<HWND>(hwnd_));
  ImGui_ImplDX11_Init(static_cast<ID3D11Device*>(device_), static_cast<ID3D11DeviceContext*>(device_context_));

  CreateTrayIcon();

  is_initialized_ = true;
  return true;
}

void UIManager::Shutdown() {
  if (!is_initialized_) {
    return;
  }

  DestroyTrayIcon();

  ImGui_ImplDX11_Shutdown();
  ImGui_ImplWin32_Shutdown();
  ImGui::DestroyContext();

  CleanupDeviceD3D();
  
  if (hwnd_) {
    ::DestroyWindow(static_cast<HWND>(hwnd_));
    ::UnregisterClassW(L"SilenceArcGUI", GetModuleHandle(nullptr));
    hwnd_ = nullptr;
  }
  
  is_initialized_ = false;
}

void UIManager::ShowWindow(bool show) {
  if (!hwnd_) {
    return;
  }
  HWND native_hwnd = static_cast<HWND>(hwnd_);
  if (show) {
    ::ShowWindow(native_hwnd, SW_RESTORE);
    ::SetForegroundWindow(native_hwnd);
    is_minimized_to_tray_ = false;
  } else {
    ::ShowWindow(native_hwnd, SW_HIDE);
    is_minimized_to_tray_ = true;
  }
}

void UIManager::CreateTrayIcon() {
  NOTIFYICONDATAW nid = { sizeof(nid) };
  nid.hWnd = static_cast<HWND>(hwnd_);
  nid.uID = ID_TRAY_APP_ICON;
  nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
  nid.uCallbackMessage = WM_TRAYICON;
  nid.hIcon = LoadIcon(NULL, IDI_APPLICATION); // Generic icon for now
  wcscpy_s(nid.szTip, L"Silence Arc");
  Shell_NotifyIconW(NIM_ADD, &nid);
}

void UIManager::DestroyTrayIcon() {
  NOTIFYICONDATAW nid = { sizeof(nid) };
  nid.hWnd = static_cast<HWND>(hwnd_);
  nid.uID = ID_TRAY_APP_ICON;
  Shell_NotifyIconW(NIM_DELETE, &nid);
}

bool UIManager::CreateDeviceD3D() {
  DXGI_SWAP_CHAIN_DESC sd;
  ZeroMemory(&sd, sizeof(sd));
  sd.BufferCount = 2;
  sd.BufferDesc.Width = 0;
  sd.BufferDesc.Height = 0;
  sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  sd.BufferDesc.RefreshRate.Numerator = 60;
  sd.BufferDesc.RefreshRate.Denominator = 1;
  sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
  sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  sd.OutputWindow = static_cast<HWND>(hwnd_);
  sd.SampleDesc.Count = 1;
  sd.SampleDesc.Quality = 0;
  sd.Windowed = TRUE;
  sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

  UINT createDeviceFlags = 0;
  D3D_FEATURE_LEVEL featureLevel;
  const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0, };
  
  IDXGISwapChain* swapChain = nullptr;
  ID3D11Device* device = nullptr;
  ID3D11DeviceContext* context = nullptr;

  HRESULT res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &swapChain, &device, &featureLevel, &context);
  if (res == DXGI_ERROR_UNSUPPORTED) {
    res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &swapChain, &device, &featureLevel, &context);
  }
  
  if (res != S_OK) {
    return false;
  }

  swap_chain_ = swapChain;
  device_ = device;
  device_context_ = context;

  CreateRenderTarget();
  return true;
}

void UIManager::CleanupDeviceD3D() {
  CleanupRenderTarget();
  if (swap_chain_) {
    static_cast<IDXGISwapChain*>(swap_chain_)->Release();
    swap_chain_ = nullptr;
  }
  if (device_context_) {
    static_cast<ID3D11DeviceContext*>(device_context_)->Release();
    device_context_ = nullptr;
  }
  if (device_) {
    static_cast<ID3D11Device*>(device_)->Release();
    device_ = nullptr;
  }
}

void UIManager::CreateRenderTarget() {
  ID3D11Texture2D* pBackBuffer;
  static_cast<IDXGISwapChain*>(swap_chain_)->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
  ID3D11RenderTargetView* rtv = nullptr;
  static_cast<ID3D11Device*>(device_)->CreateRenderTargetView(pBackBuffer, nullptr, &rtv);
  render_target_view_ = rtv;
  pBackBuffer->Release();
}

void UIManager::CleanupRenderTarget() {
  if (render_target_view_) {
    static_cast<ID3D11RenderTargetView*>(render_target_view_)->Release();
    render_target_view_ = nullptr;
  }
}

bool UIManager::ShouldClose() const {
  MSG msg;
  while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
    ::TranslateMessage(&msg);
    ::DispatchMessage(&msg);
    if (msg.message == WM_QUIT) {
      return true;
    }
  }
  return false;
}

void UIManager::SetTransparency(float alpha) {
  transparency_ = alpha;
  if (hwnd_) {
    HWND native_hwnd = static_cast<HWND>(hwnd_);
    // Simple alpha blending for the whole window
    if (alpha < 1.0f) {
      ::SetWindowLong(native_hwnd, GWL_EXSTYLE, ::GetWindowLong(native_hwnd, GWL_EXSTYLE) | WS_EX_LAYERED);
      ::SetLayeredWindowAttributes(native_hwnd, 0, (BYTE)(255 * alpha), LWA_ALPHA);
    } else {
      ::SetWindowLong(native_hwnd, GWL_EXSTYLE, ::GetWindowLong(native_hwnd, GWL_EXSTYLE) & ~WS_EX_LAYERED);
    }
  }
}

void UIManager::UpdateTelemetry(domain::TelemetryData data) {
  state_.gpu_utilization = data.gpu_utilization;
  state_.processing_latency_ms = data.processing_latency_ms;
  state_.memory_footprint_mb = data.memory_footprint_mb;
}

void UIManager::BeginFrame() {
  if (!is_initialized_) {
    return;
  }
  ImGui_ImplDX11_NewFrame();
  ImGui_ImplWin32_NewFrame();
  ImGui::NewFrame();
}

void UIManager::Render() {
  if (!is_initialized_) {
    return;
  }

  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2((float)width_, (float)height_));
  ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

  ImGui::Begin("Silence Arc - Main", nullptr, window_flags);

  // 1. Header
  ImGui::TextColored(ImVec4(0.0f, 0.5f, 1.0f, 1.0f), "SILENCE ARC");
  ImGui::Separator();

  // 2. Main Controls
  if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Noise Suppression", &state_.noise_suppression_enabled);
    ImGui::Checkbox("Voice Enhancement", &state_.voice_enhancement_enabled);
  }

  // 3. Monitoring
  if (ImGui::CollapsingHeader("Monitoring", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Input Level");
    ImGui::ProgressBar(state_.input_level, ImVec2(-1.0f, 0.0f));

    ImGui::Text("Output Level");
    ImGui::ProgressBar(state_.output_level, ImVec2(-1.0f, 0.0f));

    ImGui::Text("Suppression Depth");
    ImGui::ProgressBar(state_.suppression_depth, ImVec2(-1.0f, 0.0f));
  }

  // 4. Configuration
  if (ImGui::CollapsingHeader("Configuration")) {
    const char* apis[] = { "WASAPI", "ASIO" };
    ImGui::Combo("Audio API", &state_.selected_audio_api, apis, IM_ARRAYSIZE(apis));

    if (ImGui::BeginCombo("Input Device", state_.selected_input_device >= 0 ? state_.input_devices[state_.selected_input_device].name.c_str() : "Select...")) {
      for (int i = 0; i < (int)state_.input_devices.size(); i++) {
        bool is_selected = (state_.selected_input_device == i);
        if (ImGui::Selectable(state_.input_devices[i].name.c_str(), is_selected)) {
          state_.selected_input_device = i;
        }
        if (is_selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }

    if (ImGui::BeginCombo("Output Device", state_.selected_output_device >= 0 ? state_.output_devices[state_.selected_output_device].name.c_str() : "Select...")) {
      for (int i = 0; i < (int)state_.output_devices.size(); i++) {
        bool is_selected = (state_.selected_output_device == i);
        if (ImGui::Selectable(state_.output_devices[i].name.c_str(), is_selected)) {
          state_.selected_output_device = i;
        }
        if (is_selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }

    if (ImGui::SliderFloat("Window Transparency", &transparency_, 0.1f, 1.0f, "%.2f")) {
      SetTransparency(transparency_);
    }
  }

  // 5. Telemetry (Brief for now)
  if (ImGui::CollapsingHeader("Telemetry")) {
    ImGui::Text("GPU Util: %.1f%%", state_.gpu_utilization * 100.0f);
    ImGui::Text("Latency: %.2f ms", state_.processing_latency_ms);
    ImGui::Text("VRAM: %.1f MB", state_.memory_footprint_mb);
  }

  ImGui::End();
}


void UIManager::EndFrame() {
  if (!is_initialized_) {
    return;
  }
  ImGui::Render();
  
  const float clear_color_with_alpha[4] = { 0.45f * transparency_, 0.55f * transparency_, 0.60f * transparency_, transparency_ };
  static_cast<ID3D11DeviceContext*>(device_context_)->OMSetRenderTargets(1, reinterpret_cast<ID3D11RenderTargetView**>(&render_target_view_), nullptr);
  static_cast<ID3D11DeviceContext*>(device_context_)->ClearRenderTargetView(static_cast<ID3D11RenderTargetView*>(render_target_view_), clear_color_with_alpha);
  ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

  static_cast<IDXGISwapChain*>(swap_chain_)->Present(1, 0);
}

} // namespace infrastructure
} // namespace silence_arc
