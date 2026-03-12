#include "silence_arc/infrastructure/ui_manager.h"
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include <d3d11.h>
#include <tchar.h>
#include <string>
#include <iostream>

// Link with libraries
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace sa::infrastructure {

struct UIManager::Impl {
    ID3D11Device* pd3dDevice = nullptr;
    ID3D11DeviceContext* pd3dDeviceContext = nullptr;
    IDXGISwapChain* pSwapChain = nullptr;
    ID3D11RenderTargetView* mainRenderTargetView = nullptr;
    HWND hWnd = nullptr;
    WNDCLASSEXW wc;
    bool should_close = false;

    bool CreateDeviceD3D(HWND hWnd) {
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
        sd.OutputWindow = hWnd;
        sd.SampleDesc.Count = 1;
        sd.SampleDesc.Quality = 0;
        sd.Windowed = TRUE;
        sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

        UINT createDeviceFlags = 0;
        D3D_FEATURE_LEVEL featureLevel;
        const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0, };
        HRESULT res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &pSwapChain, &pd3dDevice, &featureLevel, &pd3dDeviceContext);
        if (res == DXGI_ERROR_UNSUPPORTED)
            res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &pSwapChain, &pd3dDevice, &featureLevel, &pd3dDeviceContext);
        if (res != S_OK) return false;

        CreateRenderTarget();
        return true;
    }

    void CleanupDeviceD3D() {
        CleanupRenderTarget();
        if (pSwapChain) { pSwapChain->Release(); pSwapChain = nullptr; }
        if (pd3dDeviceContext) { pd3dDeviceContext->Release(); pd3dDeviceContext = nullptr; }
        if (pd3dDevice) { pd3dDevice->Release(); pd3dDevice = nullptr; }
    }

    void CreateRenderTarget() {
        ID3D11Texture2D* pBackBuffer;
        pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
        pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &mainRenderTargetView);
        pBackBuffer->Release();
    }

    void CleanupRenderTarget() {
        if (mainRenderTargetView) { mainRenderTargetView->Release(); mainRenderTargetView = nullptr; }
    }
};

static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
        case WM_SIZE:
            return 0;
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU) return 0;
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}

UIManager::UIManager() : m_impl(std::make_unique<Impl>()) {}

UIManager::~UIManager() { Shutdown(); }

bool UIManager::Init(const std::string& title, int width, int height) {
    m_impl->wc = { sizeof(m_impl->wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, L"SilenceArcClass", nullptr };
    RegisterClassExW(&m_impl->wc);
    
    std::wstring wtitle(title.begin(), title.end());
    m_impl->hWnd = CreateWindowW(m_impl->wc.lpszClassName, wtitle.c_str(), WS_OVERLAPPEDWINDOW, 100, 100, width, height, nullptr, nullptr, m_impl->wc.hInstance, nullptr);

    if (!m_impl->CreateDeviceD3D(m_impl->hWnd)) {
        m_impl->CleanupDeviceD3D();
        UnregisterClassW(m_impl->wc.lpszClassName, m_impl->wc.hInstance);
        return false;
    }

    ShowWindow(m_impl->hWnd, SW_SHOWDEFAULT);
    UpdateWindow(m_impl->hWnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplWin32_Init(m_impl->hWnd);
    ImGui_ImplDX11_Init(m_impl->pd3dDevice, m_impl->pd3dDeviceContext);

    return true;
}

void UIManager::Shutdown() {
    if (!m_impl->pd3dDevice) return;
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    m_impl->CleanupDeviceD3D();
    DestroyWindow(m_impl->hWnd);
    UnregisterClassW(m_impl->wc.lpszClassName, m_impl->wc.hInstance);
}

bool UIManager::ShouldClose() const {
    MSG msg;
    while (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
        if (msg.message == WM_QUIT) return true;
    }
    return false;
}

void UIManager::BeginFrame() {
    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
}

void UIManager::Render() {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("Silence Arc Control", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);

    ImGui::Text("DeepFilterNet3 Status: Active");
    ImGui::Checkbox("Enable Noise Suppression", &m_state.noise_suppression_enabled);
    ImGui::SliderFloat("Attenuation (dB)", &m_state.suppression_limit_db, 0.0f, 100.0f);

    ImGui::Separator();
    ImGui::Text("Audio Devices");
    
    // Input Device Selection
    if (m_state.input_devices.empty()) {
        ImGui::TextDisabled("No input devices found.");
    } else {
        const char* current_input = m_state.selected_input_device >= 0 ? m_state.input_devices[m_state.selected_input_device].name.c_str() : "Select Input...";
        if (ImGui::BeginCombo("Input Device", current_input)) {
            for (int i = 0; i < (int)m_state.input_devices.size(); i++) {
                bool is_selected = (m_state.selected_input_device == i);
                if (ImGui::Selectable(m_state.input_devices[i].name.c_str(), is_selected))
                    m_state.selected_input_device = i;
            }
            ImGui::EndCombo();
        }
    }

    // Output Device Selection
    if (m_state.output_devices.empty()) {
        ImGui::TextDisabled("No output devices found.");
    } else {
        const char* current_output = m_state.selected_output_device >= 0 ? m_state.output_devices[m_state.selected_output_device].name.c_str() : "Select Output...";
        if (ImGui::BeginCombo("Output Device", current_output)) {
            for (int i = 0; i < (int)m_state.output_devices.size(); i++) {
                bool is_selected = (m_state.selected_output_device == i);
                if (ImGui::Selectable(m_state.output_devices[i].name.c_str(), is_selected))
                    m_state.selected_output_device = i;
            }
            ImGui::EndCombo();
        }
    }

    ImGui::End();
}

void UIManager::EndFrame() {
    ImGui::Render();
    const float clear_color_with_alpha[4] = { 0.45f, 0.55f, 0.60f, 1.00f };
    m_impl->pd3dDeviceContext->OMSetRenderTargets(1, &m_impl->mainRenderTargetView, nullptr);
    m_impl->pd3dDeviceContext->ClearRenderTargetView(m_impl->mainRenderTargetView, clear_color_with_alpha);
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
    m_impl->pSwapChain->Present(1, 0); 
}

void UIManager::UpdateTelemetry(const TelemetryData& data) {
}

void UIManager::UpdateSignalLevels(float input, float output, float reduction) {
}

} // namespace sa::infrastructure
