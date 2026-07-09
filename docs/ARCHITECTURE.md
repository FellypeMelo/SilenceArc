# ARCHITECTURE — SilenceArc

Pipeline de áudio real-time. 48kHz, mono, frames de 480 samples (10ms de hop). Duas engines, uma seam.

## Visão geral

3 domínios:
1. **C++ Host:** WASAPI (miniaudio), GUI (ImGui/DX11), threading, seleção de backend.
2. **GPU (SYCL/oneAPI):** STFT + features + rede DFN3 + deep filtering + ISTFT. Tudo na Arc. Sem OpenVINO.
3. **Rust core (`df.dll`):** fallback CPU. DeepFilterNet3 via tract/ONNX, exposto por C-API (`deep_filter.h`).

## Pipeline (captura → modelo → saída)

```
Mic (WASAPI shared, f32, mono, 48kHz)
  └─ MiniaudioPipeline::DataCallback  ← thread real-time. SHIM fino, nunca processa.
       ├─ AudioStreamBuffer in_buffer (framing p/ blocos de 480)
       ├─ AsyncAudioPipeline::PushInput (fila bounded, depth 8, drop-oldest)
       │     └─ Worker thread TIME_CRITICAL
       │           └─ INoiseSuppressor::ProcessFrame(480 in, 480 out)
       │                 ├─ GPU: SyclNoiseSuppressor → sycl_process()
       │                 └─ CPU: DeepFilterAdapter → df_process_frame()
       ├─ AsyncAudioPipeline::PopOutput → out_buffer
       └─ escreve exatamente frameCount no pOutput (falta = zero-fill)
Speaker (mesmo device duplex)
```

NÃO existe device virtual. Entrada = mic real, saída = speaker real. Discord/OBS não enxergam o áudio limpo. Gap conhecido (STATE.md).

## Caminho GPU (SYCLAccelerator::process_frame)

Fila SYCL in-order, buffers USM device. Um sync terminal por frame.

1. **Análise:** rola janela de 960 (hop 480), aplica janela sin², FFT real via oneMKL DFT (wnorm embutido no FORWARD_SCALE).
2. **Features:** power spectrum (481 bins) → 32 bandas ERB (matriz `erb_fb.bin`); normalização EMA α=0.99 no HOST (log10 ERB mean-norm + unit-norm complexa dos 96 bins DF).
3. **Inferência:** `OneDNNInferenceEngine::infer()` — encoder + ERB decoder + DF decoder em primitivas oneDNN (conv2d, lbr_gru, deconv, binary, concat) + kernels SYCL custom p/ permutes TNC↔NCHW. Estados persistentes: GRU states, janelas causais rolantes (erb 3 frames, df 3, c0 5). 133 tensores de `models/df3_weights/`.
4. **Filtro:** mask ERB → 481 bins (matriz inversa `mask_erb_inv_fb.bin`) × spectrum; deep filtering ordem 5 (960 coefs complexos) nos 96 bins baixos; histórico espectral rolado no device.
5. **Síntese:** IFFT + overlap-add com janela → 480 samples out.

## Caminho CPU (fallback)

`DeepFilterAdapter` → `df.dll` (Rust, vendorado em `DeepFilterNet/`). Modelo `DeepFilterNet3_onnx.tar.gz`. Mesma interface `INoiseSuppressor`.

## Camadas (Clean Architecture)

- **domain/** (`include/silence_arc/domain/`): só contratos + tipos. Zero SYCL/oneDNN/WASAPI.
  - `INoiseSuppressor` — seam ÚNICA de seleção de backend (ADR-002).
  - `IAudioPipeline`, `ITelemetryProvider`, `AudioStreamBuffer`, `AudioMetrics`, `UIState`.
- **infrastructure/**: implementações.
  - `MiniaudioPipeline` (device duplex + callback), `AsyncAudioPipeline` (worker + filas bounded),
    `MiniaudioDeviceManager` (enumeração), `SyclTelemetryProvider` (Level Zero), `UIManager` (ImGui/DX11 + system tray).
  - Bridge interno da GPU (privado, NÃO é seam): `GPUAccelerator`/`SYCLAccelerator` (DSP) + `NeuralNetworkModel`/`OneDNNInferenceEngine` (NN).
- **presentation:** `main.cpp` (composição, loop ~60fps) + `ui_manager.cpp`.

## Threads

| Thread | Papel | Regra |
|---|---|---|
| Audio callback (miniaudio) | shim push/pop | nunca bloqueia, nunca infere |
| Worker (AsyncAudioPipeline) | ProcessFrame GPU/CPU | TIME_CRITICAL; filas drop-oldest, depth 8 |
| UI (main) | ImGui, telemetria, troca de device | ~60fps; escreve UIState lido pelo worker |

## Memória / sync

USM device em tudo no hot path. Fila in-order = kernels auto-encadeiam; host só espera onde PRECISA ler: cópia de features p/ host (EMA + parte da normalização rodam no host), cópia mask/coefs pós-inferência, cópia final do output. Detalhe das primitivas: `docs/ENGINE.md`. Decisões: `docs/DECISIONS.md` + `docs/adr/`.
