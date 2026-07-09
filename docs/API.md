# API — SilenceArc

Interfaces atuais. Não é lib pública instalável — são as seams internas + duas C-APIs `extern "C"`. Mudou assinatura aqui → atualizar este arquivo (gate em QUALITY.md).

## 1. Domain (contratos, `include/silence_arc/domain/`)

### `INoiseSuppressor` (noise_suppressor.h) — seam de backend
```cpp
bool   Init(const std::string& model_path);      // GPU ignora path (usa models/df3_weights); CPU usa tar.gz ONNX
size_t GetFrameLength() const;                    // 480 em ambos backends
float  ProcessFrame(const float* in, float* out); // 480 in → 480 out; retorno: LSNR (CPU) / 0.0 (GPU) / -100 = erro
void   SetAttenuationLimit(float limit_db);       // efetivo em ambos: mix dry/wet pós-inferência, 10^(-|db|/20) (|db|>=100 = full wet, <0.01 = bypass)
void   SetDeepFilteringEnabled(bool enabled);     // GPU: efetivo; CPU: no-op
```
Impls: `SyclNoiseSuppressor` (GPU), `DeepFilterAdapter` (CPU/Rust).

### `IAudioPipeline` (audio_pipeline.h)
```cpp
struct AudioBuffer { std::vector<float> data; size_t sample_rate = 48000; };
bool Start(const std::string& input_device_id = "", const std::string& output_device_id = "");
void Stop();  bool IsRunning() const;
using ProcessCallback = std::function<void(const AudioBuffer& in, AudioBuffer& out)>;
void SetProcessCallback(ProcessCallback);
```
Impls: `MiniaudioPipeline` (device real; IDs = ÍNDICE em string, ex. "0"), `AsyncAudioPipeline` (worker; ignora device IDs; extras: `PushInput`, `PopOutput`, `SetMaxQueueDepth`, `FramesDropped`).

### `ITelemetryProvider` (telemetry_provider.h)
```cpp
struct TelemetryData { float gpu_utilization, processing_latency_ms, memory_footprint_mb; };
TelemetryData GetLatestData();  void Update();
```
Impls: `SyclTelemetryProvider` (Level Zero), `MockTelemetryProvider` (testes).

### `AudioStreamBuffer` (audio_stream_buffer.h)
`Push(data,size)` / `Available()` / `Pop(out,size)` (zero-fill se faltar) / `Reset()`. FIFO com compactação a cada ~48000 lidos.

## 2. C-API GPU (`sycl_accelerator.h`, extern "C")

```c
bool sycl_init();                                        // idempotente, mutex; acha GPU "Arc" ou default
void sycl_process(const float* in, float* out, size_t n); // n DEVE ser 480; senão retorna silencioso
void sycl_get_device_name(char* buf, size_t max);
void sycl_set_df_enabled(bool);
void sycl_reset();                                       // zera GRU states, históricos, overlap
```
Singleton global. Thread-safe por mutex único (serializa tudo).

## 3. C-API Rust (`include/deep_filter.h`, df.dll)

```c
DFState* df_create(const char* path, float atten_lim, const char* log_level);
size_t   df_get_frame_length(DFState*);          // 480
float    df_process_frame(DFState*, float* in, float* out);  // retorna LSNR
float    df_process_frame_raw(DFState*, float* in, float** gains, float** coefs);
void     df_set_atten_lim(DFState*, float db);
void     df_set_post_filter_beta(DFState*, float beta);
DynArray df_coef_size(const DFState*);  DynArray df_gain_size(const DFState*);
char*    df_next_log_msg(DFState*);  void df_free_log_msg(char*);
void     df_free(DFState*);
```
Cuidado: `df_create` pode dar panic (Rust) com arquivo inválido — checar existência antes (adapter já faz).

## 4. Contratos de dado

- Áudio: f32 intercalado NÃO — é mono puro, 48kHz fixo, frames de 480.
- Pesos GPU: `models/df3_weights/metadata.json` → `{ nome: { file, shape } }`, .bin f32 raw little-endian. 133 tensores + `erb_fb.bin` (481×32) + `mask_erb_inv_fb.bin` (32×481).
- Constantes do modelo: fft 960, hop 480, freq 481, erb 32, df_bins 96, df_order 5.
