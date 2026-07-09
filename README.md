# SilenceArc

Supressão de ruído em tempo real p/ **GPUs Intel Arc**. Equivalente ao RTX Voice, mas nativo Intel. Parte da **Arc Suite**.

**O quê:** captura o mic, roda **DeepFilterNet3** na Arc GPU, devolve voz limpa. Latência de processamento ~4-5ms por frame de 10ms.

**Por quê assim:** sem OpenVINO, sem ONNX Runtime no caminho GPU. Engine própria em **SYCL + oneDNN + oneMKL** — 133 tensores do DFN3 mapeados 1:1 em primitivas nativas, USM zero-copy, fila in-order com um sync por frame. Controle total do hot path, DSP custom entrelaçado com a rede. Racional completo: [docs/PHILOSOPHY.md](./docs/PHILOSOPHY.md) e [docs/DECISIONS.md](./docs/DECISIONS.md).

## Estado atual (honesto)

- Funciona: mic real → supressão na GPU → speaker real (duplex WASAPI shared, mono 48kHz).
- Fallback CPU automático: Rust `df.dll` (tract/ONNX) quando não há GPU SYCL.
- Bench Arc B580: avg 3.86ms, p99 4.85ms por frame (budget 10ms). Ver [docs/BENCHMARKS.md](./docs/BENCHMARKS.md).
- **Falta** (backlog em [STATE.md](./STATE.md)): device de áudio VIRTUAL (Discord/OBS ainda não enxergam a saída limpa), medição end-to-end, dereverb, echo cancel.

## Matriz de GPU

| GPU | Status | Nota |
|---|---|---|
| Arc B580 (Battlemage) | ✅ testado | bench oficial; p99 4.85ms |
| Arc B-Series (outras) | ✔️ esperado | mesma arquitetura, sem bench registrado |
| Arc A-Series (Alchemist) | ✔️ esperado | suportado por SYCL/oneDNN; sem bench registrado |
| iGPU Intel Xe (Meteor Lake+) | ❓ não testado | detector procura nome "Arc"; iGPU cai no default selector |
| Sem GPU Intel | ⚠️ fallback CPU | df.dll (Rust/tract), qualidade igual, latência maior |
| NVIDIA / AMD | ❌ | fora de escopo — use RTX Voice / AMD NR |

## Stack

C++20 (icx) · SYCL/oneAPI · oneDNN · oneMKL DFT · Level Zero (telemetria) · miniaudio (WASAPI) · Dear ImGui (DX11) · Rust df.dll (fallback) · CMake + Ninja.

## Quickstart

Pré-requisitos: GPU Arc + driver, **Intel oneAPI Base Toolkit 2024+** (icx, oneDNN, oneMKL), **Level Zero SDK**, CMake ≥3.20, Ninja. Rust só se for recompilar `df.dll`.

```powershell
git clone https://github.com/FellypeMelo/SilenceArc.git
cd SilenceArc
.\setup_intel.bat          # ambiente oneAPI

mkdir build; cd build
cmake -G "Ninja" -DCMAKE_CXX_COMPILER=icx -DCMAKE_C_COMPILER=icx ..
cmake --build . --config Release

cd ..
.\run.bat                  # abre a GUI; escolha mic e saída
```

Verificar instalação:
```powershell
.\build\test_nn_layers.exe          # carrega 133 tensores + 1 inferência na GPU
ctest --test-dir build              # suíte completa (14 testes)
.\build\bench_pipeline_latency.exe  # latência per-frame (p50/p90/p99)
```

## Uso

- **Input/Output:** mic e speaker reais. Modo shared WASAPI.
- **Noise Suppression:** toggle liga/desliga; **Attenuation Limit** 20dB soa natural, 100dB silêncio absoluto (efetivo no backend CPU; GPU: fixo por enquanto).
- **Telemetria:** latência de processamento e utilização de GPU (Level Zero) em tempo real.
- **Tray:** minimiza pra bandeja e continua rodando.

## Estrutura

```
include/silence_arc/domain/   contratos (INoiseSuppressor, IAudioPipeline...)
include/silence_arc/infrastructure/ + src/infrastructure/   implementações
models/df3_weights/           133 tensores .bin + metadata.json + filterbanks
DeepFilterNet/                fork vendorado do core Rust (fallback CPU)
scripts/export_df3_weights.py exporta checkpoint PyTorch → .bin
tests/                        unit + e2e + paridade GPUvsCPU + bench
docs/                         ARCHITECTURE, ENGINE, API, DECISIONS, BENCHMARKS, ADRs
```

Docs de trabalho: [STATE.md](./STATE.md) (estado+backlog) · [QUALITY.md](./QUALITY.md) (gates) · [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) (pipeline) · [WHITE_PAPER.md](./WHITE_PAPER.md).

## Licença

Apache 2.0 — ver [LICENSE](./LICENSE). Núcleo DeepFilterNet: MIT/Apache-2.0 (ver `DeepFilterNet/`).
