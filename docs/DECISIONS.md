# DECISIONS — SilenceArc

Índice de decisões. ADRs formais em `docs/adr/`. Retroativos marcados. Frases curtas.

## Aceitas

### ADR-001 — Modelo: DeepFilterNet3 (não RNNoise)
Arquivo: `adr/001-model-selection.md`. Status: aceita.
MOS ~4.0 vs ~3.2. Deep filtering paraleliza bem em SYCL. Custo: port PyTorch→C++/oneDNN braçal. RNNoise fica como fallback teórico p/ low-power.

### ADR-002 — Seam de backend em 2 tiers
Arquivo: `adr/002-two-tier-noise-suppression-abstraction.md`. Status: aceita.
Tier 1: `INoiseSuppressor` = ÚNICA seleção de engine (GPU vs CPU), decidida em runtime por `sycl_init()`.
Tier 2: `GPUAccelerator`+`NeuralNetworkModel` = bridge PRIVADO da engine GPU. Não é seam. Namespace `sa::` morto.

### ADR-003 (retroativo) — Runtime: SYCL/oneDNN puro, SEM OpenVINO
Status: aceita (fundacional). Ver `docs/PHILOSOPHY.md`, `conductor/tech-stack.md`.
OpenVINO = caixa-preta, overhead, menos controle p/ DSP custom entrelaçado com NN.
Escolha: oneDNN p/ primitivas NN, oneMKL DFT p/ STFT, kernels SYCL custom p/ permutes/filtro, USM zero-copy, fila in-order.
Consequência: build exige toolchain Intel completa (icx, oneAPI, Level Zero SDK); portabilidade p/ não-Intel = zero.

### ADR-004 (retroativo) — Captura: miniaudio duplex (WASAPI shared)
Status: aceita, INSUFICIENTE p/ meta de produto.
Um device duplex mono f32 48kHz; callback é shim fino; processamento em worker TIME_CRITICAL com filas bounded drop-oldest (depth 8).
`ma_share_mode_shared` no código (README fala "Exclusive Mode" — divergência).
Limite: sem endpoint virtual, apps terceiros não consomem o áudio limpo.

### ADR-005 (retroativo) — Fallback CPU: Rust df.dll vendorado
Status: aceita.
`DeepFilterNet/` vendorado no repo (commit 50279a9) p/ rastrear patches locais. `df.dll`/`df.dll.lib` COMMITADOS binários na raiz. C-API em `include/deep_filter.h`.

### ADR-006 (retroativo) — Pesos: export próprio .bin + metadata.json
Status: aceita.
`scripts/export_df3_weights.py` extrai checkpoint PyTorch → 133 `.bin` f32 + `metadata.json` (+ `erb_fb.bin`/`mask_erb_inv_fb.bin`). Engine carrega direto, sem parser ONNX no caminho GPU.

### ADR-007 (retroativo) — Normalização EMA no host, α=0.99
Status: aceita (fix de bug — α=0.1 degenerava features, mask constante ~0.16).
band_mean_norm/unit_norm rodam no host entre STFT e inferência. Custo: 2 syncs device→host por frame. Candidata a mover p/ device depois.

## Pendentes

| # | Decisão | Opções | Nota |
|---|---|---|---|
| P1 | Como criar device de áudio virtual | driver próprio (AVStream/APO) vs depender de VB-Cable instalado vs process loopback | bloqueia meta (b); driver assinado = custo alto |
| P2 | Medição de latência end-to-end | timestamps no callback vs loopback físico | meta (c) exige <20ms provado |
| P3 | Dereverberation | DFN3 já atenua um pouco; treinar/pós-filtro dedicado? | meta (d); definir escopo |
| P4 | Echo cancel (AEC) | speex/WebRTC AEC3 na frente do DFN3 vs nada | meta (e); AEC3 é C++ BSD |
| P5 | Mover normalização EMA p/ device | kernel SYCL + estado device | corta 2 syncs/frame do hot path |
| P6 | CI sem GPU Arc | runner self-hosted vs só validação CPU | hoje gates GPU só rodam local |
| P7 | Multicanal / sample rates ≠48k | resampler na entrada | hoje assume mono 48k fixo |
| P8 | Reavaliar "DFN3 em OpenVINO" (meta suite a) | manter mandato nativo (ADR-003) vs backend OpenVINO opcional p/ portabilidade | conflita com filosofia do projeto; decidir e registrar |
