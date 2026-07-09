# STATE — SilenceArc

Data: 2026-07-09 · HEAD: `2129d83` (2026-07-07)

## Estado
Funcional. Supressão de ruído real-time via engine NATIVO SYCL/oneDNN (NÃO OpenVINO).
DeepFilterNet3 reimplementado em C++/oneDNN (133 tensores) rodando na Arc GPU.
Fallback CPU = Rust `df.dll` (tract/ONNX) via `DeepFilterAdapter`.
Pipeline: mic real -> processa -> saída real (duplex miniaudio/WASAPI). NÃO cria mic virtual.
Bench mais recente (Arc B580): per-frame avg 3.86ms, p99 4.85ms (< budget 10ms). Testes 13/14 passam; latency bench passou no último run.

## Gaps vs metas
- (a) DFN3 "em OpenVINO": FEITO como intenção (DFN3 na Arc), mas por design SEM OpenVINO (ADR-003). Literal OpenVINO = pendente P8 (DECISIONS.md).
- (b) Virtual audio device (VB-Cable/WASAPI sink): AUSENTE. So duplex real->real. Bloqueia uso por Discord/OBS.
- (c) Latência <20ms medida: per-frame ~5ms medido; e2e (mic->speaker c/ STFT+fila+buffers) NÃO medido.
- (d) Dereverberation: AUSENTE (só o implícito do DFN3).
- (e) Echo cancel (AEC): AUSENTE.

## Backlog (do gap analysis)
1. Mic/saída VIRTUAL (loopback sink) — sem isso não é RTX-Voice utilizável.  [b]
2. Instrumentar+medir latência END-TO-END real, provar <20ms.  [c]
3. Validar paridade numérica GPU vs CPU (test_backend_parity com threshold), qualidade do mask.
4. Limpar código morto/debug + ligar `-Wall`/`-Wextra` e zerar warnings.
5. Decidir escopo de dereverb + AEC (features d,e); pesquisar módulo.  [d,e]

## Próximas 3 tarefas
1. PoC mic virtual (WASAPI loopback / driver) → escrever saída limpa num device que outros apps enxergam.
2. Timestamp end-to-end (entrada callback → saída callback) + linha no BENCHMARKS.md.
3. Sweep de limpeza: remover HexToDeviceId, df_p_flat, stubs test_*_mapping, `[DEBUG]` couts.

## Bloqueios
- Build exige Intel oneAPI (icx) + oneDNN + oneMKL + Level Zero SDK + Arc GPU.
- CI sem Arc: testes GPU/paridade/latência dão SKIP → sem gate de qualidade automatizado em CI.
- GPU: `SetAttenuationLimit` é no-op (paridade de feature com CPU pendente).
