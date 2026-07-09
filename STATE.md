# STATE — SilenceArc

Data: 2026-07-09 · branch `fix/audit-sweep-2026-07` (audit fix sweep)

## Estado
Funcional. Supressão de ruído real-time via engine NATIVO SYCL/oneDNN (NÃO OpenVINO).
DeepFilterNet3 reimplementado em C++/oneDNN (133 tensores) rodando na Arc GPU.
Fallback CPU = Rust `df.dll` (tract/ONNX) via `DeepFilterAdapter`.
Pipeline: mic real -> processa -> saída real (duplex miniaudio/WASAPI). NÃO cria mic virtual.
Bench (Arc B580, 2026-07-09): per-frame avg 2.39ms, p99 4.33ms (< budget 10ms). Build -Wall -Wextra limpo.
Testes: 16/16 verdes (14 baseline + AudioMetricsTest + AttenuationLimitTest), rodados na Arc. Ver Cobertura.

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
4. Decidir escopo de dereverb + AEC (features d,e); pesquisar módulo.  [d,e]
5. Registrar self-hosted runner p/ o workflow CI (`.github/workflows/ci.yml`) — hoje só roda local.

## Próximas 3 tarefas
1. PoC mic virtual (WASAPI loopback / driver) → escrever saída limpa num device que outros apps enxergam.
2. Timestamp end-to-end (entrada callback → saída callback) + linha no BENCHMARKS.md.
3. Registrar runner self-hosted (oneAPI+Arc) e validar o workflow CI rodando build+ctest.

## Cobertura (audit sweep)
- Fix medidores reais → AudioMetricsTest (RMS/dB, guarda contra o fake 0.5).
- Fix SetAttenuationLimit GPU → AttenuationLimitTest (mapa dB→mix + caso GPU bypass=identidade).
- Fix FramesDropped na UI → AudioPipelineTest.FramesDroppedIsExposedForTheUi.
- Dead code/-Wall/gitignore/[DEBUG] → cobertos pelo build verde -Wall -Wextra + 16/16 ctest.

## Bloqueios
- Build exige Intel oneAPI (icx) + oneDNN + oneMKL + Level Zero SDK + Arc GPU.
- `setvars.bat` desta máquina falha a orquestração (VS/vswhere) em modo não-interativo; icx builda por path absoluto. P/ RODAR testes, prepend ao PATH: `...\oneAPI\2026.0\bin` + `...\oneAPI\mkl\2026.0\bin` (sycl9/dnnl/mkl DLLs).
- CI sem Arc: testes GPU/paridade/latência dão SKIP → gate GPU só roda local (runner self-hosted pendente).
