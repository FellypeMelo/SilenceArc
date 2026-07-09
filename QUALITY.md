# QUALITY — Gates SilenceArc

Regra: nenhum merge sem TODOS os gates abaixo. Denso. Sem exceção silenciosa.

## 1. Build verde
- Compila com `icx` + oneAPI (oneDNN, oneMKL, Level Zero) sem erro.
- Alvos: `silence_arc` + todos os `test_*` + `bench_pipeline_latency`.

## 2. Testes verdes
- `ctest` no `build/`: suíte passa. Baseline = 14 testes.
- Testes que dependem de Arc (BackendParity, NNLayers, KernelCorrectness, GPUBridge, PipelineLatencyBench) devem PASSAR na máquina com Arc; em host sem Arc devem dar SKIP limpo (return 0), nunca falso-verde mascarando regressão.
- Mudou lógica de áudio/NN → rodar `test_e2e_samples` + `test_backend_parity`.

## 3. Latência sem regressão
- `bench_pipeline_latency`: p99 per-frame < 10ms (budget hard, já falha sozinho se estourar).
- Comparar avg/p50/p99 contra última linha de `docs/BENCHMARKS.md`. Regressão > 15% no p99 = bloqueio + justificar.
- Meta de PRODUTO: latência end-to-end (mic→speaker) < 20ms. Ainda NÃO medida — quando houver medição, vira gate.

## 4. Zero warnings novos
- Não introduzir warning novo. (Nota: build hoje NÃO liga `-Wall`; ao ligar, zerar antes de exigir o gate.)

## 5. Docs atualizados
- Mudou arquitetura/pipeline → atualizar `docs/ARCHITECTURE.md` / `docs/ENGINE.md`.
- Mudou interface pública (headers `domain/`, C-API `sycl_*`, `INoiseSuppressor`) → atualizar `docs/API.md`.
- Decisão nova/revertida → ADR em `docs/adr/` + linha em `docs/DECISIONS.md`.
- Nova medição → linha em `docs/BENCHMARKS.md` (gpu, modelo, latência, CPU%, data, commit).

## 6. STATE.md atualizado
- Toda tarefa fechada: reescrever Estado, Backlog, Próximas 3, Bloqueios. Manter ≤50 linhas.
