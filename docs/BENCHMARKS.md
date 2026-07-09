# BENCHMARKS — SilenceArc

Como medir: `./build/bench_pipeline_latency.exe` (10k frames medidos, warmup ≥2000 frames E ≥3s p/ clocks estáveis). Mede `sycl_process()` per-frame (STFT→features→NN→filtro→ISTFT). Budget hard: p99 < 10ms (frame de 480 @ 48kHz). CPU%: anotar de Task Manager/`typeperf` durante o bench. Latência e2e (mic→speaker): AINDA SEM harness — ver DECISIONS P2.

Regra: toda mudança no hot path = nova linha ANTES e DEPOIS. Nunca sobrescrever linha antiga.

## Per-frame (sycl_process)

| Data | Commit | GPU | Driver | Modelo | avg ms | p50 | p90 | p99 | max | CPU% | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-07 | 2129d83 | Arc B580 | n/a | DFN3 nativo oneDNN | 3.863 | 4.232 | 4.462 | 4.850 | 6.202 | n/d | pós-refactor de syncs; fonte: build/Testing LastTest.log |
| 2026-07-09 | fix/audit-sweep-2026-07 | Arc B580 | n/a | DFN3 nativo oneDNN | 2.387 | 2.080 | 3.306 | 4.334 | 13.558 | n/d | pós audit sweep (atten-limit é pós-inferência, fora do sycl_process medido); sem regressão vs baseline. 10k frames |

## End-to-end (mic → speaker) — pendente harness

| Data | Commit | GPU | Buffer WASAPI | e2e ms (p50/p99) | Notas |
|---|---|---|---|---|---|
| — | — | — | — | — | sem medição ainda |

## Qualidade (referência)

| Data | Commit | Teste | Métrica | Valor | Notas |
|---|---|---|---|---|---|
| 2026-03-08 | (produto) | e2e samples | redução de ruído | ~19 dB | citado em conductor/product.md; revalidar com test_e2e_samples |
