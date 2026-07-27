# Roadmap

A portabilidade do SilenceArc para SYCL/oneDNN em GPU foi construída em três fases. Este documento acompanha o que está pronto, o que está verificado e o que ainda está em aberto. Ele substitui um antigo rastreador de tarefas na raiz do repositório (`sycl-integration.md`), mantendo o mesmo histórico de fases, corrigido onde havia se desviado do código-base real (veja a nota na Fase 1, Tarefa 5).

## Fase 1 — Ambiente de build SYCL e test harness (concluída)

- [x] **Tarefa 1:** Criar `tests/sycl_test_harness.h` — um harness de asserções header-only, sem dependências, para testes em nível de dispositivo de GPU que não se encaixam no modelo de processo do GoogleTest.
- [x] **Tarefa 2:** Migrar `tests/test_sycl_discovery.cpp` para o novo harness. Verificado: compila com `icx -fsycl`.
- [x] **Tarefa 3:** Corrigir o `CMakeLists.txt` para contornar um `find_package(IntelSYCL)` quebrado e linkar o SYCL manualmente. Verificado: a etapa de configure do `cmake` é concluída.
- [x] **Tarefa 4:** Definir a abstração `GPUAccelerator` e sua implementação SYCL. Verificado: a separação de Clean Architecture era válida na época (revisada depois — veja a nota abaixo).
- [x] **Tarefa 5:** Estabelecer a fronteira da bridge FFI de GPU. Verificado: as assinaturas `extern "C"` correspondem ao código que as chama.

  > **Correção em relação ao rastreador original.** Esta tarefa estava anteriormente registrada como "Desenhar a Bridge FFI em `DeepFilterNet/libDF/src/gpu_bridge.rs`". Esse arquivo nunca existiu em nenhum lugar sob `DeepFilterNet/libDF/src/` neste repositório — o caminho já estava errado desde o início. A verdadeira fronteira FFI de GPU é o bloco `extern "C"` (`sycl_init`, `sycl_process`, `sycl_get_device_name`, `sycl_set_df_enabled`, `sycl_reset`) em `src/infrastructure/sycl_accelerator.cpp`, exercitado por `tests/test_gpu_bridge.cpp` (`GPUBridgeTest` na suíte do CTest). A tarefa em si foi genuinamente concluída; apenas o caminho de arquivo registrado para ela estava desatualizado.

- [x] **Tarefa 6 (TDD RED):** `test_sycl_discovery` falha quando nenhuma GPU Arc está presente ou o ambiente está mal configurado.
- [x] **Tarefa 7 (TDD GREEN):** Ambiente e código corrigidos até que `test_sycl_discovery` passe em hardware real.

## Fase 2 — Portabilidade dos kernels centrais e integração com oneDNN (concluída)

- [x] **Tarefa 8:** Integrar o oneDNN (DNNL) ao sistema de build.
- [x] **Tarefa 9:** Portar o STFT (análise) para kernels SYCL.
- [x] **Tarefa 10:** Portar o ISTFT (síntese) para kernels SYCL.
- [x] **Tarefa 11:** Implementar o Deep Filtering (convolução no domínio da frequência) em SYCL.
- [x] **Tarefa 12 (TDD RED):** `test_kernel_correctness` escrito contra um sinal sintético, checado contra uma baseline numérica.
- [x] **Tarefa 13 (TDD GREEN):** Lógica dos kernels corrigida até que `test_kernel_correctness` passe (MSE < 1e-13 contra a baseline).

## Fase 3 — Portabilidade da rede neural para GPU (em aberto)

- [ ] **Tarefa 14:** Mapear as camadas restantes do DeepFilterNet3 (convoluções, GRU/linear) para primitivas oneDNN.
- [ ] **Tarefa 15:** Carregar os pesos a partir dos tensores exportados (`models/df3_weights/`, 133 arquivos, veja `scripts/export_df3_weights.py`) para buffers oneDNN.
- [ ] **Tarefa 16:** Portar o caminho de inferência do encoder para GPU.
- [ ] **Tarefa 17:** Portar os decoders ERB e DF para GPU.
- [ ] **Tarefa 18 (TDD RED):** Verificar a inferência completa em GPU contra a baseline de CPU em Rust/`tract` (`test_backend_parity`).
- [ ] **Tarefa 19 (TDD GREEN):** Otimizar o fluxo de dados e o batching depois que a corretude estiver estabelecida.

## Critérios de "concluído" das Fases 1–2

- [x] `test_sycl_discovery` roda sem conflitos com o GTest.
- [x] A abstração de backend de GPU está definida e isolada da camada de domínio.
- [x] O build é estável usando `icx -fsycl`.
- [x] Os kernels de STFT/ISTFT e Deep Filtering estão funcionais na GPU.

## Notas

- Toda fase seguiu uma disciplina TDD RED→GREEN: um teste que falha é comprovado primeiro, depois vem a implementação mínima para fazê-lo passar.
- A camada de Clean Architecture referenciada pela Tarefa 4 acima foi revisada desde então: `GPUAccelerator`/`NeuralNetworkModel` não vivem mais sob `domain/` em um namespace `sa::`. A [ADR-002](./adr/002-two-tier-noise-suppression-abstraction.md) os moveu para `include/silence_arc/infrastructure/` como uma Bridge privada a `SyclNoiseSuppressor`, colapsando `sa::` em `silence_arc::infrastructure`. Trate [ARCHITECTURE.md](./ARCHITECTURE.md) e a ADR-002 como a fonte de verdade atual sobre onde esses tipos vivem, não o texto da Fase 1 deste roadmap.
- Unified Shared Memory (USM) é usada para transferência zero-copy entre host e device.
- Buffers de scratch do caminho crítico são pré-alocados para evitar overhead de alocação host-device no caminho do callback de áudio — veja [ENGINE.md](./ENGINE.md).
