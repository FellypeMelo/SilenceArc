# 📋 Plano de Investigação e Implementação Híbrida (SilenceArc)

Este documento detalha a estratégia para diagnosticar as falhas atuais do sistema e implementar a solução híbrida OpenVINO + SYCL, garantindo estabilidade e fidelidade de áudio.

---

## 🔍 Fase 1: Auditoria de Ambiente e Estabilidade
**Objetivo:** Garantir que o sistema possa ser compilado e testado sem erros de infraestrutura.

1.  **Correção de Dependências (DLLs):**
    - Identificar por que `df.dll` falha ao carregar nos testes (Erro `0xc0000135`).
    - Automatizar a cópia de artefatos do Rust (`DeepFilterNet/target/release`) para o diretório de binários do CMake.
2.  **Validação do Baseline (Rust):**
    - Rodar `test_noise_suppression` com o `DeepFilterAdapter` para confirmar a qualidade de áudio de referência.
3.  **Sanidade do SDK Intel:**
    - Verificar se `setupvars.bat` (OpenVINO) e `setvars.bat` (oneAPI) estão coexistindo sem conflitos no `PATH`.

---

## 🛠️ Fase 2: Investigação Profunda de Componentes
**Objetivo:** Entender a causa raiz das falhas na implementação SYCL nativa.

1.  **Profiling do DSP SYCL:**
    - Avaliar o impacto de performance do kernel DFT O(N^2) atual em `SyclDspEngine.cpp`.
    - Verificar se a latência de processamento está excedendo o *hop size* (real-time budget).
2.  **Análise de Sincronização USM:**
    - Auditar `NativeSyclEngine::process_frame` em busca de race conditions entre o DSP (SYCL) e a Inferência (oneDNN).
    - Verificar o uso de `sycl::queue::wait()` e barreiras de memória.
3.  **Diagnóstico de Fidelidade (Som Metálico):**
    - Testar a integridade do loopback STFT -> ISTFT (sem processamento neural) para isolar erros de janelamento ou fase.
    - Validar a interpolação de 32 bandas para 480 bins no kernel de masking.

---

## 🚀 Fase 3: Implementação da Arquitetura Híbrida
**Objetivo:** Migrar o grafo neural para OpenVINO mantendo o DSP otimizado em SYCL.

1.  **Integração do OpenVinoAdapter:**
    - Adaptar o `OpenVinoAdapter` para carregar os modelos IR do DeepFilterNet3.
    - Implementar a conversão de buffers USM (SYCL) para `ov::Tensor` sem cópias desnecessárias (se possível via host pointers).
2.  **Otimização do FFT:**
    - Substituir o DFT direto por uma implementação de FFT real (Radix-2) em SYCL ou via `oneMKL` (corrigindo o link no Windows).
3.  **Caminho Crítico de Baixa Latência:**
    - `STFT (SYCL)` -> `OpenVINO Inference (Backbone)` -> `Deep Filtering Kernel (SYCL)` -> `ISTFT (SYCL)`.

---

## 📈 Critérios de Sucesso
- [ ] **Estabilidade:** Zero ocorrências de `UR_RESULT_ERROR_DEVICE_LOST` em 1 hora de processamento.
- [ ] **Fidelidade:** Diferença de energia (MSE) < 1e-5 em relação ao adaptador Rust.
- [ ] **Performance:** Tempo total de processamento por frame < 8ms na Intel Arc B580.

---
**Data de Criação:** 13 de Março de 2026
**Responsável:** Distinguished Engineer (Gemini CLI)
