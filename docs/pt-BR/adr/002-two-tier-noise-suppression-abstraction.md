# ADR 002: Abstração de Supressão de Ruído em Duas Camadas

## Status
Aceito

## Contexto
O SilenceArc embarca duas stacks de inferência independentes:

- um engine **SYCL/oneDNN** nativo que executa o DeepFilterNet3 em GPUs Intel Arc, e
- um caminho de CPU em **Rust `libDF`** (`tract`/ONNX) usado como fallback quando nenhum dispositivo SYCL está presente.

No início do código-base, essas duas coisas estavam entrelaçadas com a decomposição interna do
engine de GPU. Duas abstrações chamadas `GPUAccelerator` e `NeuralNetworkModel` viviam sob
um diretório `domain/` em um namespace `sa::` separado, distinto do namespace
`silence_arc::domain`/`silence_arc::infrastructure` usado pelo resto da aplicação. Isso
criava três problemas:

1. **Duas coisas chamadas "domain" que não são pares.** `GPUAccelerator`/`NeuralNetworkModel`
   não são uma costura de seleção de backend — são a divisão *interna* DSP/NN de um único backend.
   Colocá-las em `domain/` sugeria que fossem um contrato de nível de aplicação como
   `INoiseSuppressor`, o que não são.
2. **Uma terceira raiz de namespace (`sa::`)** existia apenas para esses dois tipos, então quem lesse
   o código precisava rastrear `sa::` vs. `silence_arc::` sem nenhuma razão semântica para a separação.
3. Isso obscurecia onde de fato está a fronteira de seleção de backend.

## Decisão
Adotar uma abstração explícita em **duas camadas** e colapsar o namespace `sa::` em
`silence_arc::`.

**Camada 1 — a costura de seleção de backend (domínio):**
`silence_arc::domain::INoiseSuppressor` é a *única* abstração sobre "qual engine de
supressão de ruído está em execução". Suas duas implementações são
`silence_arc::infrastructure::SyclNoiseSuppressor` (GPU) e
`silence_arc::infrastructure::DeepFilterAdapter` (CPU/Rust). O `main.cpp` escolhe uma em
tempo de execução, com base no sucesso de `sycl_init()`.

**Camada 2 — a Bridge interna do engine de GPU (privada à infraestrutura):**
`GPUAccelerator` (DSP: STFT, features, filtragem, ISTFT) e `NeuralNetworkModel`
(camadas do DeepFilterNet3) são uma **divisão Bridge/SRP privada a `SyclNoiseSuppressor`**. Elas
*não* são uma segunda costura de backend. Agora vivem em
`include/silence_arc/infrastructure/`, sob `silence_arc::infrastructure`, ao lado de suas
únicas implementações concretas, `SYCLAccelerator` e `OneDNNInferenceEngine`.

O namespace `sa::` é removido por completo; o test harness interno migra de
`sa::test` para `silence_arc::test`.

## Consequências
- **Camadas mais claras.** `domain/` agora contém apenas contratos tecnologicamente agnósticos
  (`INoiseSuppressor`, `IAudioPipeline`, `ITelemetryProvider`) e tipos de valor. Nenhum header do
  SYCL ou do oneDNN é alcançável a partir da camada de domínio.
- **Uma única raiz de namespace.** Tudo é `silence_arc::{domain,infrastructure,test}`.
- **A Bridge permanece substituível, porém privada.** Separar DSP de NN por trás de
  `GPUAccelerator`/`NeuralNetworkModel` ainda permite que cada metade evolua de forma independente
  (por exemplo, um futuro executor de NN sem oneDNN) sem vazar para a costura de domínio.
- **Movimentação/renomeação pura**, verificada por um rebuild completo e pela suíte completa do
  CTest (14/14) permanecendo verde, incluindo o teste de paridade entre backends que exercita as
  duas implementações de `INoiseSuppressor` contra os mesmos fixtures.

## Relacionado
- ADR 001 — Seleção do DeepFilterNet3 como modelo principal.
