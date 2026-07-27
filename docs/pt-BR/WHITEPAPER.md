# Whitepaper do SilenceArc: Inteligência de Áudio Nativa Acelerada por GPU

> **Nota sobre este documento.** Este whitepaper foi redigido no início do projeto como um documento de posicionamento — ele descreve intenção e metas de design, não um conjunto de resultados medidos. Onde um número abaixo não é sustentado por um benchmark commitado neste repositório, ele é explicitamente marcado como meta, e não como medição. Veja [ARCHITECTURE.md](./ARCHITECTURE.md) e a seção "Resultados verificados" do README para o que de fato é checado no repositório.

## Resumo
Este documento apresenta o **SilenceArc**, uma aplicação de supressão de ruído de áudio em tempo real construída especificamente para arquiteturas de GPU Intel Arc. Ao usar primitivas nativas de SYCL e oneDNN em vez de um runtime de inferência genérico de alto nível, o SilenceArc troca uma superfície de implementação maior por controle direto sobre layout de memória, escalonamento de kernels e inferência — controle cujo racional é explicado no documento companheiro [PHILOSOPHY.md](./PHILOSOPHY.md).

## 1. Introdução
A demanda por supressão de ruído de baixa latência e alta qualidade cresceu com a ascensão do streaming, do trabalho remoto e da produção musical digital. Enquanto soluções existentes frequentemente dependem de processamento pesado em CPU ou de runtimes de IA do tipo "caixa-preta", o SilenceArc utiliza as unidades especializadas **Xe Matrix eXtensions (XMX)** das GPUs Intel Arc para oferecer uma experiência nativa e de alta fidelidade.

## 2. Inovação Técnica: Inferência SYCL Nativa
A inovação central do SilenceArc é seu engine de inferência nativo em C++. A maioria das aplicações de IA usa runtimes como OpenVINO ou ONNX Runtime para gerenciar a abstração de hardware. O SilenceArc, em vez disso, se comunica diretamente com o hardware por meio de:
-   **SYCL puro:** kernels customizados gerenciam operações de DSP específicas de áudio.
-   **Primitivas oneDNN:** operações de rede neural de baixo nível são mapeadas diretamente para as unidades de execução da Arc.
-   **Gerenciamento via USM:** Unified Shared Memory elimina o gargalo das transferências de dados entre host e device.

## 3. O Pipeline Neural
O SilenceArc integra o modelo perceptual **DeepFilterNet3**, estado da arte. O engine trata:
-   **133 tensores de pesos:** exportados do checkpoint PyTorch e mapeados para primitivas oneDNN (contagem verificada em `models/df3_weights/`).
-   **Convoluções separáveis:** otimizadas para a largura de banda de memória da arquitetura Xe.
-   **Processamento recorrente:** implementação de GRU usando as sequências otimizadas da oneAPI.

## 4. Metas de Design e Resultados Verificados
Contornar as camadas de abstração de alto nível tem a intenção de proporcionar ao SilenceArc:
-   **Baixa latência (meta, ainda não uma medição commitada):** o objetivo é processamento por frame na faixa de milissegundos de um único dígito, compatível com tempo real. O `tests/bench_pipeline_latency.cpp` calcula latência real p50/p99 contra um orçamento de 10ms de tempo real, após um protocolo de aquecimento de GPU — mas sua saída não foi commitada neste repositório, então não há ainda um número de latência de primeira mão para citar.
-   **Menor overhead de CPU:** descarregar a inferência para a GPU tem a intenção de deixar mais margem de CPU para cargas de trabalho concorrentes (jogos, streaming, codificação de vídeo). Isso não foi medido nem quantificado neste repositório.
-   **Sem dependência de OpenVINO/ONNX Runtime no caminho de inferência em GPU:** este ponto é verificável no código-fonte — veja [ARCHITECTURE.md](./ARCHITECTURE.md) e [ENGINE.md](./ENGINE.md).

O que *é* verificado neste repositório: exatamente 133 tensores de pesos do DeepFilterNet3 em `models/df3_weights/`, e 14 testes registrados no CTest (veja a seção "Resultados verificados" do README para ambos).

## 5. Conclusão e Trabalhos Futuros
O SilenceArc demonstra o grande potencial do ecossistema Intel oneAPI para aplicações criativas em tempo real. Versões futuras devem expandir essa base nativa para incluir aprimoramento de voz inteligente, correção de pitch em tempo real e suporte a configurações multi-GPU.

---
**Autor:** AI-XP Governance Framework / Fellype Melo
**Data:** 9 de março de 2026
**Licença:** Apache License 2.0
