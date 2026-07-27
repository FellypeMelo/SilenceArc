# ADR 001: Seleção do Modelo Principal de Supressão de Ruído

## Status
Aceito

## Contexto
O Silence Arc requer um modelo de supressão de ruído em tempo real que ofereça aprimoramento de voz de alta qualidade e remoção eficaz de ruído de fundo. O modelo precisa ser capaz de ser acelerado em GPUs Intel Arc usando SYCL/oneAPI, sem depender do framework OpenVINO. Comparamos dois candidatos principais: **RNNoise** e **DeepFilterNet3**.

## Decisão
Selecionamos o **DeepFilterNet3** como o modelo principal de supressão de ruído do Silence Arc.

## Racional
1.  **Qualidade de áudio reportada na literatura upstream:** a pesquisa do DeepFilterNet3 (vendorizada em `DeepFilterNet/`, com o paper original referenciado em `DeepFilterNet/README.md`) reporta melhorias de Mean Opinion Score (MOS) em relação a baselines da classe do RNNoise, com melhor preservação da naturalidade da fala e menos artefatos relacionados à FFT. Esses são os números publicados pelos autores do DeepFilterNet3, não uma avaliação de MOS que o SilenceArc tenha executado — este projeto não reproduziu nem mediu de forma independente uma comparação de MOS.
2.  **Tratamento de ruídos complexos:** Ele se destaca na remoção de ruído não-estacionário (multidões, cliques, ambientes urbanos), o que é crítico para streamers e gamers.
3.  **Alinhamento arquitetural:** A operação de "Deep Filtering" (MAD complexo ao longo de taps temporais) é altamente paralelizável e mapeia diretamente para kernels SYCL otimizados.
4.  **Eficiência de hardware — uma meta de projeto, não um resultado medido:** simulações informais de benchmark feitas durante a seleção do modelo sugeriram que a operação central de DSP poderia ser concluída na faixa de submilissegundos em uma Intel Arc B580, deixando margem para a rede neural de dois estágios (ERB e DF). Nenhum script de benchmark, dataset ou saída que sustente um número específico está commitado neste repositório — portanto, essa figura motivou a escolha, em vez de confirmá-la posteriormente. O `tests/bench_pipeline_latency.cpp` é o mecanismo que poderia produzir uma figura real e commitada de latência p50/p99 para este pipeline contra um orçamento de 10ms; até o momento, sua saída não foi commitada.
5.  **Flexibilidade:** A arquitetura do DeepFilterNet3 permite o "Deep Signal Control" solicitado em nosso guia de produto, possibilitando manipulação de granularidade fina dos coeficientes no domínio da frequência.

## Consequências
- **Esforço de implementação:** Portar o modelo do seu ambiente original em Rust/PyTorch para uma implementação em C++/SYCL/oneDNN exigirá mais esforço inicial do que o RNNoise.
- **Dependência:** Utilizaremos o **oneDNN** para aceleração das camadas de rede neural e o **oneMKL** para transformadas no domínio da frequência (FFT/IFFT).
- **Fallback:** O RNNoise permanece como um fallback válido para cenários de baixíssimo consumo de energia, caso testes futuros em GPUs integradas mostrem restrições de desempenho.
