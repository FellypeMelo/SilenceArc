# SilenceArc: Engine de Inferência SYCL/oneDNN

O coração do SilenceArc é seu engine de inferência nativo, que dispensa runtimes de alto nível para alcançar o máximo de desempenho no hardware Intel Arc.

## Mapeamento de Pesos e Topologia
O DeepFilterNet3 é composto por 133 tensores de pesos distintos. O `OneDNNInferenceEngine` mapeia esses tensores exportados do PyTorch para primitivas nativas do oneDNN.

### Estágios de Camadas:
1.  **Encoder:** 34 primitivas, incluindo convoluções depthwise-separable e uma camada GRU para extração de embeddings.
2.  **ERB Decoder:** Reconstrói a máscara ERB (Equivalent Rectangular Bandwidth) usando convoluções transpostas e skip connections.
3.  **DF Decoder:** Calcula os coeficientes de Deep Filtering (DF) para remoção de ruído de granularidade fina.

## Layouts de Memória e Permutações
Um desafio crítico na inferência em GPU é a discrepância entre layouts de memória sequenciais e espaciais.

-   **Padrão oneDNN:** Prefere **NCHW** [Batch, Channels, Time, Freq] para convoluções espaciais.
-   **Padrão GRU:** Prefere **TNC** [Time, Batch, Channels] para processamento sequencial.

Como a primitiva `reorder` de GPU do oneDNN tem limitações com strides complexos, o SilenceArc implementa **kernels SYCL customizados** para essas permutações.

### Kernel de Reordenamento Customizado (TNC -> NCHW):
O kernel mapeia a saída sequencial da GRU de volta para as dimensões espaciais usando a fórmula:
`out[n*(C*T) + c*T + t] = in[t*(N*C) + n*C + c]`

## Unified Shared Memory (USM)
O SilenceArc usa **Device USM** para todos os buffers internos. Isso permite:
-   **Zero-Copy:** Os dados são processados in-place na GPU, sem staging intermediário no host.
-   **Acesso Direto:** Kernels SYCL e primitivas oneDNN compartilham os mesmos ponteiros de memória, reduzindo a complexidade de gerenciamento.

## Fluxo de Inferência
1.  **Análise STFT:** O áudio de entrada é janelado e convertido para o domínio da frequência usando **oneMKL DFT**.
2.  **Extração de Features:** O espectro de potência e as features ERB são calculados via kernels SYCL.
3.  **Execução do Engine:** O `OneDNNInferenceEngine` executa a cadeia sequencial de primitivas (Encoder -> Decoders).
4.  **Aplicação dos Coeficientes:** Os coeficientes DF são aplicados aos bins complexos de frequência.
5.  **Síntese ISTFT:** O sinal filtrado é convertido de volta ao domínio do tempo usando FFT inversa e síntese overlap-add.

## Sincronização de Hardware
O engine usa **filas SYCL in-order** para garantir que as primitivas executem na ordem topológica correta, sem o overhead de rastreamento manual de eventos.
