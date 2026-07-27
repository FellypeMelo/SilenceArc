# SilenceArc: Guia do Usuário

O SilenceArc oferece uma interface minimalista e de alto desempenho para supressão de ruído em tempo real.

## Primeiros Passos

1.  **Inicie a aplicação:** Execute `run.bat` ou o binário `build/silence_arc.exe`.
2.  **Verifique a aceleração por GPU:** Confira a janela de console ou a seção de Telemetria na GUI para garantir que sua GPU Intel Arc está ativa.

## Controles da Interface

### 1. Seleção de Dispositivo de Áudio
-   **Dispositivo de entrada:** Selecione seu microfone no menu suspenso. O SilenceArc suporta dispositivos WASAPI e ASIO.
-   **Dispositivo de saída:** Selecione suas caixas de som ou fones de monitoramento.
-   **Observação:** O SilenceArc usa o Modo Exclusivo quando disponível, para garantir a menor latência possível.

### 2. Configurações de Supressão
-   **Habilitar supressão:** Ativa ou desativa o engine de supressão de ruído.
-   **Limite de Atenuação (dB):** Ajusta a agressividade da remoção de ruído.
    -   **20dB:** Som mais natural, preserva nuances vocais.
    -   **100dB:** Silêncio máximo, ideal para ambientes muito ruidosos.

### 3. Monitoramento de Sinal
-   **Nível de entrada:** Leitura visual em tempo real do sinal bruto do microfone.
-   **Nível de saída:** O sinal após a aplicação da supressão de ruído.
-   **Piso de ruído:** Estimativa do nível atual de ruído de fundo sendo suprimido.

### 4. Telemetria de Hardware
O SilenceArc fornece insights em tempo real sobre o desempenho do seu hardware:
-   **Utilização da GPU:** Quanto da capacidade de computação da sua GPU Intel Arc está sendo usada pelo engine de inferência.
-   **Latência de processamento:** O tempo de ida e volta (em milissegundos) para um frame ser processado na GPU.
-   **Consumo de memória:** O uso de VRAM pelas primitivas oneDNN e pelos buffers de STFT.

## Integração com a Bandeja do Sistema
O SilenceArc pode ser minimizado para a bandeja do sistema, onde continuará processando seu áudio em segundo plano com impacto mínimo na CPU.
