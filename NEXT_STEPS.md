# 🚀 SilenceArc: Próximos Passos e Otimizações

Este documento lista as tarefas pendentes e oportunidades de melhoria identificadas após a estabilização da arquitetura **DirectML**.

## ⚡ Performance & DSP
- [ ] **FFT Real via Intel MKL/IPP:** Substituir a base de DFT atual por uma implementação de FFT real (RFFT) usando as bibliotecas de alto desempenho da Intel. Isso reduzirá drasticamente o uso de CPU no processamento STFT/ISTFT.
- [ ] **Pipeline Zero-Copy:** Mover o `FeatureExtractor` (Filterbank ERB e Normalização) para a GPU via Compute Shaders (HLSL) ou operadores ONNX customizados, eliminando a transferência de buffers de features entre CPU e GPU.
- [ ] **VAD (Voice Activity Detection):** Integrar um detector de atividade vocal para desativar a inferência neural durante silêncio, economizando energia e ciclos de GPU.

## 📊 Qualidade do Áudio
- [ ] **Métricas Automatizadas:** Implementar testes automatizados usando métricas PESQ (Perceptual Evaluation of Speech Quality) ou STOI para quantificar a melhoria da voz em cada alteração de código.
- [ ] **Dynamic Post-Filter:** Refinar o beta do post-filtro de forma dinâmica baseado no SNR (Signal-to-Noise Ratio) estimado pelo modelo.

## 🖥️ UI/UX
- [ ] **Espectrograma em Tempo Real:** Adicionar uma visualização de espectrograma na interface ImGui para permitir que o usuário veja a supressão de ruído acontecendo em diferentes frequências.
- [ ] **Profile Manager:** Permitir salvar configurações de atenuação e limites para diferentes microfones ou ambientes.

## 🛠️ Infraestrutura
- [ ] **Build Multialvo:** Configurar o CMake para suportar builds otimizados para diferentes gerações de Intel Arc (Alchemist vs Battlemage).
- [ ] **Logging Centralizado:** Melhorar o sistema de telemetria para gravar logs de performance em formato JSON para análise posterior.

---
**Status Atual:** Estável em DirectML. Próxima grande prioridade: **Otimização do núcleo DSP (MKL).**
