# SilenceArc: Guia de Configuração e Build

Siga estes passos para configurar seu ambiente e compilar o SilenceArc para GPUs Intel Arc.

## Pré-requisitos

### 1. Intel oneAPI Base Toolkit
Você precisa ter o **Intel oneAPI Base Toolkit** instalado (versão 2024.0 ou mais recente).
-   **Compilador:** `icx` (Intel LLVM C++ Compiler) é necessário para o suporte a `-fsycl`.
-   **Bibliotecas:** oneDNN e oneMKL devem estar incluídos na sua instalação.

### 2. CMake
É necessária a versão 3.20 ou mais recente.

### 3. Rust (para o core do DeepFilterNet)
Se você planeja modificar a lógica do modelo, precisará do toolchain do Rust instalado. O projeto usa um `df.dll` pré-compilado para builds padrão.

## Configuração do Ambiente

Antes de compilar ou executar a aplicação, você precisa inicializar as variáveis de ambiente do oneAPI. Este projeto fornece um script auxiliar:

```powershell
# Em uma janela do PowerShell ou CMD:
.\setup_intel.bat
```

Este script invoca o `setvars.bat` da Intel e configura o ambiente para o compilador `icx` e as bibliotecas necessárias.

## Compilando o Projeto

O SilenceArc usa CMake para gerenciamento do projeto. Recomendamos o gerador **Ninja** para builds mais rápidos.

```bash
# 1. Crie um diretório de build
mkdir build
cd build

# 2. Configure com o compilador Intel LLVM
cmake -G "Ninja" -DCMAKE_CXX_COMPILER=icx -DCMAKE_C_COMPILER=icx ..

# 3. Compile a aplicação
cmake --build . --config Release
```

## Dependências em Tempo de Execução
Garanta que o `df.dll` (do diretório de destino do DeepFilterNet) esteja na mesma pasta que `silence_arc.exe`, ou disponível no PATH do sistema.

## Verificação
Execute o comando a seguir para verificar se sua GPU é detectada corretamente e se os kernels estão funcionais:
```bash
.\build\test_nn_layers.exe
```
Você deve ver uma mensagem: `[INFO] SYCL Initialized on: Intel(R) Arc(TM) ...`
