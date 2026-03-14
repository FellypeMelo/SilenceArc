@echo off
setlocal enabledelayedexpansion

echo [INFO] Configurando ambiente de execução SilenceArc (DIRECTML MODE)...

set "BUILD_DIR=%~dp0build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

:: 1. Copiar DLLs do ONNX Runtime DirectML
echo [INFO] Copiando DLLs do ONNX Runtime...
set "ORT_BIN=third_party\onnxruntime\runtimes\win-x64\native"
copy /Y "%ORT_BIN%\onnxruntime.dll" "%BUILD_DIR%\" >nul
copy /Y "%ORT_BIN%\DirectML.dll" "%BUILD_DIR%\" >nul

:: 2. Copiar DLL do Rust (DeepFilterNet)
echo [INFO] Copiando df.dll...
copy /Y "DeepFilterNet\target\release\df.dll" "%BUILD_DIR%\" >nul

echo [SUCCESS] Ambiente configurado em %BUILD_DIR%
echo [INFO] Para rodar testes: run_tests.bat
echo [INFO] Para rodar app: build\silence_arc.exe
