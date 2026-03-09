@echo off
cd /d "%~dp0build"
if exist silence_arc.exe (
    echo =======================================
    echo         Iniciando SilenceArc
    echo =======================================
    silence_arc.exe
) else (
    echo =======================================
    echo [ERRO] Executavel nao encontrado!
    echo =======================================
    echo Por favor, certifique-se de compilar o projeto 
    echo antes de tentar executa-lo.
    echo.
    pause
)
