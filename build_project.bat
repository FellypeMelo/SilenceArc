@echo off
echo [INFO] Iniciando build limpo SilenceArc com Intel oneAPI...

:: 1. Carregar ambiente Intel
set "SETVARS=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
if not exist "%SETVARS%" (
    echo [ERROR] Intel oneAPI setvars.bat nao encontrado.
    exit /b 1
)
:: Chamamos diretamente sem estar dentro de blocos complexos para evitar erro de parenteses
call "%SETVARS%"

:: 2. Limpar build anterior
if exist build rmdir /s /q build
mkdir build

:: 3. Configurar CMake
:: Usamos Ninja como gerador e forçamos o uso do icx
echo [INFO] Configurando CMake...
cmake -B build -G "Ninja" ^
    -DCMAKE_CXX_COMPILER=icx ^
    -DCMAKE_C_COMPILER=icx ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DUSE_GTEST=ON

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Falha na configuracao do CMake.
    exit /b %ERRORLEVEL%
)

:: 4. Compilar
echo [INFO] Compilando...
cmake --build build --config Release

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Falha na compilacao.
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] Build concluido com sucesso.
echo [INFO] Rodando setup de ambiente...
call setup_environment.bat
