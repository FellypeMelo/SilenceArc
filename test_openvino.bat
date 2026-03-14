@echo off
setlocal

:: 1. Setup oneAPI Environment
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64

:: 2. Setup OpenVINO Environment
call "openvino_genai_windows_2026.0.0.0_x86_64\setupvars.bat"

:: 3. Build Project
echo [INFO] Building SilenceArc with OpenVINO...
cmake --build build --config Release

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed.
    pause
    exit /b %ERRORLEVEL%
)

:: 4. Run Neural Path Energy Test
echo [INFO] Running OpenVINO Hybrid Path Energy Test...
cd build
ctest -R NeuralPathTest --output-on-failure

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Neural path energy test failed. Check logs.
    pause
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] OpenVINO Hybrid Path verified!
echo [INFO] Launching Silence Arc...
cd bin\Release
silence_arc.exe
pause
