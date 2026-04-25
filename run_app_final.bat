@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
call build_project.bat
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo [INFO] Running Neural Path Energy Test...
cd build
ctest -R NeuralPathTest --output-on-failure
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Neural path energy test failed. Signal is getting zeroed out.
    pause
    exit /b 1
)
echo [INFO] Starting Silence Arc...
silence_arc.exe
