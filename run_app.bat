call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
call build_project.bat
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
cd build\bin
start silence_arc.exe
