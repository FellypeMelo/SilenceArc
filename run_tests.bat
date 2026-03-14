@echo off
set ONEDNN_VERBOSE=all
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
ctest --test-dir build %* --output-on-failure
