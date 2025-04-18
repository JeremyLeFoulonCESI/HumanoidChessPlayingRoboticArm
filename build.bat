@echo off
:: Batch script that calls pyinstaller to build our project.
:: Usage:
:: build.bat <path to pyinstaller.exe> [source files to include...]

setlocal

set allargs=%*

set PYINSTALLER=%1

call set someargs=%%allargs:*%1=%%


set command=%PYINSTALLER% ^
    --add-data src/stockfish-windows-x86-64-avx2.exe:. ^
    --hide-console minimize-late ^
    --add-data src/camera_calibration.json:. ^
    -y ^
    %someargs%

echo Command:
echo %command%


echo running command...
%command%


