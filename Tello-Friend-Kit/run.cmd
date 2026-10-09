@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Environment missing. Run setup.cmd first.
    popd
    exit /b 1
)
if "%~1"=="" (
    ".venv\Scripts\python.exe" -m tello_lab --help
) else (
    ".venv\Scripts\python.exe" -m tello_lab %*
)
set "taskResult=%ERRORLEVEL%"
popd
exit /b %taskResult%
