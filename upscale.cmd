@echo off
setlocal
chcp 65001 >nul
set "RESIZE_ROOT=%~dp0"
set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V3\.venv\Scripts\python.exe"
if /I "%~1"=="print" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V4\.venv\Scripts\python.exe"
if /I "%~1"=="v4" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V4\.venv\Scripts\python.exe"
if /I "%~1"=="vector" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V4\.venv\Scripts\python.exe"
if /I "%~1"=="layers" if exist "%RESIZE_ROOT%APP\engines\V5\.venv\Scripts\python.exe" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V5\.venv\Scripts\python.exe"
if /I "%~1"=="layer" if exist "%RESIZE_ROOT%APP\engines\V5\.venv\Scripts\python.exe" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V5\.venv\Scripts\python.exe"
if /I "%~1"=="v5" if exist "%RESIZE_ROOT%APP\engines\V5\.venv\Scripts\python.exe" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V5\.venv\Scripts\python.exe"
if /I "%~1"=="repair" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V7\.venv\Scripts\python.exe"
if /I "%~1"=="v7" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V7\.venv\Scripts\python.exe"
rem Review is only a local browser UI. Prefer the shared V3 runtime so both
rem V5 and V7 checkpoints work without requiring the other engine's venv.
if /I "%~1"=="review" if exist "%RESIZE_ROOT%APP\engines\V3\.venv\Scripts\python.exe" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V3\.venv\Scripts\python.exe"
if /I "%~1"=="duyet" if exist "%RESIZE_ROOT%APP\engines\V3\.venv\Scripts\python.exe" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V3\.venv\Scripts\python.exe"
if /I "%~1"=="review" if not exist "%RESIZE_ROOT%APP\engines\V3\.venv\Scripts\python.exe" if exist "%RESIZE_ROOT%APP\engines\V5\.venv\Scripts\python.exe" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V5\.venv\Scripts\python.exe"
if /I "%~1"=="duyet" if not exist "%RESIZE_ROOT%APP\engines\V3\.venv\Scripts\python.exe" if exist "%RESIZE_ROOT%APP\engines\V5\.venv\Scripts\python.exe" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V5\.venv\Scripts\python.exe"
if /I "%~1"=="review" if not exist "%RESIZE_ROOT%APP\engines\V3\.venv\Scripts\python.exe" if not exist "%RESIZE_ROOT%APP\engines\V5\.venv\Scripts\python.exe" if exist "%RESIZE_ROOT%APP\engines\V7\.venv\Scripts\python.exe" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V7\.venv\Scripts\python.exe"
if /I "%~1"=="duyet" if not exist "%RESIZE_ROOT%APP\engines\V3\.venv\Scripts\python.exe" if not exist "%RESIZE_ROOT%APP\engines\V5\.venv\Scripts\python.exe" if exist "%RESIZE_ROOT%APP\engines\V7\.venv\Scripts\python.exe" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V7\.venv\Scripts\python.exe"

if not exist "%RESIZE_PYTHON%" (
  echo LOI: Khong tim thay Python noi bo trong APP.
  exit /b 2
)

"%RESIZE_PYTHON%" -B "%RESIZE_ROOT%APP\upscale_cli.py" %*
exit /b %errorlevel%
