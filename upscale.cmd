@echo off
setlocal
chcp 65001 >nul
set "RESIZE_ROOT=%~dp0"
set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V3\.venv\Scripts\python.exe"
if /I "%~1"=="print" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V4\.venv\Scripts\python.exe"
if /I "%~1"=="v4" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V4\.venv\Scripts\python.exe"
if /I "%~1"=="vector" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\engines\V4\.venv\Scripts\python.exe"

if not exist "%RESIZE_PYTHON%" (
  echo LOI: Khong tim thay Python noi bo trong APP.
  exit /b 2
)

"%RESIZE_PYTHON%" -B "%RESIZE_ROOT%APP\upscale_cli.py" %*
exit /b %errorlevel%
