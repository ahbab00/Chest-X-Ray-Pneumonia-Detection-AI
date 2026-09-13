@echo off
title Pneumonia Detection AI Web App
echo ===================================================
echo   Starting Pneumonia Detection Web App...
echo ===================================================
echo.
echo  Access locally: http://127.0.0.1:8501
echo ===================================================
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start "" "http://127.0.0.1:8501"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8501
pause
