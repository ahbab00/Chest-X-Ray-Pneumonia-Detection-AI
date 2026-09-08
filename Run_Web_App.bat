@echo off
title Pneumonia Detection AI Web App
echo ===================================================
echo   Starting Pneumonia Detection Web App...
echo ===================================================
echo.
echo  Access on this PC:   http://localhost:8501
echo  Access on your Phone: http://192.168.1.4:8501
echo  (Phone must be connected to the SAME Wi-Fi!)
echo ===================================================
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start "" "http://localhost:8501"
streamlit run app.py --server.address 0.0.0.0 --server.headless true
pause
