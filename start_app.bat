@echo off
echo Starting NeuroVoice AI Backend Server (Port 5000)...
start "NeuroVoice Backend" /min ".venv\Scripts\python.exe" backend\app.py

echo Starting NeuroVoice AI Frontend Web Server (Port 8000)...
start "NeuroVoice Frontend" /min python -m http.server 8000 --directory frontend

timeout /t 2 >nul
echo Opening Web Application in Browser...
start http://127.0.0.1:8000

echo Application successfully started!
