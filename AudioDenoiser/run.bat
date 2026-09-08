@echo off
rem Launch AudioDenoiser (real-time mic denoising UI)
cd /d "%~dp0"
"..\.venv\Scripts\python.exe" audio_denoiser_app.py
pause
