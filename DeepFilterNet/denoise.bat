@echo off
rem Denoise wav files with DeepFilterNet3 using the local repo + venv.
rem Usage: denoise.bat <noisy_file.wav> [more files...]
rem Output lands in out\ next to this script.
setlocal
set ROOT=%~dp0
set PYTHONPATH=%ROOT%DeepFilterNet
"%ROOT%..\.venv\Scripts\python.exe" "%ROOT%DeepFilterNet\df\enhance.py" --model-base-dir "%ROOT%models\DeepFilterNet3" -o "%ROOT%out" %*
endlocal
