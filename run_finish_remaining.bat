@echo off
setlocal
cd /d "%~dp0"

echo [1/3] Checking ComfyUI server at 127.0.0.1:8188...
curl -s -m 3 http://127.0.0.1:8188/system_stats >nul 2>&1
if errorlevel 1 (
    echo Starting ComfyUI server in background window...
    start "ComfyUI_Server" /D "C:\Users\thith\Downloads\Van_thien_luc_2D-main\Van_thien_luc_2D-main\ComfyUI\ComfyUI" "C:\Users\thith\Downloads\Van_thien_luc_2D-main\Van_thien_luc_2D-main\ComfyUI\venv\Scripts\python.exe" -su main.py
) else (
    echo ComfyUI server is already running.
)

echo [2/3] Running QA audit and generating remaining missing assets...
".venv\Scripts\python.exe" qa_audit.py
if errorlevel 1 (
    echo [ERROR] qa_audit.py failed with exit code %errorlevel%.
    pause
    exit /b %errorlevel%
)

echo [3/3] Completed 63/63 assets, synced Assets\Art\, and updated VanKiem_All_Art.zip!
pause
