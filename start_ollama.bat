@echo off
REM Ollama Startup Script
REM This ensures Ollama starts properly and is accessible

echo Starting Ollama Server...

REM Kill any existing Ollama processes
taskkill /F /IM ollama.exe /T 2>nul

REM Wait for processes to terminate
timeout /t 2 /nobreak >nul

REM Start Ollama (this will run in the background)
start /B "" "C:\Users\zakup\AppData\Local\Programs\Ollama\ollama.exe" serve

REM Wait for Ollama to start
echo Waiting for Ollama to initialize...
timeout /t 5 /nobreak >nul

REM Test connection
echo Testing Ollama connection...
curl -s http://localhost:11434/api/tags >nul 2>&1

if %ERRORLEVEL% == 0 (
    echo ✓ Ollama is running successfully on port 11434
) else (
    echo ✗ Ollama failed to start properly
    echo Please start Ollama desktop app manually
)

pause
