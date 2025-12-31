@echo off
REM Quick start script for Laminar v5.0 (Windows)

echo 🌊 Starting Laminar v5.0...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Dependency installation failed.
    pause
    exit /b %errorlevel%
)

REM Check Redis
echo.
echo Checking Redis...
redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Redis not running. Phase L and Phase 3 features will be limited.
    echo    Download from: https://github.com/microsoftarchive/redis/releases
) else (
    echo ✅ Redis is running
)

REM Check Ollama
echo.
echo Checking Ollama...
ollama list >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Ollama not running. LLM responses will be unavailable.
    echo    Download from: https://ollama.ai
) else (
    echo ✅ Ollama is running
    
    REM Check if dolphin model exists
    ollama list | findstr "dolphin-llama3" >nul
    if errorlevel 1 (
        echo ⚠️  Dolphin model not found.
        echo    Pull model with: ollama pull dolphin-llama3
    ) else (
        echo ✅ Dolphin model available
    )
)

REM Check UQ model
echo.
if exist "models\conformal.json" (
    echo ✅ Phase UQ model found
) else (
    echo ⚠️  Phase UQ model not found (optional^)
    echo    Calibrate with: python scripts\calibrate_uq.py
)

REM Start server
echo.
echo ============================================================
echo 🚀 Starting Laminar v5.0 Server...
echo ============================================================
echo.

python laminar_server.py
