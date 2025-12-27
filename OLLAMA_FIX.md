# Ollama Connection Fix - README

## Problem
Ollama `serve` command gets stuck during initialization when started from PowerShell background processes, causing port 11434 to not listen properly.

## Root Cause
- PowerShell background jobs and `Start-Process` don't properly attach to console
- Ollama needs a proper terminal with console handle
- Error message: "failed to get console handle is invalid"

## Solution

### Option 1: Use Ollama Desktop App (RECOMMENDED)
1. Launch **Ollama** from Start Menu or system tray
2. Icon will appear in system tray (bottom-right)
3. Right-click → Verify "Ollama is running"
4. Test: Run `ollama list` in terminal

### Option 2: Run Batch Script
1. Double-click `start_ollama.bat` in this directory
2. A window will open and start Ollama
3. Keep that window open
4. Test connection shows in the window

### Option 3: Manual PowerShell
Open a **dedicated PowerShell window** and run:
```powershell
ollama serve
```
Keep the window open - don't run in background!

## Verification
After starting Ollama, test the connection:
```powershell
ollama list
# Should show: zephyr:latest

curl http://localhost:11434/api/tags
# Should return JSON with models list
```

## For Future Reference
- ❌ **Don't use**: `Start-Job`, `Start-Process -NoNewWindow`, background `&`
- ✅ **Do use**: Ollama desktop app, dedicated terminal, or `start_ollama.bat`

## Integration with Server
Once Ollama is running:
1. Server auto-detects on http://localhost:11434
2. Refresh browser at http://localhost:8000
3. Chat messages will work
4. Laminar metrics will update in real-time
