"""
Setup instructions for Ollama + Llama 3.

Follow these steps to get Ollama running with Llama 3.
"""

# ============================================================================
# OLLAMA SETUP INSTRUCTIONS
# ============================================================================

## Step 1: Install Ollama
# 
# Windows:
#   Download from: https://ollama.com/download/windows
#   Or use winget: winget install Ollama.Ollama
#
# macOS:
#   Download from: https://ollama.com/download/mac
#   Or use brew: brew install ollama
#
# Linux:
#   curl -fsSL https://ollama.com/install.sh | sh

## Step 2: Start Ollama Server
#
# The Ollama service should start automatically after installation.
# If not, run:
#   ollama serve
#
# This starts the server on http://localhost:11434

## Step 3: Pull Llama 3 Model
#
# Download the Llama 3 8B model:
#   ollama pull llama3
#
# This will download ~4.7GB. Other options:
#   ollama pull llama3:70b     # Llama 3 70B (very large, ~40GB)
#   ollama pull llama3.1        # Llama 3.1 (latest)
#   ollama pull mistral         # Alternative: Mistral 7B

## Step 4: Test Ollama
#
# Quick test in terminal:
#   ollama run llama3
#
# Then type a message to verify it works.
# Press Ctrl+D to exit.

## Step 5: Verify API is Running
#
# Check if Ollama API responds:
#   curl http://localhost:11434/api/tags
#
# Should return JSON with available models.

# ============================================================================
# QUICK START
# ============================================================================

# If Ollama is already installed and running:
#   1. ollama pull llama3
#   2. python test_ollama.py         # Test connection
#   3. python test_with_controller.py  # Test with safety system

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Issue: "Connection refused"
# → Make sure Ollama is running: ollama serve

# Issue: "Model not found"
# → Pull the model: ollama pull llama3

# Issue: "Slow generation"
# → Llama 3 8B needs ~8GB RAM
# → Try smaller model: ollama pull phi

# Issue: "Out of memory"
# → Close other applications
# → Use quantized model (default is Q4)
# → Try: ollama pull llama3:7b-q4_0

print(__doc__)
