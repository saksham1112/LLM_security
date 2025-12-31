#!/bin/bash
# Quick start script for Laminar v5.0

echo "🌊 Starting Laminar v5.0..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install dependencies
echo "Installing dependencies..."
pip install -q fastapi uvicorn pydantic numpy scikit-learn sentence-transformers redis ollama

# Check Redis
echo ""
echo "Checking Redis..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "⚠️  Redis not running. Phase L and Phase 3 features will be limited."
    echo "   Start Redis with: redis-server"
else
    echo "✅ Redis is running"
fi

# Check Ollama
echo ""
echo "Checking Ollama..."
if ! ollama list > /dev/null 2>&1; then
    echo "⚠️  Ollama not running. LLM responses will be unavailable."
    echo "   Start Ollama with: ollama serve"
else
    echo "✅ Ollama is running"
    
    # Check if dolphin model exists
    if ! ollama list | grep -q "dolphin-llama3"; then
        echo "⚠️  Dolphin model not found."
        echo "   Pull model with: ollama pull dolphin-llama3"
    else
        echo "✅ Dolphin model available"
    fi
fi

# Check UQ model
echo ""
if [ -f "models/conformal.json" ]; then
    echo "✅ Phase UQ model found"
else
    echo "⚠️  Phase UQ model not found (optional)"
    echo "   Calibrate with: python scripts/calibrate_uq.py"
fi

# Start server
echo ""
echo "="*60
echo "🚀 Starting Laminar v5.0 Server..."
echo "="*60
echo ""

python laminar_server.py
