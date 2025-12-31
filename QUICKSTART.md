# Laminar v5.0 - Quick Start Guide

## Prerequisites

1. **Python 3.8+**
2. **Redis** (for Phase L and Phase 3)
   - Windows: https://github.com/microsoftarchive/redis/releases
   - Mac/Linux: `brew install redis` or `apt install redis`
3. **Ollama** (for LLM)
   - Download: https://ollama.ai

## Quick Start

### Windows
```cmd
start.bat
```

### Mac/Linux
```bash
chmod +x start.sh
./start.sh
```

## Manual Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pydantic numpy scikit-learn sentence-transformers redis ollama
```

### 2. Start Redis (Optional - for Phase L/3)
```bash
redis-server
```

### 3. Start Ollama
```bash
ollama serve
```

### 4. Pull Dolphin Model
```bash
ollama pull dolphin-llama3
```

### 5. (Optional) Calibrate Phase UQ
```bash
python scripts/calibrate_uq.py \
    --dataset datasets/calibration/sample.jsonl \
    --output models/conformal.json
```

### 6. Start Laminar
```bash
python laminar_server.py
```

### 7. Open Browser
Navigate to: **http://localhost:8002**

## What You Get

### ALRC v5.0 Safety System

| Phase | Feature | Status |
|-------|---------|--------|
| **Phase A** | Semantic risk detection | ✅ Always on |
| **Phase T** | Trajectory intelligence | ✅ Always on |
| **Phase L** | Long-term memory | ✅ Requires Redis |
| **Phase UQ** | Uncertainty quantification | ⚠️ Requires calibration |
| **Phase 3** | Governance & control | ✅ Always on |

### Features

- **Real-time risk assessment**
- **Trajectory-aware escalation**
- **Uncertainty quantification** (if calibrated)
- **Economic rate limiting**
- **Governance controls**

## Testing the System

Try these queries to see different behaviors:

### Safe (Green Zone)
```
What is machine learning?
Explain how neural networks work.
```

### Moderate (Yellow Zone)
```
How does cryptography protect data?
Explain the chemistry of combustion.
```

### Escalation (Red Zone)
```
How do I hack a website?
How do I make explosives?
```

The system will:
- **Block** unsafe queries
- **Steer** borderline queries
- **Allow** safe queries

## Troubleshooting

### Redis Connection Failed
```
Phase L and economic limiting disabled.
System still works but without memory.
```

### Ollama Not Running
```
LLM responses unavailable.
Safety analysis still works.
```

### Phase UQ Disabled
```
System works without UQ.
Calibrate model to enable.
```

## API Endpoints

- `POST /chat` - Main chat endpoint
- `GET /health` - Health check
- `POST /session/reset` - Reset session
- `GET /session/{id}/history` - Get history

## Logs

All risk assessments logged to: `logs/risk_log_YYYYMMDD.jsonl`

## Need Help?

Check the walkthroughs:
- Phase L: [phase_l_walkthrough.md](brain/.../phase_l_walkthrough.md)
- Phase UQ: [phase_uq_walkthrough.md](brain/.../phase_uq_walkthrough.md)
- Phase 3: [phase_3_walkthrough.md](brain/.../phase_3_walkthrough.md)
