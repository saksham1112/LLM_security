# Git Push Checklist - Product Only

## ✅ INCLUDE (Production Code)

### Core Application
```
✅ src/
   ✅ safety/alrc/          # ALRC v4.0 implementation
      - adaptive_normalizer.py
      - topic_memory.py
      - semantic_engine.py
      - risk_calculator.py
      - risk_floor.py
      - risk_logger.py
      - redis_stwm.py
      - pipeline.py
      - __init__.py
   ✅ llm/
      - dolphin_ollama.py   # LLM backend
      - __init__.py
   ✅ logging/              # Logging utilities
```

### Frontend
```
✅ frontend/laminar/
   - index.html
   - app.js
   - styles.css
```

### Server
```
✅ laminar_server.py        # Main server
```

### Configuration
```
✅ README.md                # Simple, descriptive
✅ requirements.txt         # Dependencies
✅ .gitignore              # Git exclusions
✅ start_ollama.bat        # Helper script
```

### Optional Utilities
```
✅ export_onnx.py           # Model export tool
✅ OLLAMA_SETUP.py          # Setup helper
```

---

## ❌ EXCLUDE (Not for Git)

### Logs (Private Data)
```
❌ logs/                   # Session logs, analysis
❌ *.log
```

### Test Files
```
❌ test_pipeline.py
❌ test_persona_shift.py
❌ tests/
```

### Documentation (Keep in separate branch/docs)
```
❌ docs/                   # Too much detail for main repo
❌ CLEANUP_PLAN.md
❌ CHECKPOINT_*.md
```

### Development Files
```
❌ __pycache__/
❌ .pytest_cache/
❌ .venv/
❌ *.pyc
❌ threshold_results.txt
```

### Models (Too large)
```
❌ models/                 # Download separately
❌ *.onnx
```

### Config Files (Optional)
```
❌ configs/
❌ pyproject.toml          # Optional, we have requirements.txt
```

---

## Final Structure for Git

```
secure_LLM/
├── src/
│   ├── llm/
│   │   ├── dolphin_ollama.py
│   │   └── __init__.py
│   ├── logging/
│   │   └── (logging utilities)
│   └── safety/
│       └── alrc/
│           ├── adaptive_normalizer.py
│           ├── topic_memory.py
│           ├── semantic_engine.py
│           ├── risk_calculator.py
│           ├── risk_floor.py
│           ├── risk_logger.py
│           ├── redis_stwm.py
│           ├── pipeline.py
│           └── __init__.py
├── frontend/
│   └── laminar/
│       ├── index.html
│       ├── app.js
│       └── styles.css
├── laminar_server.py
├── export_onnx.py
├── OLLAMA_SETUP.py
├── start_ollama.bat
├── README.md
├── requirements.txt
└── .gitignore
```

**Estimated Size**: ~50 files (clean, production-ready)
