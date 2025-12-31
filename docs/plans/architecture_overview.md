# Architecture Overview

## System Purpose
**Detect** risk in LLM conversations without enforcing interventions.

```
┌─────────────────────────────────────────────────────────────┐
│                    RISK DETECTION SYSTEM                     │
│                   (Detection Only - No Blocking)             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Input ──►[Risk Estimator]──►[Accumulator]──► Report   │
│       │              │                  │            │       │
│       │              ▼                  ▼            ▼       │
│       │        Risk Vector      Trajectory State   Metrics  │
│       │        (6 dimensions)   (6 modes)          (logs)   │
│       │                                                      │
│       ▼                                                      │
│  [LLM Backend] ──► Response ──►[Output Risk]──► Report      │
│                                                              │
│  NO BLOCKING - ONLY DETECTION & REPORTING                   │
└─────────────────────────────────────────────────────────────┘
```

## Core Principle: Detection-First

We **detect and report** risk without forcing interventions because:
1. Allows research on what risks actually occur
2. Avoids false positive user frustration
3. Generates labeled data for future models
4. Lets downstream systems decide on action

## What We Detect

### 1. Per-Turn Risk Vector (6 Dimensions)
```python
RiskVector:
  toxicity: float        # Harmful language
  harm_potential: float  # Dangerous capabilities
  manipulation: float    # Prompt injection attempts
  escalation_signal: float  # Pattern of escalation
  semantic_drift: float  # Topic deviation
  composite: float       # Weighted average
```

### 2. Trajectory State (6 Levels)
```
SAFE → CAUTIOUS → ELEVATED → SUSPECT → CRITICAL → UNSAFE
 0.0     0.15       0.30       0.50      0.70      0.85
```

### 3. Memory Signals (Prolonged Chats)
- Suspicious turn count
- Peak risk in window
- Escalation trend
- Topic drift over time
- Intent persistence

## Component Map

| Component | Location | Purpose |
|-----------|----------|---------|
| Risk Estimator | `src/risk/advanced.py` | TF-IDF + linguistic scoring |
| Risk Accumulator | `src/risk/accumulator.py` | Memory across turns |
| LLM Backend | `src/llm/ollama_backend.py` | Llama 3 integration |
| Controller | `src/controller/trajectory.py` | Orchestration |
| Logging | `src/logging/` | Metrics & structured logs |

## Data Flow

```
Turn N arrives
    │
    ▼
[Estimate Input Risk] ───────────────────────────► Log input_risk
    │
    ▼
[Update Memory Context] ──────────────────────────► Update trajectory
    │
    ▼
[Send to LLM with full context] ─────────────────► Ollama Llama 3
    │
    ▼
[Estimate Output Risk] ──────────────────────────► Log output_risk
    │
    ▼
[Update Accumulator] ─────────────────────────────► Update mode
    │
    ▼
[Generate Report] ────────────────────────────────► Return metrics

NO BLOCKING AT ANY STAGE
```

## What's NOT in Scope (Detection Phase)

❌ Blocking harmful requests  
❌ Modifying LLM output  
❌ Terminating sessions  
❌ User warnings  

These belong to a future **enforcement layer** that can use our detection signals.
