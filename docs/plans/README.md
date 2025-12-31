# Documentation & Plans Index

This folder contains all architectural plans, tracking documents, and implementation guides.

## Plan Documents

| Document | Purpose |
|----------|---------|
| [architecture_overview.md](architecture_overview.md) | High-level system architecture |
| [risk_detection_plan.md](risk_detection_plan.md) | Risk detection methods & formulas |
| [memory_system_plan.md](memory_system_plan.md) | Prolonged chat memory handling |
| [intent_tracking_plan.md](intent_tracking_plan.md) | Bayesian intent inference |
| [testing_guide.md](testing_guide.md) | How to test with Ollama & multi-turn |
| [implementation_status.md](implementation_status.md) | What's done vs planned |

## Quick Reference

### Current Focus: DETECTION ONLY
We're building the **detection layer** - identifying risk without enforcing interventions yet.

### Key Principle
```
Detect risk trajectory → Report findings → No forced intervention
```

### Phase Status
- ✅ Phase 1: Core Infrastructure (Complete)
- 🔄 Phase 2: Memory & Intent (In Progress)
- 📋 Phase 3: Adversarial Testing (Planned)
