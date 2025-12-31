# Files to Remove Before Git Upload

## Root Directory - Remove These Demo/Test Files

### Demo Files (Old experiments - REMOVE):
- demo_dolphin_sxs.py
- demo_granular_modes.py
- demo_inference_control.py
- demo_laminar_framework.py
- demo_long_conv_protection.py
- demo_risk_scoring.py
- demo_trajectory.py

### Old Test Files (Superseded by test_pipeline.py - REMOVE):
- test_deepseek_api.py
- test_multi_turn.py
- test_norms.py
- test_ollama.py
- test_orch.py
- test_performance.py
- test_system.py
- test_thresholds.py
- test_with_controller.py
- threshold_results.txt

### Old Server Files (Not using ALRC v4.0 - REMOVE):
- server.py (old)
- sxs_ollama_server.py (old SxS)
- sxs_server.py (old SxS)
- ollama_chat_proof.py (proof of concept)

### Setup Files (Keep but update):
- OLLAMA_FIX.md (keep for reference)
- OLLAMA_SETUP.py (keep)
- pyproject.toml (keep)
- requirements-lite.txt (remove - we have requirements.txt)

---

## src/ Directory - Remove Old Architecture

### Directories to REMOVE (Old architecture not used in ALRC v4.0):
- adversarial/
- agents/
- api/ (old API)
- baselines/
- cbf/ (Control Barrier Functions - not in v4.0)
- control/
- controller/
- decoding/
- evaluator/
- intent/
- latent/
- metrics/
- orchestrator/
- probe/
- probes/
- reachability/
- risk/ (old risk system)
- state/
- steering/

### Files to REMOVE in src/:
- config.py (old config)
- main.py (old main)
- types.py (old types)

---

## src/safety/ - Keep ALRC, Remove Old

### Keep:
- alrc/ (ENTIRE directory - this is ALRC v4.0)
  - adaptive_normalizer.py
  - topic_memory.py
  - semantic_engine.py
  - risk_calculator.py
  - risk_floor.py
  - risk_logger.py
  - redis_stwm.py
  - pipeline.py
  - __init__.py

### REMOVE (Old safety systems):
- bert_classifier.py
- domain_tiers.py
- embedding_classifier.py
- llama_guard.py
- llama_guard_hf.py
- output_filter.py
- pii_detector.py
- pipeline.py (old pipeline, NOT alrc/pipeline.py)
- post_filter.py
- session_tracker.py (replaced by ALRC memory)
- simple_rules.py
- text_normalizer.py
- toxicity_classifier.py
- train_probe.py
- zone_contracts.py

---

## src/llm/ - Keep Only Dolphin Backend

Check this directory - keep only:
- dolphin_ollama.py (our backend)
- __init__.py

Remove any old/unused LLM backends

---

## KEEP These Important Directories:

✅ **logs/** - ALL session logs and risk analysis (KEEP ALL)
✅ **docs/** - ALL documentation and plans (KEEP ALL)
✅ **frontend/laminar/** - Web UI (KEEP)
✅ **models/** - ONNX models if exported (KEEP)

---

## KEEP These Root Files:

✅ laminar_server.py - Current production server
✅ export_onnx.py - Model export
✅ test_pipeline.py - Main verification test
✅ test_persona_shift.py - Jailbreak test
✅ README.md - Updated
✅ requirements.txt - Updated
✅ .gitignore - Created
✅ start_ollama.bat - Helper script

---

## Summary Action Plan:

1. Remove all demo_*.py files
2. Remove old test_*.py files (except test_pipeline.py, test_persona_shift.py)
3. Remove old server files (server.py, sxs_*.py)
4. Remove entire old architecture in src/ (keep only llm/ and safety/)
5. Clean src/safety/ to keep only alrc/ directory
6. Keep logs/, docs/, frontend/, models/
7. Verify src/llm/ has only needed files

This will leave a clean, production ALRC v4.0 codebase!
