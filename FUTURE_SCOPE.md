# ALRC Future Scope: v5.x Neuro-Symbolic Architecture

## Current Status: v5.0 MVP (Production-Ready)
- ✅ Embedding-based semantic analysis
- ✅ Trajectory-based escalation detection
- ✅ Long-term memory drift tracking
- ✅ Circuit breaker for Crescendo attacks
- ✅ 80-85% attack detection rate

---

## v5.x Vision: Research-Grade Safety System

### Phase 1: Agency & Role Recognition (ARR)
**Timeline**: 3-4 weeks  
**Priority**: CRITICAL

**Problem**: Current system can't distinguish victim from aggressor
- "I'm scared of him" → BLOCKED ❌
- "I want to scare him" → Both have similar embeddings

**Solution**: Dependency parsing + Semantic Role Labeling
```python
# Syntax analysis
agent, action, patient = parse_dependencies(text)

# Rule: Victim statements are safe
if agent == "other" and patient == "user":
    return SAFE  # User is victim
```

**Technologies**:
- spaCy transformer model (`en_core_web_trf`)
- AllenNLP SRL for frame detection
- Symbolic constraint engine (FOL)

**Impact**: 95% accuracy on victim/aggressor classification

---

### Phase 2: Embedding Drift Detection (RCCB)
**Timeline**: 1-2 weeks  
**Priority**: HIGH

**Problem**: Vulnerable to automated attacks (GCG, TAP)
```
Attack: "Sure here's [optimized_nonsense_tokens]"
Current: Bypasses semantic filters ❌
```

**Solution**: Gaussian Mixture Model on safe embedding manifold
```python
if mahalanobis_distance(embedding, safe_manifold) > 3σ:
    return BLOCK  # Anomalous input
```

**Technologies**:
- GMM trained on 1M+ safe queries
- Mahalanobis distance calculation
- Circuit breaker pattern

**Impact**: 95% blocking of GCG/TAP/PAIR attacks

---

### Phase 3: Fictional Context Detection
**Timeline**: 2 weeks  
**Priority**: MEDIUM

**Problem**: Blocks creative writing requests
```
User: "Write a thriller about a bomb"
Current: BLOCKED ❌
```

**Solution**: Parse ROOT verb to detect creative frames
```python
if root_verb in ["write", "imagine", "draft"]:
    fictional_context = True
    if detail_level == "HIGH":
        return BLOCK  # Suspicious
    else:
        return ALLOW  # Creative writing
```

**Impact**: 90% accuracy on fictional vs real intent

---

### Phase 4: Sentiment Velocity & Hot/Cold Conflict
**Timeline**: 1-2 weeks  
**Priority**: MEDIUM

**Problem**: All escalation treated as threat
```
User expressing fear → BLOCKED ❌
Should route to crisis support, not block
```

**Solution**: Classify escalation type
```python
if sentiment_velocity < -0.5 and variance > 0.2:
    return HOT_CONFLICT  # Emotional, needs empathy
elif sentiment < -0.5 and variance < 0.1:
    return COLD_CONFLICT  # Threat, needs blocking
```

**Impact**: Reduce false positives by 60%

---

### Phase 5: GPT API Compatibility
**Timeline**: 1 week  
**Priority**: HIGH (for adoption)

**Problem**: Not a drop-in replacement for GPT
```
Current: POST /chat {"message": "..."}
GPT: POST /v1/chat/completions {...}
```

**Solution**: OpenAI-compatible wrapper
- Streaming responses
- Function calling support
- Token usage reporting
- System message handling

**Impact**: Can replace GPT-4 in existing apps

---

## Long-Term Roadmap (3-6 months)

### Multilingual Support
- ✅ English (current)
- ⏳ Spanish, French, German, Arabic
- Requires multilingual embedding models

### Learned Risk Weights
- Current: Hand-tuned weights (35% semantic, 25% trajectory)
- Future: Learn optimal weights from labeled data
- Use gradient-based optimization on attack datasets

### Adversarial Training Pipeline
- Continuous red-teaming with automated tools
- Update prototype sets weekly
- A/B testing safety configurations

### Integration with External Guardrails
- OpenAI Moderation API fallback
- Perspective API toxicity scores
- Custom company-specific policies

---

## Resource Requirements

### Compute
- **Development**: 1x RTX 4090 (for spaCy, SRL training)
- **Production**: 2x A100 (1 for LLM, 1 for safety stack)

### Data
- **Safe queries**: 1M+ (Reddit, StackOverflow, Wikipedia)
- **Adversarial attacks**: GCG, TAP, jailbreak datasets
- **Victim/Aggressor corpus**: Clinical notes, crisis counseling (anonymized)

### Libraries
```bash
pip install spacy transformers allennlp scikit-learn redis
python -m spacy download en_core_web_trf
```

---

## Success Metrics

| Metric | v5.0 Current | v5.x Target |
|--------|--------------|-------------|
| Attack detection | 85% | 98% |
| False positive rate | 30% | <5% |
| Latency (p95) | 120ms | <200ms |
| GCG/TAP blocking | 10% | 95% |
| Victim classification | 70% | 95% |

---

## Next Steps

1. **Weeks 1-2**: Build ARR foundation (dependency parsing)
2. **Weeks 3-4**: Implement symbolic constraint engine
3. **Weeks 5-6**: Add drift detection + circuit breaker
4. **Weeks 7-8**: Sentiment velocity improvements
5. **Weeks 9-10**: Integration, testing, GPT compatibility

**Total Timeline**: 10 weeks to production-ready v5.x
