# Intent Tracking Plan

## Beyond Keywords: Understanding Intent

Keywords detect **WHAT** the user says.  
Intent modeling infers **WHY** they're asking.

```
Query: "How do explosives work?"
  ↓
Keywords: {explosives} → High risk!
  ↓
Intent analysis:
  - Student studying chemistry? → Safe intent
  - Demolition engineer? → Safe intent
  - Pattern of escalating questions? → Suspicious intent
  - Followed by "where to buy materials"? → Malicious intent
```

---

## Bayesian Intent Model

### Core Idea
Maintain a **probability distribution** over user intents, updated each turn.

```python
P(intent | observations) ∝ P(observations | intent) × P(intent)
```

### Intent Categories
```python
class Intent(Enum):
    SAFE = "safe"           # Benign curiosity
    EDUCATIONAL = "edu"     # Learning purpose
    PROFESSIONAL = "pro"    # Work-related
    CURIOUS = "curious"     # Boundary testing (not malicious)
    SUSPICIOUS = "suspicious"  # Unclear intent
    MALICIOUS = "malicious"    # Harmful intent
```

### Prior Distribution (Start of Session)
```python
prior = {
    Intent.SAFE: 0.70,
    Intent.EDUCATIONAL: 0.15,
    Intent.PROFESSIONAL: 0.10,
    Intent.CURIOUS: 0.03,
    Intent.SUSPICIOUS: 0.015,
    Intent.MALICIOUS: 0.005,
}
```

---

## Likelihood Functions

### P(observation | intent)

For each turn, compute likelihood under each intent:

```python
def likelihood_safe(turn: Message, context: list[Message]) -> float:
    """Likelihood of this turn given SAFE intent."""
    # Safe users ask diverse topics
    # Low risk keywords
    # No escalation pattern
    return (
        0.8 * topic_diversity(context) +
        0.2 * (1.0 - turn.risk_vector.composite)
    )

def likelihood_malicious(turn: Message, context: list[Message]) -> float:
    """Likelihood of this turn given MALICIOUS intent."""
    # High risk keywords
    # Escalation pattern
    # Information gathering pattern
    # Hypothetical framing
    return (
        0.3 * turn.risk_vector.composite +
        0.3 * escalation_score(context) +
        0.2 * info_gathering_score(context) +
        0.2 * hypothetical_framing_score(turn)
    )
```

---

## Posterior Update (Per Turn)

```python
class IntentTracker:
    def __init__(self):
        self.posterior = {intent: prior[intent] for intent in Intent}
    
    def update(self, turn: Message, context: list[Message]) -> dict[Intent, float]:
        """Bayesian update of intent distribution."""
        
        # Compute likelihoods
        likelihoods = {
            Intent.SAFE: likelihood_safe(turn, context),
            Intent.EDUCATIONAL: likelihood_educational(turn, context),
            Intent.PROFESSIONAL: likelihood_professional(turn, context),
            Intent.CURIOUS: likelihood_curious(turn, context),
            Intent.SUSPICIOUS: likelihood_suspicious(turn, context),
            Intent.MALICIOUS: likelihood_malicious(turn, context),
        }
        
        # Bayes update
        unnormalized = {
            intent: self.posterior[intent] * likelihoods[intent]
            for intent in Intent
        }
        
        # Normalize
        total = sum(unnormalized.values())
        self.posterior = {
            intent: prob / total
            for intent, prob in unnormalized.items()
        }
        
        return self.posterior
    
    @property
    def malicious_risk(self) -> float:
        """Probability of malicious or suspicious intent."""
        return self.posterior[Intent.MALICIOUS] + self.posterior[Intent.SUSPICIOUS]
```

---

## Intent Signals for Detection

### Signal 1: Escalation + Malicious Intent
```python
if risk_accumulator.trend > 0.3 and intent_tracker.malicious_risk > 0.2:
    # Escalating trajectory with suspicious intent
    risk_boost = 0.3
```

### Signal 2: Information Gathering Pattern
```python
def detect_info_gathering(context: list[Message]) -> float:
    """Detect systematic information collection."""
    patterns = [
        ("what is", "how does"),      # Definition → Mechanism
        ("how does", "weaknesses"),   # Mechanism → Vulnerabilities
        ("weaknesses", "exploit"),    # Vulnerabilities → Exploitation
    ]
    
    matches = 0
    for i, (pattern1, pattern2) in enumerate(patterns):
        for j, turn in enumerate(context[:-1]):
            if pattern1 in turn.content.lower():
                for later_turn in context[j+1:]:
                    if pattern2 in later_turn.content.lower():
                        matches += 1
                        break
    
    return matches / len(patterns)
```

### Signal 3: Context Switch After Probing
```python
def detect_suspicious_pivot(context: list[Message]) -> float:
    """Detect topic change after hitting safety boundary."""
    
    for i, turn in enumerate(context[:-1]):
        # If a turn was risky
        if turn.risk_vector.composite > 0.5:
            # And next turn is benign but unrelated
            next_turn = context[i + 1]
            if (next_turn.risk_vector.composite < 0.1 and 
                topic_similarity(turn, next_turn) < 0.3):
                # Strategic pivot detected
                return 0.5
    
    return 0.0
```

---

## Implementation Plan

### Phase 1: Basic Intent Tracking
```python
# src/intent/basic.py

class BasicIntentTracker:
    """Rule-based intent signals (no ML)."""
    
    def __init__(self):
        self.turn_count = 0
        self.risky_turns = 0
        self.escalation_count = 0
    
    def update(self, turn: Message, context: list[Message]) -> dict[str, float]:
        self.turn_count += 1
        
        if turn.risk_vector.composite > 0.3:
            self.risky_turns += 1
        
        if len(context) > 1 and turn.risk_vector.composite > context[-2].risk_vector.composite:
            self.escalation_count += 1
        
        return {
            "risky_turn_ratio": self.risky_turns / self.turn_count,
            "escalation_ratio": self.escalation_count / max(self.turn_count - 1, 1),
            "suspicion_score": self._compute_suspicion(),
        }
```

### Phase 2: Bayesian Model (Pyro)
```python
# src/intent/bayesian.py

import pyro
import pyro.distributions as dist

class BayesianIntentTracker:
    def __init__(self):
        self.intent_prior = dist.Categorical(torch.tensor([0.7, 0.15, 0.1, 0.03, 0.015, 0.005]))
        self.observations = []
    
    def model(self, observations):
        intent = pyro.sample("intent", self.intent_prior)
        
        for obs in observations:
            likelihood = self.likelihood_fn(obs, intent)
            pyro.sample(f"obs_{obs.turn}", dist.Bernoulli(likelihood), obs=obs.is_risky)
    
    def infer(self):
        posterior = pyro.infer.Importance(self.model, num_samples=1000)
        return posterior.run(self.observations)
```

### Phase 3: Neural Intent Classifier
```python
# src/intent/neural.py (future)

class NeuralIntentClassifier:
    """Fine-tuned transformer for intent classification."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def predict(self, conversation: list[str]) -> dict[Intent, float]:
        # Encode full conversation
        inputs = self.tokenizer(
            " [SEP] ".join(conversation),
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        
        return {intent: probs[0, i].item() for i, intent in enumerate(Intent)}
```

---

## Priority Order

| Priority | Component | Complexity | Benefit |
|----------|-----------|------------|---------|
| 1 | Basic Intent Signals | Low | Quick wins |
| 2 | Pattern Detection | Medium | Catch escalation |
| 3 | Bayesian Tracking | High | Principled updates |
| 4 | Neural Classifier | High | Best accuracy |

---

## Integration with Risk Detection

```python
# In TrajectoryController

def process_turn(self, session_id, message):
    # 1. Get input risk
    input_risk = await self.risk_estimator.estimate(message, context)
    
    # 2. Update intent tracker
    intent_signals = self.intent_tracker.update(turn, context)
    
    # 3. Combine signals
    combined_risk = RiskVector(
        **input_risk.to_dict(),
        intent_risk=intent_signals["suspicion_score"],  # Add intent signal
    )
    
    # 4. Update accumulator
    self.accumulator.accumulate(combined_risk)
```

The intent tracker provides **additional context** for risk detection, not replacement.
