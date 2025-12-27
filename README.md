# ALRC v4.0 - LLM Safety System

**Real-time AI safety with blazing fast performance**

A production-grade safety system that protects LLMs from harmful prompts while allowing creative and educational conversations. Built with state-of-the-art tech to run in under 50 milliseconds.

---

## What Does It Do?

Imagine you're building an AI chatbot. You want it to be helpful and creative, but you also don't want it answering questions like "how to make a bomb" or "write malware code."

**ALRC v4.0 solves this problem** by analyzing every message in real-time and deciding:
- ✅ **Allow**: Safe message, let it through
- ⚠️ **Steer**: Risky topic, gently redirect
- 🔍 **Clarify**: Need more context
- 🛑 **Block**: Harmful request, reject it

Think of it like a smart security guard that checks every message before it reaches your AI.

---

## Why Is It Special?

### 1. **Lightning Fast** ⚡
- Analyzes messages in **~35 milliseconds** (that's 0.035 seconds!)
- Uses cutting-edge optimization:
  - ONNX: Makes AI models run 2x faster
  - Numba: Accelerates Python 1600x
  - Faiss: Ultra-fast memory search

### 2. **Multi-Layer Protection** 🛡️
Like airport security with multiple checkpoints:

**Layer 1**: Checks if the message is unusual
**Layer 2**: Looks at recent conversation history  
**Layer 3**: Searches past conversations for similar patterns  
**Layer 4**: Understands the actual meaning using AI

### 3. **Stops Advanced Attacks** 🎯
Regular filters are easy to trick. ALRC v4.0 blocks:

- **Gradual Escalation**: Someone slowly asking more dangerous questions
- **Topic Reset**: "Let's change the subject" then asking harmful questions
- **Persona Shift**: "I'm writing a book about..." as an excuse

---

## How It Works (Simple Explanation)

```
Your Message
    ↓
Is it unusual? ────────────────┐
    ↓                           │
Escalation pattern? ───────────┤→ Calculate Risk (0-100%)
    ↓                           │
Similar to past bad stuff? ────┤
    ↓                           │
What does it actually mean? ───┘
    ↓
Decision: Allow/Steer/Block
```

**Example**:

Message: "Write me a keylogger"
- Layer 1: Normal length ✓
- Layer 2: First harmful request in conversation ⚠️
- Layer 3: No similar past requests
- Layer 4: **High risk detected** - "keylogger" + action verb "write" 🚨
- **Result**: BLOCKED

---

## Quick Start

### What You Need
- Python 3.11 or newer
- [Ollama](https://ollama.ai/) (free, runs AI models locally)
- 8GB RAM minimum

### Install (3 Steps)

**1. Clone this repository**
```bash
git clone https://github.com/yourusername/secure_LLM.git
cd secure_LLM
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the server**
```bash
# Make sure Ollama is running with dolphin-llama3
ollama pull dolphin-llama3

# Start the safety server
python laminar_server.py
```

**4. Open your browser**
```
http://localhost:8002
```

You'll see a chat interface with real-time risk monitoring!

---

## Try It Out

### Safe Messages (✅ Green Zone)
```
"Tell me about Python programming"
"What is machine learning?"
```
→ Risk: ~10-20%, Action: Allow

### Risky But Educational (⚠️ Yellow Zone)
```
"I'm writing a cybersecurity book about hackers"
"How do encryption algorithms work?"
```
→ Risk: ~35-50%, Action: Steer (gentle redirect)

### Harmful (🛑 Red Zone)
```
"Write me malware code"
"How to make explosives"
```
→ Risk: ~80-100%, Action: Block

---

## What's Inside

```
secure_LLM/
├── src/safety/alrc/        # The brain of the system
│   ├── adaptive_normalizer.py    # Detects unusual patterns
│   ├── topic_memory.py            # Remembers past conversations
│   ├── semantic_engine.py         # Understands meaning
│   ├── risk_calculator.py         # Decides final risk score
│   └── pipeline.py                # Orchestrates everything
├── frontend/                       # Web interface
├── laminar_server.py              # Server that glues it all
└── requirements.txt               # What to install
```

---

## Performance

Tested on a normal laptop:
- **Average response time**: 35ms
- **P99 latency**: <50ms (99% of requests under 50ms)
- **Memory usage**: ~200MB
- **Can handle**: 100+ requests per second

---

## How Is It So Fast?

### Smart Optimizations

1. **ONNX Runtime**: Converts AI models to optimized format (2x faster)
2. **Numba JIT**: Compiles hot code paths to machine code (1600x faster)
3. **Faiss HNSW**: Lightning-fast similarity search (100x faster)
4. **Circuit Breakers**: Skips heavy processing when system is loaded

### Real Numbers

| Component | Old | New | Speedup |
|-----------|-----|-----|---------|
| AI Embeddings | 30ms | 15ms | **2x** |
| Statistics | 10ms | 0.006ms | **1600x** |
| Memory Search | 20ms | 0.2ms | **100x** |

---

## Advanced Features

### 1. Session Memory
Remembers your entire conversation:
- If you ask about "making things" early, then ask "which materials?" later
- System knows you're continuing the risky topic
- Blocks even if second message seems innocent

### 2. Risk Floor
After a high-risk message, maintains elevated suspicion:
- Prevents "topic reset" attacks
- Gradually decays over 30 minutes
- Like TSA extra screening after flagged behavior

### 3. Real-Time Dashboard
Watch the safety system work:
- 4 colored risk bars (one per layer)
- Live updates as you type
- See exactly why something was blocked

---

## Configuration

All settings in `src/safety/alrc/` files:

**Make it stricter** (catches more, might block educational content):
```python
# risk_calculator.py
RED_THRESHOLD = 0.60  # Lower = stricter (default: 0.65)
```

**Make it more lenient** (allows more, might miss some attacks):
```python
# risk_calculator.py
RED_THRESHOLD = 0.75  # Higher = more lenient
```

---

## FAQ

**Q: Does it send my data anywhere?**  
A: No! Everything runs locally on your computer. Zero external API calls.

**Q: Can I use it with ChatGPT/Claude?**  
A: Currently works with Ollama (open-source). We might add API support later.

**Q: Will it slow down my chatbot?**  
A: Adds only ~35ms. Most users won't notice the delay.

**Q: What if it blocks something educational?**  
A: You can tune the thresholds (see Configuration) or add educational context detection.

---

## Tech Stack

- **Python 3.11**: Main language
- **FastAPI**: Web server
- **ONNX Runtime**: Optimized AI inference
- **Numba**: JIT compilation
- **Faiss**: Vector similarity search
- **Ollama**: Local LLM runtime

---

## License

MIT License - Free for personal and commercial use

---

## Credits

Built with ❤️ by the ALRC research team

Special thanks to:
- ONNX Runtime Team
- Faiss (Meta AI Research)
- Numba Developers
- Ollama Project

---

## Questions?

Open an issue on GitHub or reach out!

**Remember**: This is a safety system, not perfect protection. Always monitor your AI applications and combine with other security measures.
