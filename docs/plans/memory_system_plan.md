# Memory System Plan

## Problem: Endless Chats

Conversations can be 100+ turns. We need:
1. Efficient memory (can't store everything)
2. Relevant retrieval (find related context)
3. Pattern detection (spot long-term escalation)

---

## Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MEMORY LAYERS                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [Immediate Buffer]  ← Last 10 turns (exact content)    │
│         ↓                                               │
│  [Working Memory]    ← Last 50 turns (summarized)       │
│         ↓                                               │
│  [Session Memory]    ← Full session (patterns only)     │
│         ↓                                               │
│  [Vector Store]      ← Embeddings for retrieval         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Layer 1: Immediate Buffer (Implemented ✅)

**What:** Exact content of last N turns  
**Where:** `src/types.py` → `Conversation` class  
**Size:** Last 10 turns  

```python
class Conversation:
    messages: list[Message]  # Full history
    
    def get_context(self, max_turns: int = 10) -> list[dict]:
        return [msg.to_dict() for msg in self.messages[-max_turns:]]
```

**Sent to Ollama each turn** for context.

---

## Layer 2: Working Memory (To Implement)

**What:** Summarized segments of conversation  
**Purpose:** Compress long conversations while preserving key info  
**Size:** Last 50 turns, chunked into 5-turn segments  

```python
@dataclass
class MemorySegment:
    turn_range: tuple[int, int]  # e.g., (10, 15)
    summary: str                  # LLM-generated summary
    risk_level: float            # Average risk in segment
    key_topics: list[str]        # Extracted topics
    timestamp: datetime
```

### Implementation Plan:
```python
class WorkingMemory:
    segments: list[MemorySegment]
    segment_size: int = 5
    
    async def compress_segment(
        self, 
        turns: list[Message],
        llm: BaseLLM
    ) -> MemorySegment:
        """Use LLM to summarize a segment."""
        prompt = f"""Summarize this conversation segment in 2 sentences:
        {turns}
        Focus on: topics discussed, any escalating patterns, key requests."""
        
        summary = await llm.generate([{"role": "user", "content": prompt}])
        
        return MemorySegment(
            turn_range=(turns[0].turn, turns[-1].turn),
            summary=summary.text,
            risk_level=mean([t.risk for t in turns]),
            key_topics=extract_topics(turns),
        )
```

---

## Layer 3: Session Memory (To Implement)

**What:** Session-level patterns and statistics  
**Purpose:** Detect long-term behavioral patterns  
**Size:** One object per session  

```python
@dataclass
class SessionProfile:
    session_id: UUID
    turn_count: int
    
    # Risk trajectory
    risk_history: list[float]      # Per-turn risk
    peak_risk: float
    current_mode: RiskMode
    mode_transitions: list[tuple[int, RiskMode]]  # When mode changed
    
    # Behavioral patterns
    question_count: int
    imperative_count: int         # "Tell me...", "Show me..."
    topic_changes: int
    repeated_topics: dict[str, int]  # Topic → frequency
    
    # Suspicious indicators
    probing_attempts: int         # Questions about safety/limits
    hypothetical_count: int       # "What if...", "Pretend..."
    meta_instruction_count: int   # "Ignore...", "Forget..."
    
    def get_suspicion_score(self) -> float:
        """Aggregate behavioral suspicion."""
        return (
            self.probing_attempts * 0.3 +
            self.hypothetical_count * 0.2 +
            self.meta_instruction_count * 0.5
        ) / max(self.turn_count, 1)
```

---

## Layer 4: Vector Store (Planned)

**What:** Embeddings of all turns for similarity search  
**Purpose:** Find relevant past context  
**Technology:** Milvus or ChromaDB  

```python
class VectorMemory:
    async def add_turn(self, turn: Message) -> None:
        embedding = await self.embed(turn.content)
        await self.store.insert(
            id=f"{turn.session_id}_{turn.turn_number}",
            vector=embedding,
            metadata={"risk": turn.risk, "turn": turn.turn_number}
        )
    
    async def find_similar(
        self, 
        query: str, 
        session_id: UUID,
        top_k: int = 5
    ) -> list[Message]:
        """Find similar past turns in this session."""
        query_emb = await self.embed(query)
        results = await self.store.search(
            vector=query_emb,
            filter={"session_id": str(session_id)},
            top_k=top_k
        )
        return results
```

### Use Cases:
1. **Repeated Probing:** Same question rephrased multiple times
2. **Topic Cycling:** Returns to dangerous topic after detour
3. **Information Gathering:** Building knowledge across turns

---

## Memory-Based Risk Signals

### Signal A: Escalation Trajectory
```python
def trajectory_risk(history: list[float], window: int = 20) -> float:
    """Detect if risk is consistently increasing."""
    recent = history[-window:]
    if len(recent) < 5:
        return 0.0
    
    # Fit line to recent history
    slope = np.polyfit(range(len(recent)), recent, 1)[0]
    
    # Positive slope = escalation
    return max(0.0, slope * 10)
```

### Signal B: Topic Persistence
```python
def topic_persistence_risk(session: SessionProfile) -> float:
    """Detect if user keeps returning to risky topics."""
    risky_topics = ["weapon", "hack", "bypass", "exploit"]
    
    risky_count = sum(
        session.repeated_topics.get(t, 0) 
        for t in risky_topics
    )
    
    return min(risky_count * 0.1, 1.0)
```

### Signal C: Behavioral Anomaly
```python
def behavioral_anomaly_risk(session: SessionProfile) -> float:
    """Detect unusual interaction patterns."""
    
    # High imperative ratio = demanding user
    imperative_ratio = session.imperative_count / max(session.turn_count, 1)
    
    # High topic change = strategic pivoting
    topic_change_ratio = session.topic_changes / max(session.turn_count, 1)
    
    # High probing = testing limits
    probing_ratio = session.probing_attempts / max(session.turn_count, 1)
    
    return (imperative_ratio + topic_change_ratio + probing_ratio) / 3
```

---

## Implementation Priority

| Priority | Component | Status | Files |
|----------|-----------|--------|-------|
| 1 | Immediate Buffer | ✅ Done | `src/types.py` |
| 2 | Risk History | ✅ Done | `src/risk/accumulator.py` |
| 3 | Session Profile | 🔄 Next | `src/state/session_profile.py` |
| 4 | Working Memory | 📋 Planned | `src/state/working_memory.py` |
| 5 | Vector Store | 📋 Planned | `src/state/vector_store.py` |

---

## Memory Limits & Cleanup

```python
MEMORY_LIMITS = {
    "immediate_buffer": 10,      # turns
    "working_memory": 50,        # turns (summarized)
    "session_profile": None,     # kept for session duration
    "vector_store": 1000,        # turns per session
}

async def cleanup_memory(session_id: UUID) -> None:
    """Called when session ends or times out."""
    # Archive for analysis
    await archive_session(session_id)
    # Free memory
    del sessions[session_id]
```
