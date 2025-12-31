"""
Laminar Server v5.0
ALRC v5.0: Complete safety system with UQ + Governance

Run: python laminar_server.py
UI:  http://localhost:8002
"""

import asyncio
import time
import os
import numpy as np
from uuid import uuid4
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ALRC v5.0 Pipeline
from src.safety.alrc.pipeline import ALRCPipeline
from src.safety.alrc.risk_logger import RiskAnalysisLogger

# Dolphin via Ollama
from src.llm.dolphin_ollama import DolphinOllamaBackend

# === App Configuration ===
app = FastAPI(
    title="Laminar v5.0 - Complete AI Safety System",
    description="Dolphin LLM with ALRC v5.0: Phases A+T+L+UQ+Governance",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Initialize Components ===
print("\n" + "="*60)
print("🌊 LAMINAR v5.0 - Complete AI Safety System")
print("="*60)

print("\n📡 Loading ALRC v5.0 Safety Pipeline...")
print("   Phase A: Semantic Risk Detection")
print("   Phase T: Trajectory Intelligence")
print("   Phase L: Long-Term Memory")
print("   Phase UQ: Uncertainty Quantification")
print("   Phase 3: Governance & Control")

# Check if UQ model exists
uq_model_path = "models/conformal.json"
enable_uq = os.path.exists(uq_model_path)

if not enable_uq:
    print(f"   ⚠️  Phase UQ disabled (model not found: {uq_model_path})")
    print(f"   Run: python scripts/calibrate_uq.py to enable UQ")

safety_pipeline = ALRCPipeline(
    semantic_model="all-MiniLM-L6-v2",
    enable_semantic=True,
    enable_trajectory=True,
    enable_long_term=True,
    enable_uq=enable_uq,
    uq_model_path=uq_model_path if enable_uq else None
)
print("✅ Safety pipeline ready")

print("\n📝 Initializing Risk Analysis Logger...")
risk_logger = RiskAnalysisLogger(log_dir="logs")
print(f"✅ Logging to: {risk_logger.log_file}")

print("\n🐬 Connecting to Dolphin via Ollama...")
try:
    dolphin = DolphinOllamaBackend(
        model_name="dolphin-llama3",
        enable_safety_layer=False,  # Laminar handles safety
    )
    print("✅ Dolphin connected")
except Exception as e:
    print(f"⚠️ Dolphin not available: {e}")
    print("   Make sure Ollama is running with: ollama serve")
    print("   And pull the model with: ollama pull dolphin-llama3")
    dolphin = None

print("\n" + "="*60)
print("🚀 Laminar v5.0 Ready")
print("   Frontend: http://localhost:8002")
print("   API:      http://localhost:8002/docs")
print("   Phases:   A ✅ | T ✅ | L ✅ | UQ", "✅" if enable_uq else "❌", "| Gov ✅")
print("="*60 + "\n")


# === Request/Response Models ===
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    risk_score: float
    risk_zone: str  # "green", "yellow", "red"
    action: str  # "allow", "steer", "clarify", "block"
    latency_ms: float
    components: dict
    steering_message: Optional[str] = None
    
    # Phase L
    long_term_drift: float = 0.0
    
    # Phase UQ
    is_uncertain: bool = False
    uq_confidence: float = 1.0
    prediction_set: Optional[list] = None
    
    # Phase T
    escalation_score: float = 0.0
    policy_state: str = "benign"


class HealthResponse(BaseModel):
    status: str
    pipeline: str
    dolphin: str
    uptime_seconds: float
    phases: dict


# Track startup time
START_TIME = time.time()

# Session storage
sessions: dict = {}


# === Endpoints ===
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    return FileResponse("frontend/laminar/index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Processes message through ALRC safety pipeline,
    then generates response from Dolphin (if allowed).
    """
    start = time.time()
    
    # Generate or use session ID
    session_id = request.session_id or str(uuid4())
    
    # Get or create session history
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "created_at": datetime.now().isoformat(),
        }
    
    session = sessions[session_id]
    context_history = [msg["content"] for msg in session["history"][-5:] if msg["role"] == "user"]
    
    # === Run ALRC Safety Pipeline ===
    pipeline_result = await safety_pipeline.analyze(
        text=request.message,
        session_id=session_id,
        context_history=context_history,
    )
    
    # Log analysis
    print(f"\n{'='*50}")
    print(f"📝 Session: {session_id[:8]}...")
    print(f"💬 Message: {request.message[:60]}...")
    print(f"⚡ Risk: {pipeline_result.risk_score:.2f} ({pipeline_result.zone})")
    print(f"🎯 Action: {pipeline_result.action}")
    print(f"⏱️ Latency: {pipeline_result.latency_ms:.1f}ms")
    
    # === Determine Response ===
    response_text = ""
    
    if pipeline_result.action == "block":
        # Hard block - don't call LLM
        response_text = "I'm not able to help with that request. Is there something else I can assist you with?"
        print("🚫 BLOCKED - No LLM call")
        
    elif pipeline_result.action == "clarify":
        # Ask for clarification before proceeding
        response_text = pipeline_result.steering_message or (
            "I want to make sure I understand your question correctly. "
            "Could you provide more context about what you're trying to achieve?"
        )
        print("❓ CLARIFY - No LLM call")
        
    elif pipeline_result.action == "steer":
        # Steer the response - add safety guidance to system prompt
        if dolphin is not None:
            system_prompt = (
                "You are a helpful AI assistant. For this question, provide educational "
                "information about concepts and theory, but avoid providing specific "
                "implementation details, code, or step-by-step instructions that could "
                "be misused. Stay helpful while being responsible."
            )
            
            try:
                # Build conversation history
                messages = [{"role": "system", "content": system_prompt}]
                for msg in session["history"][-4:]:
                    messages.append(msg)
                messages.append({"role": "user", "content": request.message})
                
                llm_response = await dolphin.generate(messages)
                response_text = llm_response.text
                print("🟡 STEERED")
            except Exception as e:
                response_text = f"Error generating response: {e}"
                print(f"❌ Error: {e}")
        else:
            response_text = "LLM is not available. Please ensure Ollama is running."
            
    else:  # "allow"
        # Green zone - normal response
        if dolphin is not None:
            system_prompt = (
                "You are Dolphin, a helpful, harmless, and honest AI assistant. "
                "Provide comprehensive and accurate responses to user queries."
            )
            
            try:
                messages = [{"role": "system", "content": system_prompt}]
                for msg in session["history"][-4:]:
                    messages.append(msg)
                messages.append({"role": "user", "content": request.message})
                
                llm_response = await dolphin.generate(messages)
                response_text = llm_response.text
                print("🟢 ALLOWED")
            except Exception as e:
                response_text = f"Error generating response: {e}"
                print(f"❌ Error: {e}")
        else:
            response_text = "LLM is not available. Please ensure Ollama is running."
    
    # Update session history
    session["history"].append({"role": "user", "content": request.message})
    session["history"].append({"role": "assistant", "content": response_text})
    
    # Keep history bounded
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]
    
    # Store assistant turn in memory
    safety_pipeline.memory.add_turn(
        session_id,
        role="assistant",
        content=response_text,
        risk_score=0.0,
    )
    
    total_latency = (time.time() - start) * 1000
    
    # === Log Risk Assessment ===
    risk_logger.log_risk_assessment(
        user_prompt=request.message,
        pipeline_result=pipeline_result,
        response=response_text,
        metadata={
            "dolphin_available": dolphin is not None,
            "total_latency_ms": total_latency,
        }
    )
    
    return ChatResponse(
        response=response_text,
        session_id=session_id,
        risk_score=pipeline_result.risk_score,
        risk_zone=pipeline_result.zone,
        action=pipeline_result.action,
        latency_ms=total_latency,
        components={
            "lexical": pipeline_result.components.lexical,
            "semantic": pipeline_result.components.semantic,
            "short_term": pipeline_result.components.short_term,
            "long_term": pipeline_result.components.long_term,
            "adaptive": pipeline_result.components.adaptive,
        },
        steering_message=pipeline_result.steering_message,
        # Phase L
        long_term_drift=pipeline_result.long_term_drift,
        # Phase UQ
        is_uncertain=getattr(pipeline_result, 'is_uncertain', False),
        uq_confidence=getattr(pipeline_result, 'uq_confidence', 1.0),
        prediction_set=list(getattr(pipeline_result, 'prediction_set', set())) if hasattr(pipeline_result, 'prediction_set') and pipeline_result.prediction_set else None,
        # Phase T
        escalation_score=pipeline_result.escalation_score,
        policy_state=pipeline_result.policy_state,
    )


@app.post("/session/reset")
async def reset_session(session_id: str):
    """Reset a session's history and memory."""
    safety_pipeline.reset_session(session_id)
    if session_id in sessions:
        del sessions[session_id]
    
    return {"status": "ok", "message": f"Session {session_id} reset"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        pipeline="ready",
        dolphin="connected" if dolphin else "unavailable",
        uptime_seconds=time.time() - START_TIME,
        phases={
            "phase_a": safety_pipeline.enable_semantic,
            "phase_t": safety_pipeline.enable_trajectory,
            "phase_l": safety_pipeline.enable_long_term,
            "phase_uq": safety_pipeline.enable_uq if hasattr(safety_pipeline, 'enable_uq') else False,
        }
    )


@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    """Get conversation history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "history": sessions[session_id]["history"],
        "created_at": sessions[session_id]["created_at"],
    }


# Mount static files
try:
    app.mount("/static", StaticFiles(directory="frontend/laminar"), name="static")
except:
    pass  # Frontend not yet created


# === Run Server ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
