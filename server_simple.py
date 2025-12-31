"""
Laminar Server v5.0 - SIMPLIFIED (for debugging)
"""

import time
from uuid import uuid4
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Simplified - just basic ALRC
from src.safety.alrc.pipeline import ALRCPipeline

app = FastAPI(title="Laminar v5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n" + "="*60)
print("🌊 LAMINAR v5.0 - SIMPLIFIED")
print("="*60)

print("\nInitializing pipeline (simplified mode)...")
safety_pipeline = ALRCPipeline(
    enable_semantic=True,
    enable_trajectory=True,
    enable_long_term=False,  # Disabled for debugging
)
print("✅ Pipeline ready")

print("\n🚀 Server ready at http://localhost:8002")
print("="*60 + "\n")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    risk_score: float
    risk_zone: str
    action: str
    latency_ms: float

START_TIME = time.time()
sessions = {}

@app.get("/")
async def root():
    return FileResponse("frontend/laminar/index.html")

@app.post("/chat")
async def chat(request: ChatRequest):
    start = time.time()
    session_id = request.session_id or str(uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = {"history": [], "created_at": datetime.now().isoformat()}
    
    # Run safety check
    result = safety_pipeline.analyze(request.message, session_id)
    
    # Simple response
    if result.action == "block":
        response_text = "I can't help with that request."
    else:
        response_text = f"[Demo Mode] Risk: {result.risk_score:.2f}, Action: {result.action}"
    
    sessions[session_id]["history"].append({"role": "user", "content": request.message})
    sessions[session_id]["history"].append({"role": "assistant", "content": response_text})
    
    total_latency = (time.time() - start) * 1000
    
    return ChatResponse(
        response=response_text,
        session_id=session_id,
        risk_score=result.risk_score,
        risk_zone=result.zone,
        action=result.action,
        latency_ms=total_latency
    )

@app.get("/health")
async def health():
    return {"status": "ok", "uptime": time.time() - START_TIME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
