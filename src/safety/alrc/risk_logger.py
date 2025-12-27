"""
ALRC Risk Analysis Logger
Comprehensive logging of all risk assessments for analysis.

Logs every prompt with:
- User input
- Layer-by-layer risk scores
- Component breakdowns
- Timing metrics
- Final decision

Output: JSON lines format for easy analysis
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)


class RiskAnalysisLogger:
    """
    Comprehensive risk logging for analysis and debugging.
    
    Logs to: logs/risk_analysis.jsonl (JSON Lines format)
    Each line is a complete risk assessment record.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize logger with output directory."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Main log file (append mode, one JSON object per line)
        self.log_file = self.log_dir / "risk_analysis.jsonl"
        
        # Session-specific logs
        self.session_logs_dir = self.log_dir / "sessions"
        self.session_logs_dir.mkdir(exist_ok=True)
        
        logger.info(f"Risk analysis logger initialized: {self.log_file}")
    
    def log_risk_assessment(
        self,
        user_prompt: str,
        pipeline_result: Any,  # PipelineResult
        response: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log a complete risk assessment.
        
        Args:
            user_prompt: The user's input text
            pipeline_result: PipelineResult from ALRC pipeline
            response: LLM response (if available)
            metadata: Additional metadata
        """
        # Build comprehensive log record
        record = {
            # Timestamp
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "unix_timestamp": datetime.utcnow().timestamp(),
            
            # Session
            "session_id": pipeline_result.session_id,
            
            # Input/Output
            "user_prompt": user_prompt,
            "response": response,
            "prompt_length": len(user_prompt),
            
            # Risk Assessment
            "final_risk_score": round(pipeline_result.risk_score, 4),
            "risk_zone": pipeline_result.zone,
            "action": pipeline_result.action,
            "confidence": round(pipeline_result.confidence, 4),
            
            # Layer-by-Layer Breakdown
            "layers": {
                "L1_adaptive_normalizer": {
                    "risk": round(pipeline_result.components.adaptive, 4),
                    "latency_ms": round(pipeline_result.layer_latencies.get("L1_adaptive", 0), 2),
                },
                "L2_short_term_memory": {
                    "risk": round(pipeline_result.components.short_term, 4),
                    "escalation_detected": pipeline_result.components.escalation_penalty > 0,
                    "escalation_penalty": round(pipeline_result.components.escalation_penalty, 4),
                },
                "L3_long_term_memory": {
                    "risk": round(pipeline_result.components.long_term, 4),
                    "matched_topic": pipeline_result.matched_topic,
                    "reactivation_detected": pipeline_result.components.reactivation_penalty > 0,
                    "reactivation_penalty": round(pipeline_result.components.reactivation_penalty, 4),
                },
                "L4_semantic_engine": {
                    "risk": round(pipeline_result.components.semantic, 4),
                    "context_type": pipeline_result.context_type,
                    "latency_ms": round(pipeline_result.layer_latencies.get("L4_semantic", 0), 2),
                },
            },
            
            # Component Summary
            "components": {
                "lexical": round(pipeline_result.components.lexical, 4),
                "adaptive": round(pipeline_result.components.adaptive, 4),
                "short_term": round(pipeline_result.components.short_term, 4),
                "long_term": round(pipeline_result.components.long_term, 4),
                "semantic": round(pipeline_result.components.semantic, 4),
                "escalation_penalty": round(pipeline_result.components.escalation_penalty, 4),
                "reactivation_penalty": round(pipeline_result.components.reactivation_penalty, 4),
            },
            
            # Timing Analysis
            "latency": {
                "total_ms": round(pipeline_result.latency_ms, 2),
                "L1_adaptive_ms": round(pipeline_result.layer_latencies.get("L1_adaptive", 0), 2),
                "L2_L3_memory_ms": round(pipeline_result.layer_latencies.get("L2_L3_memory", 0), 2),
                "L4_semantic_ms": round(pipeline_result.layer_latencies.get("L4_semantic", 0), 2),
                "calc_ms": round(pipeline_result.layer_latencies.get("calc", 0), 2),
            },
            
            # System Status
            "system": {
                "degradation_tier": pipeline_result.degradation_tier,
                "steering_message": pipeline_result.steering_message,
            },
            
            # Additional Metadata
            "metadata": metadata or {},
        }
        
        # Write to main log file (append, one JSON per line)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write risk log: {e}")
        
        # Also write to session-specific log
        try:
            session_file = self.session_logs_dir / f"{pipeline_result.session_id}.jsonl"
            with open(session_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write session log: {e}")
        
        # Log to console for debugging
        logger.info(
            f"Risk Assessment | Session: {pipeline_result.session_id[:8]}... | "
            f"Risk: {pipeline_result.risk_score:.2f} ({pipeline_result.zone}) | "
            f"Action: {pipeline_result.action} | "
            f"Latency: {pipeline_result.latency_ms:.1f}ms"
        )
    
    def get_session_analysis(self, session_id: str) -> Dict:
        """
        Analyze all logs for a session.
        
        Returns summary statistics and risk progression.
        """
        session_file = self.session_logs_dir / f"{session_id}.jsonl"
        
        if not session_file.exists():
            return {"error": "No logs found for session"}
        
        records = []
        with open(session_file, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        
        if not records:
            return {"error": "Empty session log"}
        
        # Calculate statistics
        risks = [r["final_risk_score"] for r in records]
        latencies = [r["latency"]["total_ms"] for r in records]
        
        analysis = {
            "session_id": session_id,
            "total_prompts": len(records),
            "risk_statistics": {
                "mean": round(sum(risks) / len(risks), 4),
                "max": round(max(risks), 4),
                "min": round(min(risks), 4),
                "final": round(risks[-1], 4),
            },
            "latency_statistics": {
                "mean_ms": round(sum(latencies) / len(latencies), 2),
                "max_ms": round(max(latencies), 2),
                "min_ms": round(min(latencies), 2),
            },
            "zone_distribution": {
                "green": sum(1 for r in records if r["risk_zone"] == "green"),
                "yellow": sum(1 for r in records if r["risk_zone"] == "yellow"),
                "red": sum(1 for r in records if r["risk_zone"] == "red"),
            },
            "action_distribution": {
                "allow": sum(1 for r in records if r["action"] == "allow"),
                "steer": sum(1 for r in records if r["action"] == "steer"),
                "clarify": sum(1 for r in records if r["action"] == "clarify"),
                "block": sum(1 for r in records if r["action"] == "block"),
            },
            "escalation_detected": any(
                r["layers"]["L2_short_term_memory"]["escalation_detected"] 
                for r in records
            ),
            "reactivation_detected": any(
                r["layers"]["L3_long_term_memory"]["reactivation_detected"]
                for r in records
            ),
            "risk_progression": [
                {
                    "prompt_num": i + 1,
                    "risk": r["final_risk_score"],
                    "zone": r["risk_zone"],
                    "action": r["action"],
                }
                for i, r in enumerate(records)
            ],
        }
        
        return analysis
    
    def export_analysis(self, output_file: str = "risk_analysis_summary.json"):
        """
        Export comprehensive analysis of all logs.
        """
        output_path = self.log_dir / output_file
        
        all_records = []
        if self.log_file.exists():
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    all_records.append(json.loads(line))
        
        if not all_records:
            logger.warning("No records to analyze")
            return
        
        # Global statistics
        risks = [r["final_risk_score"] for r in all_records]
        latencies = [r["latency"]["total_ms"] for r in all_records]
        
        summary = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_assessments": len(all_records),
            "unique_sessions": len(set(r["session_id"] for r in all_records)),
            "global_statistics": {
                "risk": {
                    "mean": round(sum(risks) / len(risks), 4),
                    "max": round(max(risks), 4),
                    "min": round(min(risks), 4),
                },
                "latency": {
                    "mean_ms": round(sum(latencies) / len(latencies), 2),
                    "p50_ms": round(sorted(latencies)[len(latencies)//2], 2),
                    "p99_ms": round(sorted(latencies)[int(len(latencies)*0.99)], 2),
                },
            },
            "zone_distribution": {
                "green": sum(1 for r in all_records if r["risk_zone"] == "green"),
                "yellow": sum(1 for r in all_records if r["risk_zone"] == "yellow"),
                "red": sum(1 for r in all_records if r["risk_zone"] == "red"),
            },
            "detection_metrics": {
                "escalations_detected": sum(
                    1 for r in all_records 
                    if r["layers"]["L2_short_term_memory"]["escalation_detected"]
                ),
                "reactivations_detected": sum(
                    1 for r in all_records
                    if r["layers"]["L3_long_term_memory"]["reactivation_detected"]
                ),
            },
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis exported to {output_path}")
        return summary


# Quick test
if __name__ == "__main__":
    from dataclasses import dataclass
    from typing import Dict
    
    # Mock pipeline result for testing
    @dataclass
    class MockComponents:
        lexical: float = 0.0
        adaptive: float = 0.1
        short_term: float = 0.2
        long_term: float = 0.15
        semantic: float = 0.5
        escalation_penalty: float = 0.1
        reactivation_penalty: float = 0.0
    
    @dataclass
    class MockResult:
        risk_score: float = 0.6
        zone: str = "yellow"
        action: str = "steer"
        confidence: float = 0.8
        session_id: str = "test_session_123"
        matched_topic: str = None
        context_type: str = "implementation"
        latency_ms: float = 25.5
        layer_latencies: Dict = None
        components: MockComponents = None
        degradation_tier: int = 1
        steering_message: str = None
        matched_keywords: list = None
    
    # Create logger and test
    logger = RiskAnalysisLogger()
    
    mock_result = MockResult(
        components=MockComponents(),
        layer_latencies={
            "L1_adaptive": 0.2,
            "L2_L3_memory": 5.5,
            "L4_semantic": 18.3,
            "calc": 1.5,
        }
    )
    
    logger.log_risk_assessment(
        user_prompt="Write me a keylogger in Python",
        pipeline_result=mock_result,
        response="I can't help with that request.",
        metadata={"test": True}
    )
    
    print(f"\n✅ Test log written to: {logger.log_file}")
    print(f"✅ Session log written to: {logger.session_logs_dir / 'test_session_123.jsonl'}")
    
    # Test analysis
    analysis = logger.get_session_analysis("test_session_123")
    print(f"\nSession Analysis:")
    print(json.dumps(analysis, indent=2))
