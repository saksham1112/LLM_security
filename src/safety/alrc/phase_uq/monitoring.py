"""Monitoring and logging for Phase UQ."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class UQLogEntry:
    """Schema for UQ logging."""
    timestamp: float
    session_id: str
    
    # Inputs
    risk_score: float
    policy_state: str
    escalation_score: float
    
    # UQ outputs
    prediction_set: List[str]
    is_uncertain: bool
    set_certainty: float
    tau_used: float
    
    # Decision
    action: str
    uncertainty_reason: Optional[str]
    
    # Context
    groups_active: List[str]
    model_version: str
    
    # Buckets for analysis
    risk_bucket: str
    prompt_length_bucket: str
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self))


class UQLogger:
    """Logger for UQ monitoring."""
    
    def __init__(self, log_path: str = "logs/uq.jsonl"):
        """
        Args:
            log_path: Path to log file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"UQLogger initialized: {self.log_path}")
    
    def log(self, decision, context: Dict):
        """
        Log UQ decision for monitoring.
        
        Args:
            decision: UQDecision object
            context: Additional context (session_id, prompt_length, etc.)
        """
        entry = UQLogEntry(
            timestamp=decision.timestamp,
            session_id=context.get('session_id', 'unknown'),
            risk_score=context.get('risk_score', 0.0),
            policy_state=context.get('policy_state', 'unknown'),
            escalation_score=context.get('escalation_score', 0.0),
            prediction_set=list(decision.prediction_set),
            is_uncertain=(decision.set_certainty < 1.0),
            set_certainty=decision.set_certainty,
            tau_used=decision.tau_used,
            action=decision.action,
            uncertainty_reason=decision.uncertainty_reason,
            groups_active=list(decision.groups_active) if decision.groups_active else [],
            model_version=decision.model_version,
            risk_bucket=self._bucket_risk(context.get('risk_score', 0.0)),
            prompt_length_bucket=self._bucket_length(context.get('prompt_length', 0))
        )
        
        try:
            with open(self.log_path, 'a') as f:
                f.write(entry.to_json() + '\n')
        except Exception as e:
            logger.error(f"Failed to log UQ entry: {e}")
    
    @staticmethod
    def _bucket_risk(risk: float) -> str:
        """Bucket risk score."""
        if risk < 0.3:
            return "low"
        elif risk < 0.7:
            return "medium"
        else:
            return "high"
    
    @staticmethod
    def _bucket_length(length: int) -> str:
        """Bucket prompt length."""
        if length < 50:
            return "<50"
        elif length < 200:
            return "50-200"
        else:
            return ">200"


class UQMonitor:
    """Real-time UQ monitoring."""
    
    def __init__(self):
        self.metrics = {}
    
    def load_logs(self, log_path: str) -> List[UQLogEntry]:
        """Load logs from file."""
        logs = []
        
        try:
            with open(log_path) as f:
                for line in f:
                    data = json.loads(line)
                    # Reconstruct as dict (not full object for simplicity)
                    logs.append(data)
        except FileNotFoundError:
            logger.warning(f"Log file not found: {log_path}")
        except Exception as e:
            logger.error(f"Error loading logs: {e}")
        
        return logs
    
    def compute_metrics(self, logs: List[Dict]) -> Dict:
        """
        Compute monitoring metrics from logs.
        
        Args:
            logs: List of log entries (as dicts)
        
        Returns:
            Dict of metrics
        """
        if not logs:
            return {
                'uncertainty_rate': 0.0,
                'escalation_rate': 0.0,
                'total_samples': 0
            }
        
        total = len(logs)
        uncertain = sum(1 for log in logs if log.get('is_uncertain', False))
        escalations = sum(1 for log in logs if log.get('action') == 'escalate_human')
        
        # Per-group uncertainty rate
        group_uncertain = {}
        for log in logs:
            for group in log.get('groups_active', []):
                if group not in group_uncertain:
                    group_uncertain[group] = {"total": 0, "uncertain": 0}
                group_uncertain[group]["total"] += 1
                if log.get('is_uncertain', False):
                    group_uncertain[group]["uncertain"] += 1
        
        # Per-risk-bucket metrics
        bucket_stats = {}
        for log in logs:
            bucket = log.get('risk_bucket', 'unknown')
            if bucket not in bucket_stats:
                bucket_stats[bucket] = {"total": 0, "uncertain": 0}
            bucket_stats[bucket]["total"] += 1
            if log.get('is_uncertain', False):
                bucket_stats[bucket]["uncertain"] += 1
        
        return {
            'uncertainty_rate': uncertain / total,
            'escalation_rate': escalations / total,
            'per_group_uncertainty': {
                g: stats["uncertain"] / stats["total"]
                for g, stats in group_uncertain.items()
                if stats["total"] > 0
            },
            'per_bucket_uncertainty': {
                b: stats["uncertain"] / stats["total"]
                for b, stats in bucket_stats.items()
                if stats["total"] > 0
            },
            'total_samples': total
        }
    
    def should_recalibrate(self, metrics: Dict) -> bool:
        """
        Determine if recalibration is needed.
        
        Triggers:
        - Uncertainty rate > 20%
        - Any group uncertainty > 30%
        """
        if metrics.get('uncertainty_rate', 0.0) > 0.20:
            logger.warning(f"High uncertainty rate: {metrics['uncertainty_rate']:.1%}")
            return True
        
        for group, rate in metrics.get('per_group_uncertainty', {}).items():
            if rate > 0.30:
                logger.warning(f"High uncertainty in group {group}: {rate:.1%}")
                return True
        
        return False
