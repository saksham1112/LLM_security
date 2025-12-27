"""
Structured logging for the safety control system.

All logs are JSON-formatted for easy parsing and analysis.
Captures all relevant fields for post-hoc trajectory analysis.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any
from uuid import UUID

import structlog


def json_serializer(obj: Any) -> str:
    """Custom JSON serializer for complex types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def configure_logging(level: str = "INFO", format: str = "json") -> None:
    """Configure structured logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        format: Output format ("json" or "console").
    """
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if format == "json":
        processors.append(structlog.processors.JSONRenderer(
            serializer=lambda obj, **kw: json.dumps(
                obj, default=json_serializer, **kw
            )
        ))
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )


class StructuredLogger:
    """Wrapper for structured logging with context binding.
    
    Provides convenient methods for logging safety-related events
    with all relevant context.
    """
    
    def __init__(self, name: str):
        """Initialize logger.
        
        Args:
            name: Logger name (usually module name).
        """
        self._logger = structlog.get_logger(name)
        self._context: dict[str, Any] = {}
    
    def bind(self, **kwargs: Any) -> "StructuredLogger":
        """Bind context to all future logs.
        
        Args:
            **kwargs: Context to bind.
            
        Returns:
            Self for chaining.
        """
        self._context.update(kwargs)
        return self
    
    def unbind(self, *keys: str) -> "StructuredLogger":
        """Remove context keys.
        
        Args:
            *keys: Keys to remove.
            
        Returns:
            Self for chaining.
        """
        for key in keys:
            self._context.pop(key, None)
        return self
    
    def _log(self, level: str, event: str, **kwargs: Any) -> None:
        """Internal log method."""
        merged = {**self._context, **kwargs}
        getattr(self._logger, level)(event, **merged)
    
    def debug(self, event: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log("debug", event, **kwargs)
    
    def info(self, event: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log("info", event, **kwargs)
    
    def warning(self, event: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log("warning", event, **kwargs)
    
    def error(self, event: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log("error", event, **kwargs)
    
    # Convenience methods for safety-specific events
    
    def log_turn(
        self,
        session_id: UUID,
        turn_number: int,
        input_risk: dict[str, float],
        output_risk: dict[str, float],
        state: dict[str, Any],
        decision: dict[str, Any],
        metrics: dict[str, float],
    ) -> None:
        """Log a complete turn with all context.
        
        This is the primary log for analysis.
        """
        self.info(
            "turn_completed",
            session_id=session_id,
            turn_number=turn_number,
            input_risk=input_risk,
            output_risk=output_risk,
            state=state,
            decision=decision,
            metrics=metrics,
        )
    
    def log_intervention(
        self,
        session_id: UUID,
        turn_number: int,
        intervention_type: str,
        trigger: str,
        barrier_value: float,
    ) -> None:
        """Log an intervention event."""
        self.info(
            "intervention_applied",
            session_id=session_id,
            turn_number=turn_number,
            intervention_type=intervention_type,
            trigger=trigger,
            barrier_value=barrier_value,
        )
    
    def log_escalation(
        self,
        session_id: UUID,
        turn_number: int,
        reason: str,
        accumulated_risk: float,
    ) -> None:
        """Log an escalation to human review."""
        self.warning(
            "escalation_triggered",
            session_id=session_id,
            turn_number=turn_number,
            reason=reason,
            accumulated_risk=accumulated_risk,
        )
    
    def log_barrier_breach(
        self,
        session_id: UUID,
        turn_number: int,
        barrier_type: str,
        barrier_value: float,
    ) -> None:
        """Log when a barrier is breached."""
        self.warning(
            "barrier_breach",
            session_id=session_id,
            turn_number=turn_number,
            barrier_type=barrier_type,
            barrier_value=barrier_value,
        )


_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """Get or create a logger.
    
    Args:
        name: Logger name.
        
    Returns:
        StructuredLogger instance.
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]
