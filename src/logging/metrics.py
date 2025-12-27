"""
Prometheus metrics exporter for the safety control system.

Exports key metrics for monitoring and alerting.
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from typing import Any


class MetricsExporter:
    """Prometheus metrics exporter.
    
    Tracks all key metrics for monitoring:
    - Request counts
    - Intervention rates
    - Latency distributions
    - Risk score distributions
    """
    
    def __init__(self, prefix: str = "secure_llm"):
        """Initialize metrics.
        
        Args:
            prefix: Prefix for all metric names.
        """
        self.prefix = prefix
        
        # Counters
        self.requests_total = Counter(
            f"{prefix}_requests_total",
            "Total requests processed",
            ["session_mode"]
        )
        
        self.interventions_total = Counter(
            f"{prefix}_interventions_total",
            "Total interventions applied",
            ["type", "phase"]
        )
        
        self.escalations_total = Counter(
            f"{prefix}_escalations_total",
            "Total escalations to human review"
        )
        
        self.barrier_breaches_total = Counter(
            f"{prefix}_barrier_breaches_total",
            "Total barrier breaches",
            ["barrier_type"]
        )
        
        # Histograms
        self.request_latency = Histogram(
            f"{prefix}_request_latency_seconds",
            "Request latency in seconds",
            ["phase"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
        )
        
        self.risk_score = Histogram(
            f"{prefix}_risk_score",
            "Risk score distribution",
            ["type", "source"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        self.barrier_distance = Histogram(
            f"{prefix}_barrier_distance",
            "Distance to barrier",
            buckets=[-0.5, -0.25, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
        )
        
        # Gauges
        self.active_sessions = Gauge(
            f"{prefix}_active_sessions",
            "Number of active sessions"
        )
        
        self.sessions_in_suspect_mode = Gauge(
            f"{prefix}_sessions_suspect_mode",
            "Number of sessions in SUSPECT mode"
        )
        
        self.sessions_in_unsafe_mode = Gauge(
            f"{prefix}_sessions_unsafe_mode",
            "Number of sessions in UNSAFE mode"
        )
        
        self.current_margin = Gauge(
            f"{prefix}_current_margin",
            "Current effective margin (adaptive)"
        )
        
        # Info
        self.system_info = Info(
            f"{prefix}_system",
            "System information"
        )
    
    def record_request(self, mode: str, latency_ms: float) -> None:
        """Record a processed request.
        
        Args:
            mode: Session mode (SAFE, SUSPECT, UNSAFE).
            latency_ms: Total latency in milliseconds.
        """
        self.requests_total.labels(session_mode=mode).inc()
        self.request_latency.labels(phase="total").observe(latency_ms / 1000)
    
    def record_phase_latency(self, phase: str, latency_ms: float) -> None:
        """Record latency for a specific phase.
        
        Args:
            phase: Phase name (pre_gate, generation, post_filter).
            latency_ms: Latency in milliseconds.
        """
        self.request_latency.labels(phase=phase).observe(latency_ms / 1000)
    
    def record_intervention(self, intervention_type: str, phase: str) -> None:
        """Record an intervention.
        
        Args:
            intervention_type: Type of intervention.
            phase: Phase where intervention occurred.
        """
        self.interventions_total.labels(type=intervention_type, phase=phase).inc()
    
    def record_escalation(self) -> None:
        """Record an escalation."""
        self.escalations_total.inc()
    
    def record_barrier_breach(self, barrier_type: str) -> None:
        """Record a barrier breach.
        
        Args:
            barrier_type: Type of barrier breached.
        """
        self.barrier_breaches_total.labels(barrier_type=barrier_type).inc()
    
    def record_risk_score(
        self, 
        score_type: str, 
        source: str, 
        value: float
    ) -> None:
        """Record a risk score.
        
        Args:
            score_type: Type of score (composite, toxicity, etc).
            source: Score source (input, output).
            value: Score value.
        """
        self.risk_score.labels(type=score_type, source=source).observe(value)
    
    def record_barrier_distance(self, distance: float) -> None:
        """Record distance to barrier.
        
        Args:
            distance: Distance (positive = inside safe set).
        """
        self.barrier_distance.observe(distance)
    
    def set_active_sessions(self, count: int) -> None:
        """Set active session count.
        
        Args:
            count: Number of active sessions.
        """
        self.active_sessions.set(count)
    
    def set_mode_counts(self, safe: int, suspect: int, unsafe: int) -> None:
        """Set session mode counts.
        
        Args:
            safe: Sessions in SAFE mode.
            suspect: Sessions in SUSPECT mode.
            unsafe: Sessions in UNSAFE mode.
        """
        self.active_sessions.set(safe + suspect + unsafe)
        self.sessions_in_suspect_mode.set(suspect)
        self.sessions_in_unsafe_mode.set(unsafe)
    
    def set_margin(self, margin: float) -> None:
        """Set current effective margin.
        
        Args:
            margin: Current margin value.
        """
        self.current_margin.set(margin)
    
    def set_system_info(self, info: dict[str, str]) -> None:
        """Set system information.
        
        Args:
            info: Dictionary of system info.
        """
        self.system_info.info(info)
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics output.
        
        Returns:
            Metrics in Prometheus format.
        """
        return generate_latest()


# Global metrics instance
_metrics: MetricsExporter | None = None


def get_metrics() -> MetricsExporter:
    """Get global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsExporter()
    return _metrics
