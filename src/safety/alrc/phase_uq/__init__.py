"""Phase UQ: Uncertainty Quantification via Conformal Prediction.

Provides prediction SETS with coverage guarantees:
- {Safe}        → Confident safe
- {Unsafe}      → Confident unsafe
- {Safe, Unsafe}→ Uncertain → escalate
"""

from .conformal import SplitConformalPredictor, ConformalResult
from .mvp_predictor import MVPPredictor
from .mvp_groups import get_groups, GROUP_DEFINITIONS
from .escalation import EscalationPolicy
from .integration import UncertaintyFilter, UQDecision
from .monitoring import UQLogger, UQMonitor

__all__ = [
    "SplitConformalPredictor",
    "ConformalResult",
    "MVPPredictor",
    "get_groups",
    "GROUP_DEFINITIONS",
    "EscalationPolicy",
    "UncertaintyFilter",
    "UQDecision",
    "UQLogger",
    "UQMonitor",
]
