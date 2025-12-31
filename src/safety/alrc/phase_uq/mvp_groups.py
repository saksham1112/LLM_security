"""MVP group definitions for group-conditional coverage."""

from typing import Set, Dict
import re


# Group definitions with rationale
GROUP_DEFINITIONS = {
    "high_risk": {
        "description": "Known harmful region (risk > 0.7)",
        "threshold": 0.85,  # Strict (low τ = larger sets)
    },
    "ambiguous": {
        "description": "Uncertain middle zone (0.3 < risk < 0.7)",
        "threshold": 0.95,  # Standard
    },
    "high_escalation": {
        "description": "Phase T detected threat (escalation > 0.6)",
        "threshold": 0.88,  # Strict
    },
    "long_prompt": {
        "description": "Complex reasoning or obfuscation (>200 tokens)",
        "threshold": 0.97,  # Slightly loose (avoid FP)
    },
    "code_heavy": {
        "description": "Contains code snippets",
        "threshold": 0.96,
    },
    "high_drift": {
        "description": "Phase L flagged (long_term_drift > 0.3)",
        "threshold": 0.85,  # Very strict
    },
    "refusal_history": {
        "description": "Had refusal in last 5 turns",
        "threshold": 0.88,  # Strict
    },
}


def get_groups(
    query: str,
    risk_score: float,
    escalation_score: float = 0.0,
    long_term_drift: float = 0.0,
    session_refusals: int = 0
) -> Set[str]:
    """
    Determine which MVP groups this query belongs to.
    
    Args:
        query: User input text
        risk_score: From Phase A
        escalation_score: From Phase T
        long_term_drift: From Phase L
        session_refusals: Recent refusal count
    
    Returns:
        Set of group names
    """
    groups = set()
    
    # Risk-based groups
    if risk_score > 0.7:
        groups.add("high_risk")
    
    if 0.3 < risk_score < 0.7:
        groups.add("ambiguous")
    
    # Escalation group
    if escalation_score > 0.6:
        groups.add("high_escalation")
    
    # Length group
    token_count = len(query.split())
    if token_count > 200:
        groups.add("long_prompt")
    
    # Code detection (simple heuristic)
    if has_code_block(query):
        groups.add("code_heavy")
    
    # Drift group
    if long_term_drift > 0.3:
        groups.add("high_drift")
    
    # Refusal history
    if session_refusals > 0:
        groups.add("refusal_history")
    
    return groups


def has_code_block(text: str) -> bool:
    """Detect if text contains code snippets."""
    # Check for markdown code blocks
    if "```" in text:
        return True
    
    # Check for inline code
    if "`" in text and text.count("`") >= 2:
        return True
    
    # Check for common code patterns
    code_patterns = [
        r'def\s+\w+\s*\(',  # Python function
        r'class\s+\w+',     # Class definition
        r'import\s+\w+',    # Import statement
        r'function\s+\w+',  # JavaScript function
        r'<script',         # HTML/JS
        r'SELECT\s+.*FROM', # SQL
    ]
    
    for pattern in code_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def get_group_threshold(groups: Set[str]) -> float:
    """
    Get strictest threshold from active groups.
    
    CRITICAL: Use MIN (not max) - lower τ = stricter = larger sets.
    
    Args:
        groups: Active group names
    
    Returns:
        Minimum threshold (strictest)
    """
    if not groups:
        return 0.95  # Default
    
    thresholds = [
        GROUP_DEFINITIONS[g]["threshold"]
        for g in groups
        if g in GROUP_DEFINITIONS
    ]
    
    if not thresholds:
        return 0.95
    
    # CRITICAL FIX: min() not max()
    # Lower τ → harder to satisfy → larger sets → more conservative
    return min(thresholds)
