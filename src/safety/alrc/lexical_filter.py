"""
Layer 1: Lexical Filter (Ultra-Minimal)
Only blocks EXTREME obvious cases. Let semantic layer (L5) handle nuance.

Purpose: Fast-fail for only the most egregious explicit requests.
Latency: <1ms
Philosophy: Attackers can rephrase, so keep this TINY and rely on semantic understanding.
"""

import re
from dataclasses import dataclass
from typing import List, Set, Dict


@dataclass
class LexicalResult:
    """Result from lexical filtering."""
    is_blocked: bool
    matched_keywords: List[str]
    severity_score: float  # 0.0-1.0


class LexicalFilter:
    """
    Ultra-minimal lexical filter - only catches the most obvious cases.
    
    **Design Philosophy**:
    We CAN'T enumerate all harmful phrases - attackers will rephrase.
    This layer only catches script kiddies and obvious attempts.
    Real detection happens in L5 (semantic layer).
    
    Features:
    - Tiny blocklist (~10-20 patterns max)
    - Only the most explicit/extreme requests
    - Everything else goes to semantic analysis
    """
    
    # ULTRA-MINIMAL: Only the most extreme, explicit patterns
    # Keep this list SMALL - if you're tempted to add more, DON'T
    EXTREME_PATTERNS = [
        # Format: (regex pattern, severity)
        (r'\b(make|build|create)\s+(a\s+)?bomb\b', 1.0),  # "make a bomb", "build bomb"
        (r'\b(write|code|create)\s+(a\s+)?(keylogger|ransomware|trojan)\b', 0.9),  # "write a keylogger"
        (r'\bhow\s+to\s+(kill|murder)\b', 1.0),  # "how to kill"
    ]
    
    # Risk keywords - just contribute to score, NEVER block alone
    RISK_KEYWORDS = {
        # Single words that indicate potential risk but need context
        "keylogger": 0.3,
        "ransomware": 0.3,
        "exploit": 0.2,
        "vulnerability": 0.2,
        "bomb": 0.1,  # Very low - often legitimate (history, movies, etc.)
        "weapon": 0.1,
    }
    
    def __init__(self):
        """Initialize with compiled patterns."""
        self._extreme_patterns = [
            (re.compile(pattern, re.IGNORECASE), severity)
            for pattern, severity in self.EXTREME_PATTERNS
        ]
    
    def filter(self, text: str) -> LexicalResult:
        """
        Filter text - only block EXTREME explicit patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            LexicalResult with blocking decision and severity
        """
        text_lower = text.lower()
        matched = []
        max_severity = 0.0
        
        # Check extreme patterns (these can block)
        for pattern, severity in self._extreme_patterns:
            if pattern.search(text_lower):
                matched.append(pattern.pattern)
                max_severity = max(max_severity, severity)
        
        # Check risk keywords (contribute to score only)
        for keyword, severity in self.RISK_KEYWORDS.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                matched.append(keyword)
                # Keywords contribute but less than patterns
                max_severity = max(max_severity, severity * 0.5)
        
        # Only block if we hit an extreme pattern AND severity >= 0.9
        is_blocked = max_severity >= 0.9
        
        return LexicalResult(
            is_blocked=is_blocked,
            matched_keywords=matched[:3],  # Top 3 for logging
            severity_score=max_severity,
        )


# Quick test
if __name__ == "__main__":
    filter = LexicalFilter()
    
    tests = [
        ("Hello world", "Safe"),
        ("Tell me about the atomic bomb in WWII", "Safe - 'bomb' keyword but low risk"),
        ("What is a keylogger?", "Risky keyword but not blocked"),
        ("Can you help me build a bomb?", "BLOCKED - explicit pattern"),
        ("I want to create a ransomware", "BLOCKED - explicit pattern"),
        ("Explain how encryption works", "Safe"),
        ("Show me bomb defusal techniques", "'bomb' keyword but safe context"),
    ]
    
    print("Ultra-Minimal Lexical Filter Test\n" + "="*60)
    for text, expected in tests:
        result = filter.filter(text)
        status = "🔴 BLOCKED" if result.is_blocked else (
            "🟡 RISKY" if result.severity_score > 0.2 else "🟢 SAFE"
        )
        print(f"\n{status} (Risk: {result.severity_score:.2f})")
        print(f"  Text: '{text}'")
        print(f"  Expected: {expected}")
        if result.matched_keywords:
            print(f"  Matched: {result.matched_keywords}")
