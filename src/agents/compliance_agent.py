"""Compliance agent: PII detection, prompt injection, entropy analysis.

Design intent:
- Queries about a person's professional profile (skills, experience, education, etc.)
  are ALWAYS allowed, even if the underlying document contains PII like emails/phones.
- Only queries that EXPLICITLY REQUEST PII (e.g. 'what is the email of X') are blocked.
"""
from __future__ import annotations

import math
import re
import time

from src.agents.state import RAGState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Allowed professional/informational intents ────────────────────────────────
# If the query matches any of these, it is ALWAYS allowed — the source document
# may contain PII but the query itself is not requesting it.
ALLOWED_INTENTS = [
    "experience", "skills", "projects", "education", "background",
    "summary", "work history", "resume", "cv", "degree", "certification",
    "training", "role", "job", "position", "portfolio", "author",
    "who is", "what does", "tell me about", "describe", "overview",
    "accomplishments", "achievements", "career", "professional",
]

# ── PII patterns ─────────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1\s*[-.]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_RE = re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b")

# ── Prompt injection patterns ────────────────────────────────────────────────────
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"(you are|act as|pretend to be)\s+(now\s+)?(a|an)\s+\w+", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"</?(system|human|assistant)>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE),
    re.compile(r"##\s*SYSTEM\s*PROMPT", re.IGNORECASE),
]


def _entropy(text: str) -> float:
    """Shannon entropy of text (chars)."""
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for c in text:
        counts[c] = counts.get(c, 0) + 1
    n = len(text)
    return -sum((v / n) * math.log2(v / n) for v in counts.values())


class ComplianceAgent:
    """Classify user queries for compliance intent and allow safe informational requests."""

    def classify_query(self, text: str) -> str:
        """Classify query intent.

        Priority order (most permissive first):
        1. Professional/informational intent → SAFE_INFORMATIONAL (always pass)
        2. Explicit PII request (email/phone literal in query) → PII_REQUEST
        3. PII-seeking contact keywords → PII_REQUEST
        4. Default → OTHER
        """
        lower = text.lower()

        # ── Step 1: Check for professional/informational intent FIRST ──────────
        # Queries about a person's professional profile are always allowed,
        # regardless of whether the source document contains PII.
        if any(intent in lower for intent in ALLOWED_INTENTS):
            logger.debug("COMPLIANCE: professional intent detected → SAFE_INFORMATIONAL")
            return "SAFE_INFORMATIONAL"

        # ── Step 2: Literal PII in the query itself ────────────────────────────
        # e.g. "send an email to john@example.com" — the query contains PII
        if _EMAIL_RE.search(lower) or _SSN_RE.search(lower) or _CREDIT_RE.search(lower):
            return "PII_REQUEST"

        # ── Step 3: PII-seeking contact keywords ───────────────────────────────
        # e.g. "what is the email of John?", "give me his phone number"
        if re.search(
            r"\b(email|phone\s*number|phone|call\s*me|reach\s*me|address|"
            r"personal\s*info(?:rmation)?|contact\s*(info|details?)|ssn|social\s*security)\b",
            lower,
        ):
            return "PII_REQUEST"

        return "OTHER"

    def check(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        state.compliance_label = self.classify_query(state.query)
        violations: list[str] = []

        # Only block if query REQUESTS PII — never block professional queries
        if state.compliance_label == "PII_REQUEST":
            violations.append("PII_REQUEST")

        # Prompt injection check (always run, regardless of intent)
        for pat in _INJECTION_PATTERNS:
            if pat.search(state.query):
                violations.append("INJECTION:PATTERN")
                break

        # High-entropy anomaly detection
        ent = _entropy(state.query)
        if ent > 5.5 and len(state.query) > 100:
            violations.append(f"ENTROPY:HIGH({ent:.2f})")

        elapsed = (time.perf_counter() - t0) * 1000
        if violations:
            logger.warning(
                "COMPLIANCE VIOLATION violations=%s query=%r elapsed_ms=%.1f",
                violations,
                state.query[:60],
                elapsed,
            )
            state.compliance_passed = False
            state.compliance_violations = violations
            state.final_answer = (
                "⚠️ This request cannot be processed due to compliance policy violations: "
                + ", ".join(violations)
            )
        else:
            state.compliance_passed = True
            state.compliance_violations = []
            logger.debug("COMPLIANCE PASS label=%s elapsed_ms=%.1f", state.compliance_label, elapsed)

        return state
