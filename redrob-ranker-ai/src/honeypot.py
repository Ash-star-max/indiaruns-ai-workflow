"""
honeypot.py — Honeypot candidate detection

~80 candidates in the dataset have intentionally impossible profiles.
These are forced to relevance tier 0 in the ground truth.
Having >10 honeypots in the top 100 triggers Stage 3 disqualification.

Detection signals (each independently flagging):
    1. Temporal impossibility: career duration > plausible company age
    2. Expert skill with zero duration_months (≥3 such skills)
    3. Career history date conflicts (impossible overlaps)
    4. Salary range inversion (min > max)
    5. Years of experience vs graduation date anomaly
    6. Total career months >> stated years_of_experience * 12

Design principle: prefer false negatives over false positives.
Missing a honeypot hurts less than eliminating a real candidate.
"""

from __future__ import annotations

# TODO: implement in Phase 2
# Planned exports:
#   - detect_honeypot(candidate: CandidateRecord) -> bool
#   - honeypot_confidence(candidate: CandidateRecord) -> float   (0.0–1.0)
#   - build_honeypot_flags(candidates: list[CandidateRecord]) -> np.ndarray
