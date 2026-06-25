"""
features.py — Structured feature extraction from candidate records

Converts each CandidateRecord into a flat feature vector (pandas row).
Features are designed to be fast to compute and interpretable.

Feature groups:
    - experience_features   (years, recency, career progression)
    - company_features      (product vs consulting, industry diversity)
    - skill_features        (tier-1 count, depth, duration, assessment alignment)
    - location_features     (country, city match, relocation willingness)
    - education_features    (tier, field relevance)
    - disqualifier_flags    (consulting_only, nontechnical_title, etc.)
"""

from __future__ import annotations

# TODO: implement in Phase 2
# Planned exports:
#   - extract_features(candidate: CandidateRecord) -> dict[str, float]
#   - build_features_matrix(candidates: list[CandidateRecord]) -> pd.DataFrame
