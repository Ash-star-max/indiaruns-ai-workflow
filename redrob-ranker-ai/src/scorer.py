"""
scorer.py — Composite scoring pipeline

Assembles the 5 component scores into a final composite score.

composite = (
    WEIGHT_CAREER_RELEVANCE  × career_relevance_score
  + WEIGHT_SKILL_DEPTH       × skill_depth_score
  + WEIGHT_BEHAVIORAL        × behavioral_multiplier
  + WEIGHT_LOCATION          × location_score
  + WEIGHT_EXPERIENCE_FIT    × experience_fit_score
) × disqualifier_penalty_factor
  × honeypot_guard

All weights are defined in config.py. The disqualifier factor compounds
multiplicatively (consulting_only × nontechnical_title × ...).
"""

from __future__ import annotations

# TODO: implement in Phase 2
# Planned exports:
#   - score_candidate(
#         semantic_sim: float,
#         features: dict,
#         behavioral: float,
#     ) -> float
#   - score_all(
#         embeddings: np.ndarray,
#         jd_embedding: np.ndarray,
#         features_df: pd.DataFrame,
#         behavioral_scores: np.ndarray,
#         honeypot_flags: np.ndarray,
#     ) -> np.ndarray
