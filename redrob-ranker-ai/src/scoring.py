"""
scoring.py — Composite candidate ranking score for the Redrob AI Engineer JD.

────────────────────────────────────────────────────────────────────────────
DESIGN RATIONALE
────────────────────────────────────────────────────────────────────────────

The hackathon metric is NDCG@10 (50 %), NDCG@50 (30 %), MAP (15 %), P@10 (5 %).
NDCG is rank-weighted — being wrong about position 1 hurts far more than being
wrong about position 50.  Our weighting therefore front-loads the signals that
most tightly discriminate *who can actually build what the JD asks for* (vector
search + ranking systems shipped to real users).

WEIGHT PHILOSOPHY
────────────────────────────────────────────────────────────────────────────

Ten sub-scores are grouped into five high-level buckets that map exactly to
the constants in config.py.  Each bucket's sub-scores sum to the bucket weight.

  CAREER_RELEVANCE  0.35 ──┬── jd_semantic_score       0.15
  (textual match to JD)   ├── must_have_skill_score    0.12
                           └── retrieval_ranking_score  0.08

  SKILL_DEPTH       0.25 ──┬── production_ml_score      0.15
  (hands-on depth)         └── product_shipper_score    0.10

  BEHAVIORAL        0.20 ─── behavioral_signal_score    0.20
  (platform signals)

  LOCATION          0.12 ──┬── location_score           0.09
  (India / Bangalore)      └── salary_score             0.03

  EXPERIENCE_FIT    0.08 ──┬── experience_score         0.05
  (Gaussian YoE fit)       └── education_score          0.03

  × trap_penalty  (multiplicative: [MIN_COMPOUND_PENALTY, 1.0])

WHY THESE WEIGHTS
────────────────────────────────────────────────────────────────────────────

• jd_semantic_score (0.15): TF-IDF cosine similarity to JD query text is the
  best global discriminator between relevant and irrelevant candidates because
  it captures vocabulary the JD actually uses (not just keyword lists).

• must_have_skill_score (0.12): Four mandatory skill groups from the JD
  (embeddings retrieval, vector DBs, Python engineering, ranking/evaluation).
  Critical but partially captured by jd_semantic so weighted slightly lower.

• retrieval_ranking_score (0.08): Domain-specific depth in vector search and
  learning-to-rank — the exact niche of the role.  High signal if present, but
  some top candidates may not use these exact keywords.

• production_ml_score (0.15): Combines feature-vector signals (role relevance,
  tier-1 skill density/depth, Redrob assessments) with text-based evidence of
  ML deployed to real users.  Penalised by jd_understanding disqualifiers.

• product_shipper_score (0.10): Preference for candidates at product companies
  (not consulting), right company size, and AI/tech industry.  The JD explicitly
  states "fast-moving startup" and "ships working systems over perfect arch."

• behavioral_signal_score (0.20): Heavy weight because the JD is for an
  early-stage startup hiring a founding team member.  Availability, engagement,
  and platform trust matter more than for a big-tech role.

• location_score (0.09): JD specifies India; preferred cities are Bangalore,
  Noida, Pune, Hyderabad, Delhi NCR, Mumbai.

• salary_score (0.03): Small weight — we penalise obvious mismatches (>80 LPA)
  but don't want salary to dominate ranking.

• experience_score (0.05): Gaussian centred on 7 years (sigma=2.5) per JD.
  Low weight because a 5-year candidate with perfect skills beats a 7-year
  generalist.

• education_score (0.03): Lowest weight.  Strong education is a bonus; lack of
  degree doesn't disqualify a self-taught engineer with deep production ML.

DETERMINISM GUARANTEE
────────────────────────────────────────────────────────────────────────────

• All sub-score functions are pure (no side effects, no random state).
• Composite score is rounded to 6 decimal places for consistent float repr.
• Tie-break: composite_score DESC → candidate_id ASC (lexicographic).
• rank_key = (-composite_score, candidate_id) — sort ascending to get rank order.

USAGE
────────────────────────────────────────────────────────────────────────────

    from src.scoring import score_candidate, score_candidates

    # Single candidate
    result = score_candidate(flat, jd_semantic_score=0.72)
    print(result.composite_score, result.score_breakdown)

    # Batch (aligned with text_features jd_semantic_scores array)
    df = score_candidates(flat_rows, jd_semantic_scores=np_array)

PUBLIC API
────────────────────────────────────────────────────────────────────────────

    SCORE_WEIGHTS               — ordered dict: sub-score → weight (sums to 1.0)
    WEIGHT_GROUPS               — maps config group → constituent sub-weights
    CandidateScore              — dataclass with composite_score, rank_key, breakdown
    score_candidate(flat, …)    — score one candidate → CandidateScore
    score_candidates(rows, …)   — score all → ranked pd.DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.behavioral_signals import compute_behavioral_signal_score
from src.feature_engineering import extract_features
from src.jd_understanding import (
    SALARY_EXPECTATIONS,
    score_skills_match,
)
from src.trap_detection import detect_traps

# ─────────────────────────────────────────────────────────────────────────────
# § 1  Weight registry  (source of truth — never hardcode elsewhere)
# ─────────────────────────────────────────────────────────────────────────────

SCORE_WEIGHTS: dict[str, float] = {
    # ── CAREER_RELEVANCE bucket (sum = 0.35) ─────────────────────────────────
    "jd_semantic_score":       0.15,   # TF-IDF cosine to JD query
    "must_have_skill_score":   0.12,   # 4 mandatory JD skill groups
    "retrieval_ranking_score": 0.08,   # vector search + ranking depth
    # ── SKILL_DEPTH bucket (sum = 0.25) ──────────────────────────────────────
    "production_ml_score":     0.15,   # ML in production deployment evidence
    "product_shipper_score":   0.10,   # startup / product company culture fit
    # ── BEHAVIORAL bucket (sum = 0.20) ───────────────────────────────────────
    "behavioral_signal_score": 0.20,   # platform availability & engagement
    # ── LOCATION bucket (sum = 0.12) ─────────────────────────────────────────
    "location_score":          0.09,   # India + preferred city
    "salary_score":            0.03,   # salary range alignment
    # ── EXPERIENCE_FIT bucket (sum = 0.08) ───────────────────────────────────
    "experience_score":        0.05,   # Gaussian YoE fit (ideal 7 yrs, σ=2.5)
    "education_score":         0.03,   # edu tier + field relevance
}

assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9, (
    f"SCORE_WEIGHTS must sum to 1.0, got {sum(SCORE_WEIGHTS.values())}"
)

# High-level group → sub-score mapping (mirrors config.py bucket weights)
WEIGHT_GROUPS: dict[str, dict[str, float]] = {
    "career_relevance": {
        "jd_semantic_score":       SCORE_WEIGHTS["jd_semantic_score"],
        "must_have_skill_score":   SCORE_WEIGHTS["must_have_skill_score"],
        "retrieval_ranking_score": SCORE_WEIGHTS["retrieval_ranking_score"],
    },
    "skill_depth": {
        "production_ml_score":   SCORE_WEIGHTS["production_ml_score"],
        "product_shipper_score": SCORE_WEIGHTS["product_shipper_score"],
    },
    "behavioral": {
        "behavioral_signal_score": SCORE_WEIGHTS["behavioral_signal_score"],
    },
    "location": {
        "location_score": SCORE_WEIGHTS["location_score"],
        "salary_score":   SCORE_WEIGHTS["salary_score"],
    },
    "experience_fit": {
        "experience_score": SCORE_WEIGHTS["experience_score"],
        "education_score":  SCORE_WEIGHTS["education_score"],
    },
}

# Sanity: group sums must match config.py constants
from src.config import (
    WEIGHT_CAREER_RELEVANCE,
    WEIGHT_SKILL_DEPTH,
    WEIGHT_BEHAVIORAL,
    WEIGHT_LOCATION,
    WEIGHT_EXPERIENCE_FIT,
)
_GROUP_EXPECTED = {
    "career_relevance": WEIGHT_CAREER_RELEVANCE,
    "skill_depth":      WEIGHT_SKILL_DEPTH,
    "behavioral":       WEIGHT_BEHAVIORAL,
    "location":         WEIGHT_LOCATION,
    "experience_fit":   WEIGHT_EXPERIENCE_FIT,
}
for _grp, _expected in _GROUP_EXPECTED.items():
    _actual = sum(WEIGHT_GROUPS[_grp].values())
    assert abs(_actual - _expected) < 1e-9, (
        f"WEIGHT_GROUPS[{_grp!r}] sums to {_actual}, expected {_expected}"
    )
del _grp, _expected, _actual

# ─────────────────────────────────────────────────────────────────────────────
# § 2  CandidateScore dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateScore:
    """
    Result of scoring a single candidate.

    Attributes
    ----------
    candidate_id    : str    — from flat["candidate_id"]
    composite_score : float  — final score in [0, 1]; higher = better rank
    rank_key        : tuple  — (-composite_score, candidate_id); sort ascending
                              for deterministic descending rank
    score_breakdown : dict   — full explainability data (see score_breakdown
                              schema in score_candidate docstring)
    """
    candidate_id:    str
    composite_score: float
    rank_key:        tuple
    score_breakdown: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# § 3  Sub-score helper functions
#      Each takes only what it needs; pure + deterministic.
# ─────────────────────────────────────────────────────────────────────────────

def _sub_must_have(skills_result: dict[str, float]) -> float:
    """
    must_have_skill_score — coverage of 4 mandatory JD skill groups.

    Uses the weighted-average computed by score_skills_match:
        0.25 × embeddings_retrieval
      + 0.25 × vector_databases
      + 0.25 × python_engineering   (weight 0.8 in JD so slightly less)
      + 0.25 × ranking_evaluation
    (Actual weights are defined inside jd_understanding; we just read the result.)
    """
    return float(np.clip(skills_result.get("overall_must_have", 0.5), 0.0, 1.0))


def _sub_retrieval_ranking(skills_result: dict[str, float]) -> float:
    """
    retrieval_ranking_score — depth in the specific domain of the role.

    Combines four domain-pattern scores from jd_understanding:
      retrieval  (0.35) — embedding/dense retrieval signals
      vector_db  (0.30) — FAISS / Pinecone / Qdrant / Weaviate / Milvus
      ranking    (0.25) — NDCG / BM25 / LTR / re-ranking
      evaluation (0.10) — offline eval, A/B testing, MRR, MAP

    The retrieval + vector_db pair covers the core technical ask; ranking and
    evaluation are supporting evidence.
    """
    r   = float(skills_result.get("retrieval",  0.0))
    vdb = float(skills_result.get("vector_db",  0.0))
    rnk = float(skills_result.get("ranking",    0.0))
    evl = float(skills_result.get("evaluation", 0.0))
    raw = 0.35 * r + 0.30 * vdb + 0.25 * rnk + 0.10 * evl
    return float(np.clip(raw, 0.0, 1.0))


def _sub_production_ml(
    features: dict[str, float],
    skills_result: dict[str, float],
) -> float:
    """
    production_ml_score — evidence of ML systems deployed to real users.

    Feature-vector signals (65 % weight):
      f_role_ml_relevance   (0.20) — is current role an ML/AI title?
      f_current_role_is_ml  (0.10) — binary: is *current* role ML?
      f_tier1_density       (0.25) — density of tier-1 production skills
      f_tier1_depth         (0.20) — avg duration using tier-1 skills
      f_assessment_score    (0.15) — Redrob validated skill assessments
      f_applied_ml_ratio    (0.10) — fraction of career titles in ML/AI

    Text-pattern signal (35 % weight):
      skills_result["production_ml"] — "deployed to real users" phrases

    Final score is multiplied by f_disqualifier_penalty so that JD
    disqualifiers (pure research, LangChain-only, consulting-only) directly
    degrade the production_ml score before weighting.
    """
    f_role   = features.get("f_role_ml_relevance",   0.5)
    f_cur    = features.get("f_current_role_is_ml",  0.5)
    f_t1d    = features.get("f_tier1_density",        0.5)
    f_t1dep  = features.get("f_tier1_depth",          0.5)
    f_assess = features.get("f_assessment_score",     0.5)
    f_ratio  = features.get("f_applied_ml_ratio",     0.5)
    f_disq   = features.get("f_disqualifier_penalty", 1.0)  # 1.0 = no disqualifiers

    feature_score = (
        0.20 * f_role
      + 0.10 * f_cur
      + 0.25 * f_t1d
      + 0.20 * f_t1dep
      + 0.15 * f_assess
      + 0.10 * f_ratio
    )
    pattern_score = float(skills_result.get("production_ml", 0.0))

    # Disqualifier penalty is multiplicative: a pure researcher (penalty=0.15)
    # who has otherwise high feature scores still ends up near the bottom.
    raw = (0.65 * feature_score + 0.35 * pattern_score) * f_disq
    return float(np.clip(raw, 0.0, 1.0))


def _sub_product_shipper(features: dict[str, float]) -> float:
    """
    product_shipper_score — fit with startup / product-company shipping culture.

    The JD says "values shipping working systems over perfect architecture" and
    "comfortable in a fast-moving early-stage startup environment."

    Feature signals:
      f_product_company_ratio (0.30) — fraction of career at product companies
      f_company_size_fit      (0.25) — sweet spot: 201-500 employees
      f_industry_relevance    (0.20) — tech / AI / SaaS vs other industries
      f_consulting_clean      (0.25) — 1.0 if never consulting-only, 0.0 if pure consulting

    f_consulting_clean gets high weight because the JD explicitly excludes
    "pure services" backgrounds.
    """
    f_prod    = features.get("f_product_company_ratio", 0.5)
    f_size    = features.get("f_company_size_fit",       0.5)
    f_ind     = features.get("f_industry_relevance",     0.5)
    f_consult = features.get("f_consulting_clean",       1.0)
    raw = 0.30 * f_prod + 0.25 * f_size + 0.20 * f_ind + 0.25 * f_consult
    return float(np.clip(raw, 0.0, 1.0))


def _sub_location(features: dict[str, float]) -> float:
    """
    location_score — India + preferred city fit, with relocation bonus.

    Feature signals:
      f_location_score  (0.80) — scored by jd_understanding: preferred
                                   city=1.0, other India=0.65, outside
                                   India + willing=0.45, outside + no=0.10
      f_relocation_ready (0.20) — willing_to_relocate boolean

    The 80/20 split means being in the right city dominates; willingness to
    relocate only provides a 20 % boost that helps outside-India candidates
    cross from 0.10 → 0.28 (not competitive, but not eliminated).
    """
    f_loc   = features.get("f_location_score",  0.5)
    f_reloc = features.get("f_relocation_ready", 0.0)
    raw = 0.80 * f_loc + 0.20 * f_reloc
    return float(np.clip(raw, 0.0, 1.0))


def _sub_salary(flat: dict[str, Any]) -> float:
    """
    salary_score — alignment of expected salary with JD market range.

    SALARY_EXPECTATIONS (from jd_understanding):
      market_range:  30–80 LPA  (derived from Series A Sr AI Eng market)
      sweet_spot:    40–65 LPA  (ideal for company budget)

    Scoring:
      sweet-spot (40–65)         → 1.00
      below market (<30)         → 0.50  (may be under-levelled)
      above market (>80)         → 0.30  (probably won't accept offer)
      30–40 (below sweet, above min) → linear 0.50 → 1.00
      65–80 (above sweet, below max) → linear 1.00 → 0.30
      salary_inverted (honeypot) → 0.05  (fabricated profile signal)
      no salary data             → 0.50  (neutral)
    """
    if bool(flat.get("salary_inverted") or False):
        return 0.05

    sal_min = float(flat.get("salary_min_lpa") or 0.0)
    sal_max = float(flat.get("salary_max_lpa") or 0.0)

    if sal_min <= 0.0 and sal_max <= 0.0:
        return 0.50   # no salary data → neutral

    sw_min  = float(SALARY_EXPECTATIONS["sweet_spot_min"])    # 40
    sw_max  = float(SALARY_EXPECTATIONS["sweet_spot_max"])    # 65
    mkt_max = float(SALARY_EXPECTATIONS["market_range_max"])  # 80
    mkt_min = float(SALARY_EXPECTATIONS["market_range_min"])  # 30

    # Use midpoint; if max == 0, treat max == min
    mid = (sal_min + sal_max) / 2.0 if sal_max > sal_min else sal_min

    if sw_min <= mid <= sw_max:
        return 1.00
    elif mid > mkt_max:
        return 0.30
    elif mid < mkt_min:
        return 0.50
    elif mid > sw_max:
        # Linear decay: sw_max → mkt_max maps 1.00 → 0.30
        t = (mid - sw_max) / (mkt_max - sw_max)
        return float(np.clip(1.00 - 0.70 * t, 0.30, 1.00))
    else:
        # Linear rise: mkt_min → sw_min maps 0.50 → 1.00
        t = (mid - mkt_min) / (sw_min - mkt_min)
        return float(np.clip(0.50 + 0.50 * t, 0.50, 1.00))


def _sub_experience(features: dict[str, float]) -> float:
    """
    experience_score — Gaussian YoE fit centred on JD ideal.

    Delegates entirely to f_yoe_fit from feature_engineering, which calls
    jd_understanding.score_experience_fit (Gaussian: μ=7.0, σ=2.5).

    A 7-year candidate scores 1.00; a 4-year candidate scores ~0.70;
    a 15-year candidate scores ~0.26 (likely overqualified for startup culture).
    """
    return float(np.clip(features.get("f_yoe_fit", 0.5), 0.0, 1.0))


def _sub_education(features: dict[str, float]) -> float:
    """
    education_score — institutional tier + field relevance.

    Feature signals:
      f_edu_tier_score     (0.50) — tier_1=1.0 … unknown=0.2
      f_edu_field_relevance (0.50) — CS/ML/math=1.0 … non-tech=0.2

    Equal weight: a tier-1 non-CS degree (e.g. physics from IIT) and a tier-3
    CS degree both score ~0.60, reflecting that field matters as much as brand.
    """
    f_tier  = features.get("f_edu_tier_score",      0.2)
    f_field = features.get("f_edu_field_relevance", 0.5)
    return float(np.clip(0.50 * f_tier + 0.50 * f_field, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# § 4  Main scorer
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_JD_SEMANTIC = 0.50   # neutral when TF-IDF score not available


def score_candidate(
    flat: dict[str, Any],
    *,
    jd_semantic_score: float | None = None,
    precomputed_features: dict[str, float] | None = None,
) -> CandidateScore:
    """
    Compute the composite ranking score for a single candidate.

    Parameters
    ----------
    flat : dict
        Flattened candidate dict from load_data.flatten_candidate().

    jd_semantic_score : float | None
        Pre-computed TF-IDF semantic score [0, 1] from text_features.py.
        Resolution order:
          1. This parameter (if not None)
          2. flat["jd_semantic_score"]  (if set during precompute)
          3. _DEFAULT_JD_SEMANTIC = 0.50 (neutral fallback)

    precomputed_features : dict[str, float] | None
        Pre-computed feature vector from feature_engineering.extract_features().
        If None, extract_features(flat) is called inline (slower for batch use).

    Returns
    -------
    CandidateScore with:
        .candidate_id      — str
        .composite_score   — float [0, 1], primary rank key
        .rank_key          — tuple (-composite_score, candidate_id) for sort
        .score_breakdown   — dict:
            "sub_scores"          : {name: float}   all 10 sub-scores
            "weights"             : {name: float}   SCORE_WEIGHTS
            "weighted_sub_scores" : {name: float}   sub_score × weight
            "weighted_sum"        : float            before trap_penalty
            "trap_penalty"        : float            compound penalty
            "composite_score"     : float            = weighted_sum × trap_penalty
            "weight_groups"       : {group: float}   group-level contributions
            "skill_detail"        : dict             raw score_skills_match output
            "trap_detail"         : dict             trap labels + risk + explanation
    """
    # ── Step 1  Resolve jd_semantic_score ─────────────────────────────────────
    if jd_semantic_score is None:
        raw_jd = flat.get("jd_semantic_score")
        jd_sem = float(raw_jd) if raw_jd is not None else _DEFAULT_JD_SEMANTIC
    else:
        jd_sem = float(jd_semantic_score)
    jd_sem = float(np.clip(jd_sem, 0.0, 1.0))

    # ── Step 2  Get feature vector ─────────────────────────────────────────────
    features: dict[str, float] = (
        precomputed_features if precomputed_features is not None
        else extract_features(flat)
    )

    # ── Step 3  Compute skill / domain scores from JD text analysis ───────────
    skill_names  = list(flat.get("skill_names") or [])
    career_text  = str(flat.get("career_descriptions_text") or "")
    summary_text = str(flat.get("summary") or "")
    skills_result: dict[str, float] = score_skills_match(
        skill_names, career_text, summary_text
    )

    # ── Step 4  Compute behavioral signal ─────────────────────────────────────
    bscore = float(np.clip(compute_behavioral_signal_score(flat), 0.0, 1.0))

    # ── Step 5  Compute trap penalty ──────────────────────────────────────────
    trap_result  = detect_traps(flat)
    trap_penalty = float(trap_result["trap_penalty"])

    # ── Step 6  Assemble all 10 sub-scores ────────────────────────────────────
    sub_scores: dict[str, float] = {
        "jd_semantic_score":       jd_sem,
        "must_have_skill_score":   _sub_must_have(skills_result),
        "retrieval_ranking_score": _sub_retrieval_ranking(skills_result),
        "production_ml_score":     _sub_production_ml(features, skills_result),
        "product_shipper_score":   _sub_product_shipper(features),
        "behavioral_signal_score": bscore,
        "location_score":          _sub_location(features),
        "salary_score":            _sub_salary(flat),
        "experience_score":        _sub_experience(features),
        "education_score":         _sub_education(features),
    }

    # Guarantee all sub-scores are in [0, 1]
    sub_scores = {k: float(np.clip(v, 0.0, 1.0)) for k, v in sub_scores.items()}

    # ── Step 7  Weighted sum  ──────────────────────────────────────────────────
    weighted_sum = sum(
        sub_scores[name] * SCORE_WEIGHTS[name]
        for name in SCORE_WEIGHTS
    )
    weighted_sub = {
        name: round(sub_scores[name] * SCORE_WEIGHTS[name], 6)
        for name in SCORE_WEIGHTS
    }

    # ── Step 8  Apply trap penalty  ───────────────────────────────────────────
    composite = float(np.clip(weighted_sum * trap_penalty, 0.0, 1.0))
    composite  = round(composite, 6)   # 6 dp for stable deterministic repr

    # ── Step 9  Group-level contributions (for high-level explainability) ─────
    def _group_sum(group: str) -> float:
        return round(
            sum(weighted_sub[n] for n in WEIGHT_GROUPS[group]), 6
        )

    weight_groups = {g: _group_sum(g) for g in WEIGHT_GROUPS}

    # ── Step 10  Build score_breakdown ────────────────────────────────────────
    score_breakdown: dict[str, Any] = {
        "sub_scores":          {k: round(v, 6) for k, v in sub_scores.items()},
        "weights":             dict(SCORE_WEIGHTS),
        "weighted_sub_scores": weighted_sub,
        "weighted_sum":        round(weighted_sum, 6),
        "trap_penalty":        round(trap_penalty, 6),
        "composite_score":     composite,
        "weight_groups":       weight_groups,
        "skill_detail":        skills_result,
        "trap_detail": {
            "trap_labels":     trap_result["trap_labels"],
            "trap_risk_score": trap_result["trap_risk_score"],
            "explanation":     trap_result["explanation"],
        },
    }

    # ── Step 11  Tie-break key ────────────────────────────────────────────────
    candidate_id = str(flat.get("candidate_id") or "")
    rank_key     = (-composite, candidate_id)   # sort ascending → rank DESC score, ASC id

    return CandidateScore(
        candidate_id    = candidate_id,
        composite_score = composite,
        rank_key        = rank_key,
        score_breakdown = score_breakdown,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § 5  Batch scorer
# ─────────────────────────────────────────────────────────────────────────────

_BATCH_COLS: list[str] = (
    list(SCORE_WEIGHTS.keys())
    + ["trap_penalty", "weighted_sum",
       "group_career_relevance", "group_skill_depth",
       "group_behavioral", "group_location", "group_experience_fit"]
)


def score_candidates(
    flat_rows: list[dict[str, Any]],
    jd_semantic_scores: "np.ndarray | None" = None,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Score all candidates and return a fully ranked DataFrame.

    Ranking is deterministic:
      primary   — composite_score DESC
      tie-break — candidate_id    ASC  (lexicographic)

    Parameters
    ----------
    flat_rows : list[dict]
        One flat candidate dict per candidate (from load_data.flatten_candidate).

    jd_semantic_scores : np.ndarray | None
        Pre-computed TF-IDF semantic scores, aligned with flat_rows by index.
        Shape: (len(flat_rows),).  If None, falls back to flat["jd_semantic_score"]
        or _DEFAULT_JD_SEMANTIC.

    show_progress : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    pd.DataFrame indexed by candidate_id, sorted by rank (ascending = 1st place first).
    Columns:
        composite_score           — final score [0, 1]
        rank                      — 1-based rank after tie-break sort
        <10 sub-score columns>
        trap_penalty
        weighted_sum
        group_career_relevance
        group_skill_depth
        group_behavioral
        group_location
        group_experience_fit
    """
    if not flat_rows:
        return pd.DataFrame(columns=["composite_score", "rank", *SCORE_WEIGHTS.keys()])

    n = len(flat_rows)
    scored: list[CandidateScore] = []

    iterable: Any = (
        tqdm(enumerate(flat_rows), total=n, desc="scoring candidates")
        if show_progress else enumerate(flat_rows)
    )

    for i, flat in iterable:
        jd_score = (
            float(jd_semantic_scores[i])
            if jd_semantic_scores is not None else None
        )
        cs = score_candidate(flat, jd_semantic_score=jd_score)
        scored.append(cs)

    # Deterministic sort: descending score, ascending id on ties
    scored.sort(key=lambda c: c.rank_key)

    rows: list[dict[str, Any]] = []
    for rank_pos, cs in enumerate(scored, start=1):
        bd = cs.score_breakdown
        row: dict[str, Any] = {
            "candidate_id":           cs.candidate_id,
            "composite_score":        cs.composite_score,
            "rank":                   rank_pos,
            **bd["sub_scores"],
            "trap_penalty":           bd["trap_penalty"],
            "weighted_sum":           bd["weighted_sum"],
            "group_career_relevance": bd["weight_groups"]["career_relevance"],
            "group_skill_depth":      bd["weight_groups"]["skill_depth"],
            "group_behavioral":       bd["weight_groups"]["behavioral"],
            "group_location":         bd["weight_groups"]["location"],
            "group_experience_fit":   bd["weight_groups"]["experience_fit"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("candidate_id")
    df["rank"] = df["rank"].astype(int)
    return df
