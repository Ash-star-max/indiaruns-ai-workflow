"""
feature_engineering.py — Normalized feature extraction from candidate flat dicts.

Converts a flat dict (from load_data.flatten_candidate) into a clean numeric
feature vector where every value is a float in [0, 1].  Higher = better fit.

Input:  flat dict produced by flatten_candidate()
Output: dict[str, float]  (keys == FEATURE_NAMES)
        OR pd.DataFrame   from build_feature_matrix()

Feature groups (27 features)
-----------------------------
  Experience     (5)  yoe_fit · yoe_normalized · ml_title_ratio · seniority · length
  Role           (2)  role_is_ml · role_ml_relevance
  Company        (3)  product_ratio · company_size_fit · industry_relevance
  Skills         (5)  tier1_density · tier1_depth · tier2_breadth · endorsements · assessment
  Education      (2)  edu_tier · edu_field
  Certifications (2)  cert_ml_relevance · cert_count_score
  Languages      (2)  english_proficiency · language_diversity
  Location       (3)  location_score · notice_score · relocation_ready
  Disqualifiers  (3)  disqualifier_penalty · consulting_clean · activity_score

Public API
----------
    FEATURE_NAMES                        list[str] — all 27 feature keys
    extract_features(flat)               dict[str, float]
    build_feature_matrix(rows, ...)      pd.DataFrame  (float32, index=candidate_id)

    # Low-level scorers (also importable for testing / introspection)
    score_yoe_fit(years)
    score_seniority(title)
    score_role_ml_relevance(title)
    score_company_size(size_str)
    score_industry(industry)
    score_edu_tier(tier)
    score_edu_field(field)
    score_cert_relevance(cert_names)
    score_english_proficiency(proficiency)
    score_activity(days_since_active, open_to_work)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.jd_understanding import (
    compute_disqualifier_penalty,
    score_experience_fit,
    score_location_match,
    score_notice_period,
)

# ─────────────────────────────────────────────────────────────────────────────
# § 1  Feature name registry
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # ── Experience (5) ────────────────────────────────────────────────────────
    "f_yoe_fit",              # Gaussian fit centred on JD ideal (7 yrs, σ=2.5)
    "f_yoe_normalized",       # raw YoE / 12, capped at 1
    "f_applied_ml_ratio",     # fraction of career titles that are ML/AI roles
    "f_career_seniority",     # seniority level of most-recent title → [0,1]
    "f_career_length_score",  # total_career_months / 120, capped at 1
    # ── Current role (2) ─────────────────────────────────────────────────────
    "f_current_role_is_ml",   # 1.0 if current title is ML/AI, else 0.0
    "f_role_ml_relevance",    # scored relevance of current title (graded, not binary)
    # ── Company (3) ──────────────────────────────────────────────────────────
    "f_product_company_ratio",  # fraction of career at product (non-consulting) cos
    "f_company_size_fit",       # company-size fit for startup founding team hire
    "f_industry_relevance",     # tech/AI industry score of current role
    # ── Skills (5) ───────────────────────────────────────────────────────────
    "f_tier1_density",           # tier1_skill_count / skill_count (capped)
    "f_tier1_depth",             # avg tier-1 skill duration → [0,1]
    "f_tier2_breadth",           # tier2 skill count score
    "f_skill_endorsement_score", # avg per-skill endorsements (normalized)
    "f_assessment_score",        # Redrob skill assessment average
    # ── Education (2) ────────────────────────────────────────────────────────
    "f_edu_tier_score",       # institutional tier (tier_1=1.0 … unknown=0.2)
    "f_edu_field_relevance",  # CS / ML / math field relevance
    # ── Certifications (2) ───────────────────────────────────────────────────
    "f_cert_ml_relevance",    # fraction of certs that are ML-relevant
    "f_cert_count_score",     # normalized certification count
    # ── Languages (2) ────────────────────────────────────────────────────────
    "f_english_proficiency_score",  # English proficiency → [0,1]
    "f_language_diversity",         # normalized language count
    # ── Location & availability (3) ──────────────────────────────────────────
    "f_location_score",    # India + preferred city = 1.0
    "f_notice_score",      # ≤30 days = 1.0, >120 days = 0.20
    "f_relocation_ready",  # willing_to_relocate boolean → 0.0 or 1.0
    # ── Disqualifiers (3) ────────────────────────────────────────────────────
    "f_disqualifier_penalty",  # compound penalty (1.0 = clean, 0.15 = eliminated)
    "f_consulting_clean",      # 1.0 if not consulting-only, 0.0 if consulting-only
    "f_activity_score",        # recency of platform activity (exponential decay)
]

assert len(FEATURE_NAMES) == 27, f"Expected 27 features, got {len(FEATURE_NAMES)}"

# ─────────────────────────────────────────────────────────────────────────────
# § 2  Internal lookup tables
# ─────────────────────────────────────────────────────────────────────────────

# ML / AI title keyword signals — any hit → role is ML-relevant
_ML_TITLE_KEYWORDS: frozenset[str] = frozenset({
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp engineer", "nlp", "natural language",
    "deep learning", "applied scientist", "research scientist",
    "research engineer", "computer vision", "recommendation",
    "search engineer", "ranking engineer", "retrieval", "embeddings",
    "recsys", "mlops", "ml platform", "ai researcher",
})

# Adjacent technical roles (not ML but close enough for partial credit)
_ADJACENT_TECH_KEYWORDS: frozenset[str] = frozenset({
    "software engineer", "software developer", "backend engineer",
    "data engineer", "analytics engineer", "platform engineer",
    "python developer", "full stack", "sre", "site reliability",
})

# Company size → startup-founding-team fit score
_COMPANY_SIZE_SCORES: dict[str, float] = {
    "1-10":      0.55,
    "11-50":     0.75,
    "51-200":    0.90,
    "201-500":   1.00,
    "501-1000":  0.90,
    "1001-5000": 0.80,
    "5001-10000":0.70,
    "10001+":    0.60,
}

# Industry keywords → tech relevance
_TECH_INDUSTRY_KEYWORDS: frozenset[str] = frozenset({
    "technology", "software", "saas", "ai", "machine learning",
    "fintech", "e-commerce", "ecommerce", "internet", "data",
    "cloud", "platform", "startup", "product", "analytics",
})
_ADJACENT_INDUSTRY_KEYWORDS: frozenset[str] = frozenset({
    "finance", "banking", "financial services", "investment",
    "healthcare", "pharma", "biotech", "logistics", "telecom",
})

# Education tier → score
_EDU_TIER_SCORES: dict[str, float] = {
    "tier_1": 1.00,
    "tier_2": 0.80,
    "tier_3": 0.55,
    "tier_4": 0.35,
    "unknown": 0.20,
}

# Education field keywords → relevance score
_ML_FIELD_KEYWORDS: frozenset[str] = frozenset({
    "computer science", "software engineering", "information technology",
    "artificial intelligence", "machine learning", "data science",
    "statistics", "mathematics", "computer engineering",
    "information systems",
})
_ADJACENT_FIELD_KEYWORDS: frozenset[str] = frozenset({
    "physics", "electrical engineering", "electronics",
    "electronics and communication", "quantitative finance",
    "economics", "operations research",
})

# Certification keywords → ML-relevant
_ML_CERT_KEYWORDS: frozenset[str] = frozenset({
    "machine learning", "deep learning", "tensorflow", "pytorch",
    "natural language processing", "nlp", "data science",
    "aws machine learning", "gcp", "azure ai", "azure ml",
    "google cloud ml", "kaggle", "fast.ai",
    "stanford", "coursera ml", "neural network",
    "applied ai", "ai engineer", "ml engineer",
})

# English proficiency string → score
_ENGLISH_PROF_SCORES: dict[str, float] = {
    "native":               1.00,
    "native speaker":       1.00,
    "bilingual":            1.00,
    "fluent":               0.95,
    "professional":         0.90,
    "full professional":    0.90,
    "advanced":             0.80,
    "upper intermediate":   0.70,
    "intermediate":         0.60,
    "elementary":           0.35,
    "beginner":             0.20,
    "limited":              0.20,
    "unknown":              0.70,   # assume decent English for Indian candidates
}

# Activity score half-life (days) — exponential decay
_ACTIVITY_HALFLIFE_DAYS: float = 60.0

# ─────────────────────────────────────────────────────────────────────────────
# § 3  Low-level scorer functions  (pure, independently testable)
# ─────────────────────────────────────────────────────────────────────────────

def score_yoe_fit(years: float) -> float:
    """
    Gaussian fit for years-of-experience against JD ideal (μ=7, σ=2.5).
    Delegates to jd_understanding.score_experience_fit.
    Returns float in (0, 1].
    """
    return score_experience_fit(float(years))


def score_seniority(title: str) -> float:
    """
    Map a job title to a seniority score in [0, 1].

    Captures career progression level, not domain relevance:
      intern / trainee                → 0.10
      junior / associate              → 0.25
      (plain engineer, no prefix)     → 0.45
      senior                          → 0.70
      lead / tech lead                → 0.75
      staff                           → 0.80
      principal / fellow              → 0.85
      manager (non-executive)         → 0.40  (management pivot, not ideal for IC role)
      director / vp / head / CxO      → 0.35  (far from IC)
    """
    t = title.lower().strip()
    if not t:
        return 0.45

    # Executive / director — not ideal for founding IC role
    if any(kw in t for kw in ("cto", "ceo", "cpo", "cmo", "chief", "director",
                               "vice president", "vp ", "head of")):
        return 0.35
    if "manager" in t:
        return 0.40
    if any(kw in t for kw in ("principal", "fellow", "distinguished")):
        return 0.85
    if "staff" in t:
        return 0.80
    if any(kw in t for kw in ("lead", "tech lead", "technical lead")):
        return 0.75
    if "senior" in t:
        return 0.70
    if any(kw in t for kw in ("junior", "associate", "jr.")):
        return 0.25
    if any(kw in t for kw in ("intern", "trainee", "student", "fresher")):
        return 0.10
    return 0.45   # plain engineer / scientist with no seniority prefix


def score_role_ml_relevance(title: str) -> float:
    """
    Score how relevant a job title is to the ML/AI engineering JD.

    1.00  — direct ML/AI title
    0.40  — adjacent technical (SWE, data engineer)
    0.15  — non-technical (HR, sales, etc.)
    0.00  — empty title
    """
    t = title.lower().strip()
    if not t:
        return 0.00
    if any(kw in t for kw in _ML_TITLE_KEYWORDS):
        return 1.00
    if any(kw in t for kw in _ADJACENT_TECH_KEYWORDS):
        return 0.40
    return 0.15


def score_company_size(size_str: str) -> float:
    """
    Map the canonical company-size band to a startup-culture fit score.
    Sweet spot is 201-500 employees (proven scale but still product-focused).
    """
    return _COMPANY_SIZE_SCORES.get(size_str.strip(), 0.50)


def score_industry(industry: str) -> float:
    """
    Score the relevance of an industry string to a tech/AI product company.

    1.00  — direct tech / AI / software
    0.65  — adjacent (fintech, healthcare, banking)
    0.35  — other / unrelated
    """
    ind = industry.lower()
    if any(kw in ind for kw in _TECH_INDUSTRY_KEYWORDS):
        return 1.00
    if any(kw in ind for kw in _ADJACENT_INDUSTRY_KEYWORDS):
        return 0.65
    return 0.35


def score_edu_tier(tier: str) -> float:
    """Map education tier string to a score in [0,1]."""
    return _EDU_TIER_SCORES.get(tier.strip().lower(), 0.20)


def score_edu_field(field: str) -> float:
    """
    Score how relevant an education field is to ML engineering.

    1.00  — CS / software / data science / AI / mathematics / statistics
    0.65  — physics / EE / electronics / economics / OR
    0.40  — general engineering (mechanical, civil, chemical)
    0.20  — non-technical / unknown
    """
    f = field.lower().strip()
    if not f:
        return 0.20
    if any(kw in f for kw in _ML_FIELD_KEYWORDS):
        return 1.00
    if any(kw in f for kw in _ADJACENT_FIELD_KEYWORDS):
        return 0.65
    if "engineering" in f or "technology" in f:
        return 0.40
    return 0.20


def score_cert_relevance(cert_names: list[str]) -> tuple[float, float]:
    """
    Score a list of certification names for ML relevance.

    Returns (ml_relevance, count_score):
      ml_relevance   — fraction of certs that are ML-relevant, or 0 if none
      count_score    — normalized cert count (cap at 5 certs = 1.0)
    """
    if not cert_names:
        return 0.0, 0.0
    ml_count = sum(
        1 for name in cert_names
        if any(kw in name.lower() for kw in _ML_CERT_KEYWORDS)
    )
    ml_relevance = ml_count / len(cert_names)
    count_score  = min(1.0, len(cert_names) / 5.0)
    return ml_relevance, count_score


def score_english_proficiency(proficiency: str) -> float:
    """Map English proficiency string → [0,1]. Defaults to 0.70 for unknown."""
    key = proficiency.lower().strip()
    return _ENGLISH_PROF_SCORES.get(key, 0.70)


def score_activity(days_since_active: int, open_to_work: bool) -> float:
    """
    Compute platform activity recency score using exponential decay.

    Half-life = 60 days. Open-to-work flag boosts the score by 10%.
    Score is capped at [0, 1].
    """
    if days_since_active >= 9999:
        # Sentinel value from flatten_candidate when last_active_date is None
        base = 0.05
    else:
        decay = -math.log(2) / _ACTIVITY_HALFLIFE_DAYS
        base = math.exp(decay * days_since_active)

    if open_to_work:
        base = min(1.0, base * 1.10)
    return round(float(base), 4)


# ─────────────────────────────────────────────────────────────────────────────
# § 4  Applied ML career ratio helper
# ─────────────────────────────────────────────────────────────────────────────

def _applied_ml_title_ratio(career_titles: list[str]) -> float:
    """
    Estimate the fraction of career roles that were in ML/AI positions.

    Based on title matching only (duration data not available per-title in
    the flat dict).  Returns 0.0 for an empty career history.
    """
    if not career_titles:
        return 0.0
    ml_count = sum(
        1 for t in career_titles
        if any(kw in t.lower() for kw in _ML_TITLE_KEYWORDS)
    )
    return ml_count / len(career_titles)


# ─────────────────────────────────────────────────────────────────────────────
# § 5  Main extraction function
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(flat: dict[str, Any]) -> dict[str, float]:
    """
    Convert a flat candidate dict (from flatten_candidate) into a normalized
    feature dict.  Every value is a float clipped to [0, 1].

    Parameters
    ----------
    flat : dict produced by load_data.flatten_candidate()

    Returns
    -------
    dict[str, float] with exactly FEATURE_NAMES keys.
    """
    g = flat.get   # convenience alias; avoids KeyError on optional fields

    # ── Experience ────────────────────────────────────────────────────────────
    yoe        = float(g("years_of_experience", 0.0))
    total_mos  = int(g("total_career_months", 0))
    titles     = g("career_titles", []) or []
    most_recent_title = str(g("most_recent_title", "") or g("current_title", "") or "")

    f_yoe_fit            = score_yoe_fit(yoe)
    f_yoe_normalized     = min(1.0, yoe / 12.0)
    f_applied_ml_ratio   = _applied_ml_title_ratio(titles)
    f_career_seniority   = score_seniority(most_recent_title)
    f_career_length_score = min(1.0, total_mos / 120.0)

    # ── Current role ─────────────────────────────────────────────────────────
    current_title     = str(g("current_title", "") or most_recent_title or "")
    ml_relevance      = score_role_ml_relevance(current_title)
    f_current_role_is_ml  = 1.0 if ml_relevance >= 1.0 else 0.0
    f_role_ml_relevance   = ml_relevance

    # ── Company ───────────────────────────────────────────────────────────────
    f_product_company_ratio = float(g("product_company_ratio", 0.0))
    f_company_size_fit      = score_company_size(str(g("current_company_size", "") or ""))
    f_industry_relevance    = score_industry(str(g("current_industry", "") or ""))

    # ── Skills ───────────────────────────────────────────────────────────────
    skill_count    = max(int(g("skill_count", 0)), 1)   # avoid div/0
    tier1_count    = int(g("tier1_skill_count", 0))
    tier2_count    = int(g("tier2_skill_count", 0))
    tier1_avg_dur  = float(g("tier1_avg_duration_months", 0.0))
    total_endorse  = int(g("total_skill_endorsements", 0))
    assess_scores  = g("skill_assessment_scores", {}) or {}

    # tier1_density: fraction of skills that are tier-1, bounded so that
    # having many tier-1 skills (≥5) earns full score
    f_tier1_density = min(1.0, tier1_count / 5.0)

    # tier1_depth: avg duration of tier-1 skills; 24 months = full score
    f_tier1_depth   = min(1.0, tier1_avg_dur / 24.0)

    # tier2_breadth: each of ≤8 tier-2 skills adds to breadth
    f_tier2_breadth = min(1.0, tier2_count / 8.0)

    # endorsement score: avg per-skill endorsements; 10 avg = full score
    avg_endorse  = total_endorse / skill_count
    f_skill_endorsement_score = min(1.0, avg_endorse / 10.0)

    # assessment score: average of available Redrob scores (assumed 0-100 scale)
    if assess_scores:
        raw_avg = sum(float(v) for v in assess_scores.values()) / len(assess_scores)
        f_assessment_score = min(1.0, raw_avg / 100.0)
    else:
        f_assessment_score = 0.50   # neutral: no assessment ≠ bad candidate

    # ── Education ─────────────────────────────────────────────────────────────
    edu_tier   = str(g("highest_edu_tier", "unknown") or "unknown")
    edu_fields = g("edu_fields", []) or []
    f_edu_tier_score    = score_edu_tier(edu_tier)
    # Take best field relevance across all education entries
    if edu_fields:
        f_edu_field_relevance = max(score_edu_field(f) for f in edu_fields)
    else:
        f_edu_field_relevance = 0.20

    # ── Certifications ────────────────────────────────────────────────────────
    cert_names = g("cert_names", []) or []
    f_cert_ml_relevance, f_cert_count_score = score_cert_relevance(cert_names)

    # ── Languages ─────────────────────────────────────────────────────────────
    eng_proficiency   = str(g("english_proficiency", "unknown") or "unknown")
    lang_count        = int(g("language_count", 0))
    f_english_proficiency_score = score_english_proficiency(eng_proficiency)
    f_language_diversity        = min(1.0, lang_count / 4.0)   # 4+ = full score

    # ── Location & availability ───────────────────────────────────────────────
    country      = str(g("country", "") or "")
    location     = str(g("location", "") or "")
    willing      = bool(g("willing_to_relocate", False))
    notice_days  = int(g("notice_period_days", 90))
    f_location_score   = score_location_match(country, location, willing)
    f_notice_score     = score_notice_period(notice_days)
    f_relocation_ready = 1.0 if willing else 0.0

    # ── Disqualifiers ─────────────────────────────────────────────────────────
    f_disqualifier_penalty = float(compute_disqualifier_penalty(flat))
    f_consulting_clean     = 0.0 if bool(g("is_consulting_only", False)) else 1.0

    days_inactive   = int(g("days_since_last_active", 9999))
    open_to_work    = bool(g("open_to_work_flag", False))
    f_activity_score = score_activity(days_inactive, open_to_work)

    # ── Assemble and clip all values to [0, 1] ────────────────────────────────
    raw = {
        "f_yoe_fit":                 f_yoe_fit,
        "f_yoe_normalized":          f_yoe_normalized,
        "f_applied_ml_ratio":        f_applied_ml_ratio,
        "f_career_seniority":        f_career_seniority,
        "f_career_length_score":     f_career_length_score,
        "f_current_role_is_ml":      f_current_role_is_ml,
        "f_role_ml_relevance":       f_role_ml_relevance,
        "f_product_company_ratio":   f_product_company_ratio,
        "f_company_size_fit":        f_company_size_fit,
        "f_industry_relevance":      f_industry_relevance,
        "f_tier1_density":           f_tier1_density,
        "f_tier1_depth":             f_tier1_depth,
        "f_tier2_breadth":           f_tier2_breadth,
        "f_skill_endorsement_score": f_skill_endorsement_score,
        "f_assessment_score":        f_assessment_score,
        "f_edu_tier_score":          f_edu_tier_score,
        "f_edu_field_relevance":     f_edu_field_relevance,
        "f_cert_ml_relevance":       f_cert_ml_relevance,
        "f_cert_count_score":        f_cert_count_score,
        "f_english_proficiency_score": f_english_proficiency_score,
        "f_language_diversity":      f_language_diversity,
        "f_location_score":          f_location_score,
        "f_notice_score":            f_notice_score,
        "f_relocation_ready":        f_relocation_ready,
        "f_disqualifier_penalty":    f_disqualifier_penalty,
        "f_consulting_clean":        f_consulting_clean,
        "f_activity_score":          f_activity_score,
    }
    return {k: float(np.clip(v, 0.0, 1.0)) for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# § 6  Batch builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    flat_rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Apply extract_features to every candidate row, return a tidy float32 DataFrame.

    Parameters
    ----------
    flat_rows      : list of flat dicts OR a DataFrame from load_flat().
    show_progress  : show tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        Index   : candidate_id
        Columns : FEATURE_NAMES  (27 float32 columns, all in [0, 1])
    """
    if isinstance(flat_rows, pd.DataFrame):
        rows = flat_rows.to_dict("records")
    else:
        rows = list(flat_rows)

    iterable = tqdm(rows, desc="Extracting features") if show_progress else rows
    feature_rows = [extract_features(r) for r in iterable]

    df = pd.DataFrame(feature_rows, columns=FEATURE_NAMES, dtype=np.float32)
    candidate_ids = [
        r.get("candidate_id", str(i)) for i, r in enumerate(rows)
    ]
    df.index = pd.Index(candidate_ids, name="candidate_id")
    return df
