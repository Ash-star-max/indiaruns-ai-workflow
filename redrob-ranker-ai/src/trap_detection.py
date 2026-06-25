"""
trap_detection.py — Detect manipulated, low-quality, or misleading candidate profiles.

Ten trap detectors produce three output values:

    trap_risk_score : float [0, 1]   higher = more suspicious
    trap_labels     : list[str]      names of triggered traps
    trap_penalty    : float [0, 1]   multiplicative penalty on final composite score

Each detector returns a TrapSignal with:
    name           : str
    triggered      : bool
    confidence     : float [0, 1]    how strongly evidence points at the trap
    penalty_factor : float [0, 1]    1.0 = no penalty; lower = larger penalty
    evidence       : dict[str, Any]  raw signals that fired
    description    : str             human-readable verdict

Compound penalty
----------------
trap_penalty = product(s.penalty_factor for s in triggered_signals)
Clamped to [MIN_COMPOUND_PENALTY, 1.0] so no candidate reaches zero.

trap_risk_score
---------------
Severity-weighted average of triggered confidences:
    weight_i = 1 - penalty_factor_i   (harsher penalties count more)

Flat dict keys consumed
-----------------------
summary, headline, career_descriptions_text, candidate_text,
skill_names, skill_count, tier1_skill_count, tier2_skill_count,
expert_zero_duration_count, tier1_avg_duration_months, tier2_avg_duration_months,
career_titles, n_career_roles, total_career_months, years_of_experience,
most_recent_title, is_consulting_only, product_company_ratio,
highest_edu_tier, edu_fields,
profile_completeness_score, connection_count, github_activity_score,
salary_inverted, salary_min_lpa, salary_max_lpa,
days_since_last_active, open_to_work_flag,
applications_submitted_30d, profile_views_received_30d, search_appearance_30d,
recruiter_response_rate, interview_completion_rate,
offer_acceptance_rate, verified_email, verified_phone, linkedin_connected,
saved_by_recruiters_30d

Public API
----------
    TRAP_NAMES              — ordered list of 10 trap names
    TRAP_PENALTY_FACTORS    — per-trap base penalty when triggered
    MIN_COMPOUND_PENALTY    — floor for compound penalty
    TrapSignal              — dataclass for per-trap result
    detect_traps(flat) -> dict          ← main entry point
    run_all_detectors(flat) -> list[TrapSignal]   ← list of all 10 signals
    build_trap_report(flat_rows, *, show_progress) -> pd.DataFrame
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# § 1  Constants & registries
# ─────────────────────────────────────────────────────────────────────────────

MIN_COMPOUND_PENALTY: float = 0.05   # Even the worst candidate retains 5%

TRAP_NAMES: list[str] = [
    "keyword_stuffing",
    "fake_ai_profile",
    "generic_chatgpt_user",
    "research_only",
    "low_quality_profile",
    "inactive_candidate",
    "inconsistent_career",
    "suspicious_timeline",
    "ai_keywords_no_production",
    "behavioral_trust_issues",
]

# Penalty factor applied when each trap triggers (1.0 = no penalty)
TRAP_PENALTY_FACTORS: dict[str, float] = {
    "keyword_stuffing":           0.70,
    "fake_ai_profile":            0.20,
    "generic_chatgpt_user":       0.75,
    "research_only":              0.30,
    "low_quality_profile":        0.55,
    "inactive_candidate":         0.65,
    "inconsistent_career":        0.60,
    "suspicious_timeline":        0.50,
    "ai_keywords_no_production":  0.55,
    "behavioral_trust_issues":    0.60,
}

# Minimum confidence before a trap is marked triggered
_THRESHOLDS: dict[str, float] = {
    "keyword_stuffing":           0.45,
    "fake_ai_profile":            0.45,
    "generic_chatgpt_user":       0.40,
    "research_only":              0.45,
    "low_quality_profile":        0.45,
    "inactive_candidate":         0.50,
    "inconsistent_career":        0.45,
    "suspicious_timeline":        0.45,
    "ai_keywords_no_production":  0.45,
    "behavioral_trust_issues":    0.45,
}

# ── Text corpora compiled once at import ─────────────────────────────────────

# Generic ChatGPT / template phrases (look for these in summary)
_CHATGPT_PHRASES: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bpassionate about\b",
    r"\bresults[- ]driven\b",
    r"\bproven track record\b",
    r"\bcross[- ]functional teams?\b",
    r"\bleveraging cutting[- ]edge\b",
    r"\binnovative solutions?\b",
    r"\bseeking (new )?opportunities\b",
    r"\b(aspiring|budding) (ai|ml|data|machine learning)\b",
    r"\b(machine learning|ai|data science) enthusiast\b",
    r"\bdynamic professional\b",
    r"\bself[- ]motivated\b",
    r"\bteam player\b",
    r"\bsynergiz\w+\b",
    r"\bthought leader\b",
    r"\bgame[- ]changer\b",
    r"\bout[- ]of[- ]the[- ]box thinking\b",
    r"\bbest practices\b",
    r"\bgo[- ]getter\b",
    r"\bdetail[- ]oriented\b",
    r"\bstrong communication skills\b",
    r"\bdedicated professional\b",
    r"\bvalue[- ]add\b",
    r"\bwin[- ]win\b",
    r"\bparadigm shift\b",
]]

# AI / ML buzzwords that can be stuffed into profiles
_AI_BUZZWORDS: set[str] = {
    "machine learning", "deep learning", "neural network", "artificial intelligence",
    "generative ai", "large language model", "llm", "natural language processing",
    "nlp", "computer vision", "reinforcement learning", "transformer", "attention",
    "fine-tuning", "bert", "gpt", "llama", "mistral", "claude", "openai",
    "pytorch", "tensorflow", "keras", "jax", "scikit-learn", "huggingface",
    "langchain", "llamaindex", "semantic search", "vector database", "embedding",
    "rag", "retrieval augmented generation", "faiss", "pinecone", "weaviate",
    "qdrant", "chromadb", "milvus", "elasticsearch", "opensearch",
    "data science", "data engineering", "mlops", "feature engineering",
    "model training", "model evaluation", "hyperparameter tuning", "ndcg",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "aws", "azure", "gcp", "google cloud", "amazon web services",
    "big data", "spark", "hadoop", "kafka", "airflow",
}

# Phrases that indicate real production deployment (reward these; penalise absence)
_PRODUCTION_KEYWORDS: set[str] = {
    "deployed", "production", "serving", "inference", "latency", "throughput",
    "api", "endpoint", "microservice", "rest api", "grpc",
    "docker", "kubernetes", "k8s", "helm", "terraform",
    "mlflow", "bentoml", "torchserve", "triton", "ray serve",
    "a/b test", "ab test", "shadow mode", "canary", "rollout",
    "monitoring", "drift detection", "alerting", "sla",
    "pipeline", "orchestration", "ci/cd", "github actions",
    "real-time", "batch inference", "online serving",
    "scaling", "load balancing", "replicas", "auto-scaling",
    "million", "billion", "requests per second", "rps", "qps",
    "cost reduction", "latency improvement", "accuracy improvement",
}

# Academic / research-only keywords
_ACADEMIC_KEYWORDS: set[str] = {
    "published", "publication", "paper", "arxiv", "preprint",
    "thesis", "dissertation", "phd", "ph.d", "doctorate",
    "professor", "faculty", "post-doc", "postdoc", "research associate",
    "cited", "citation", "h-index", "journal", "conference paper",
    "workshop", "proceedings", "peer review", "peer-reviewed",
    "iclr", "neurips", "icml", "acl", "emnlp", "cvpr", "eccv",
    "grant", "fellowship", "stipend", "scholarship",
}

# Seniority levels for career regression detection
_SENIORITY: dict[str, int] = {
    "intern":      0,  "trainee":   0,  "apprentice": 0,
    "junior":      1,  "associate": 1,  "entry":      1,  "graduate":   1,
    "mid":         2,  "engineer":  2,  "developer":  2,  "analyst":    2,
    "senior":      3,  "sr":        3,  "lead":       4,  "tech lead":  4,
    "principal":   5,  "staff":     5,  "architect":  5,
    "manager":     3,  "head":      5,  "director":   6,
    "vp":          7,  "vice president": 7,
    "cto":         8,  "chief":     8,
}

# Academic/research title keywords
_RESEARCH_TITLES: set[str] = {
    "research", "researcher", "scientist", "phd", "post-doc", "postdoc",
    "professor", "faculty", "academic", "intern", "fellow", "student",
    "graduate", "ra ", "ta ", "teaching assistant", "research assistant",
}


# ─────────────────────────────────────────────────────────────────────────────
# § 2  TrapSignal dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrapSignal:
    name:           str
    triggered:      bool
    confidence:     float                        # [0, 1]
    penalty_factor: float                        # 1.0 = no penalty
    evidence:       dict[str, Any] = field(default_factory=dict)
    description:    str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name":           self.name,
            "triggered":      self.triggered,
            "confidence":     round(self.confidence, 4),
            "penalty_factor": self.penalty_factor,
            "evidence":       self.evidence,
            "description":    self.description,
        }


# ─────────────────────────────────────────────────────────────────────────────
# § 3  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _g(flat: dict, key: str, default: Any = None) -> Any:
    """Safe getter that treats None the same as missing."""
    v = flat.get(key)
    return default if v is None else v


def _text(flat: dict) -> str:
    """Combined lower-cased candidate text for keyword matching."""
    parts = [
        str(_g(flat, "summary", "")),
        str(_g(flat, "headline", "")),
        str(_g(flat, "career_descriptions_text", "")),
    ]
    return " ".join(parts).lower()


def _count_set_hits(text: str, word_set: set[str]) -> int:
    """Count how many multi-word phrases from word_set appear in text."""
    return sum(1 for phrase in word_set if phrase in text)


def _title_seniority(title: str) -> int:
    """Map a job title to a seniority level integer (higher = more senior)."""
    t = title.lower()
    best = 2   # default: mid-level
    for kw, level in _SENIORITY.items():
        if kw in t:
            if level != 2:  # don't override with default
                best = max(best, level) if level > 2 else min(best, level)
    # Re-evaluate: take the most specific match
    for kw, level in sorted(_SENIORITY.items(), key=lambda x: len(x[0]), reverse=True):
        if kw in t:
            return level
    return 2


def _make_signal(name: str, confidence: float, evidence: dict,
                 description: str) -> TrapSignal:
    """Build a TrapSignal, computing triggered from confidence vs threshold."""
    threshold = _THRESHOLDS[name]
    triggered = confidence >= threshold
    penalty   = TRAP_PENALTY_FACTORS[name] if triggered else 1.0
    return TrapSignal(
        name=name,
        triggered=triggered,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        penalty_factor=penalty,
        evidence=evidence,
        description=description,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § 4  Individual detectors
# ─────────────────────────────────────────────────────────────────────────────

def detect_keyword_stuffing(flat: dict) -> TrapSignal:
    """
    Detect profiles crammed with AI buzzwords or an excessive skills list.

    Signals
    -------
    - skill_count > 25 (each +5 adds ~0.10 confidence)
    - expert_zero_duration_count > 3 (skill listed as "expert" with 0 months)
    - Buzzword density in summary + career text > 18%
    """
    skill_count     = int(_g(flat, "skill_count",              0))
    expert_zero     = int(_g(flat, "expert_zero_duration_count", 0))
    text            = _text(flat)
    words           = text.split()
    total_words     = max(len(words), 1)

    # Signal A: skills list length (normal upper bound ~20)
    skills_score = min(1.0, max(0.0, (skill_count - 25) / 25.0))

    # Signal B: expert/advanced skill with 0 usage months
    zero_score = min(1.0, expert_zero / 6.0)

    # Signal C: buzzword density
    buzz_hits   = _count_set_hits(text, _AI_BUZZWORDS)
    density     = buzz_hits / total_words
    density_score = min(1.0, density / 0.18)

    base = 0.40 * skills_score + 0.30 * zero_score + 0.30 * density_score
    # A single dominant signal (e.g. 10 expert-skills with 0 usage) should alone trigger
    confidence = max(base, zero_score * 0.48, density_score * 0.48)
    evidence = {
        "skill_count":              skill_count,
        "expert_zero_duration_count": expert_zero,
        "buzzword_density":         round(density, 4),
        "buzzword_hits":            buzz_hits,
    }
    parts = []
    if skill_count > 25:
        parts.append(f"{skill_count} skills listed (normal ≤ 20)")
    if expert_zero > 3:
        parts.append(f"{expert_zero} 'expert' skills with 0 months of use")
    if density > 0.12:
        parts.append(f"buzzword density {density:.1%} in profile text")
    description = "; ".join(parts) if parts else "No keyword stuffing detected"

    return _make_signal("keyword_stuffing", confidence, evidence, description)


def detect_fake_ai_profile(flat: dict) -> TrapSignal:
    """
    Detect synthetically generated or blatantly fabricated profiles.

    Hard honeypot signals (guaranteed fake if any fire)
    ---------------------------------------------------
    - salary_inverted: expected_salary_min > expected_salary_max

    Soft signals (accumulate red flags)
    ------------------------------------
    - profile_completeness == 100.0
    - All 3 verification booleans True
    - expert_zero_duration_count >= 8
    - github_activity_score == -1 despite high completeness
    - years_of_experience divisible by 5 AND >= 10
    - connection_count == exactly 500 (LinkedIn display cap)
    - Summary length < 20 words despite completeness >= 85
    """
    salary_inv   = bool(_g(flat, "salary_inverted",              False))
    completeness = float(_g(flat, "profile_completeness_score",  0.0))
    expert_zero  = int(_g(flat, "expert_zero_duration_count",    0))
    github       = float(_g(flat, "github_activity_score",       -1.0))
    yoe          = int(_g(flat, "years_of_experience",            0))
    connections  = int(_g(flat, "connection_count",               0))
    verified_all = (bool(_g(flat, "verified_email",   False))
                and bool(_g(flat, "verified_phone",   False))
                and bool(_g(flat, "linkedin_connected", False)))
    summary_text = str(_g(flat, "summary", ""))
    summary_len  = len(summary_text.split())

    evidence: dict[str, Any] = {}
    red_flags = 0

    # Hard honeypot — immediate high confidence
    if salary_inv:
        red_flags += 5
        evidence["salary_inverted"] = True

    if expert_zero >= 8:
        red_flags += 3
        evidence["expert_zero_duration_count"] = expert_zero
    elif expert_zero >= 4:
        red_flags += 1
        evidence["expert_zero_duration_count"] = expert_zero

    if completeness >= 100.0:
        red_flags += 1
        evidence["perfect_completeness"] = True

    if verified_all:
        red_flags += 1
        evidence["all_fields_verified"] = True

    if completeness >= 85 and (github < 0 or github < 5):
        red_flags += 2
        evidence["high_completeness_no_github"] = True

    if yoe >= 10 and yoe % 5 == 0:
        red_flags += 1
        evidence["round_experience_years"] = yoe

    if connections == 500:
        red_flags += 1
        evidence["connection_count_at_cap"] = True

    if completeness >= 85 and summary_len < 20:
        red_flags += 1
        evidence["thin_summary_words"] = summary_len

    max_flags = 11
    confidence = min(1.0, red_flags / max_flags)
    parts = []
    if salary_inv:
        parts.append("salary range is inverted (honeypot signal)")
    if expert_zero >= 4:
        parts.append(f"{expert_zero} expert-level skills with 0 months usage")
    if red_flags >= 4 and not parts:
        parts.append(f"{red_flags} profile anomaly flags detected")
    description = "; ".join(parts) if parts else "No fabrication signals detected"

    return _make_signal("fake_ai_profile", confidence, evidence, description)


def detect_generic_chatgpt_user(flat: dict) -> TrapSignal:
    """
    Detect summaries written by or heavily polished with ChatGPT.

    Matches a curated list of generic template phrases that appear
    frequently in AI-generated professional summaries.  3+ matches triggers.
    """
    summary = str(_g(flat, "summary", ""))
    if not summary.strip():
        return _make_signal(
            "generic_chatgpt_user", 0.0, {"phrase_hits": 0},
            "No summary text to analyse",
        )

    hits: list[str] = []
    for pat in _CHATGPT_PHRASES:
        if pat.search(summary):
            hits.append(pat.pattern)

    n = len(hits)
    # 3 phrases → low confidence, 6+ → high confidence
    confidence = min(1.0, max(0.0, (n - 2) / 6.0))
    evidence   = {"phrase_hits": n, "matched_phrases": hits[:5]}   # cap list at 5
    description = (
        f"Summary contains {n} generic template phrase(s): "
        + ", ".join(repr(h) for h in hits[:3])
        if hits else "No ChatGPT template phrases detected"
    )
    return _make_signal("generic_chatgpt_user", confidence, evidence, description)


def detect_research_only(flat: dict) -> TrapSignal:
    """
    Detect candidates who have exclusively academic or research backgrounds
    with no evidence of production software engineering.

    Signals
    -------
    - career_titles dominated by research / academic titles
    - career_descriptions_text has high academic keyword density
    - tier1_skill_count == 0 (no production AI/ML platform skills)
    - product_company_ratio == 0 (never worked at a product company)
    - is_consulting_only (consulting without product work)
    """
    career_titles     = list(_g(flat, "career_titles",         []))
    career_desc       = str(_g(flat, "career_descriptions_text", "")).lower()
    tier1_count       = int(_g(flat, "tier1_skill_count",       0))
    product_ratio     = float(_g(flat, "product_company_ratio", 0.5))
    n_roles           = int(_g(flat, "n_career_roles",          0))

    # Signal A: fraction of career titles that are research/academic
    def _is_research_title(t: str) -> bool:
        tl = t.lower()
        return any(kw in tl for kw in _RESEARCH_TITLES)

    research_title_count = sum(1 for t in career_titles if _is_research_title(t))
    research_title_ratio = research_title_count / max(n_roles, 1)

    # Signal B: academic keyword density in career text
    words = career_desc.split()
    academic_hits = sum(1 for w in words if w.strip(".,;:()") in _ACADEMIC_KEYWORDS
                        or any(kw in career_desc for kw in _ACADEMIC_KEYWORDS))
    # Faster: count phrase hits
    acad_hit_count = _count_set_hits(career_desc, _ACADEMIC_KEYWORDS)
    acad_density   = acad_hit_count / max(len(words), 1)

    # Signal C: no tier-1 production skills
    no_prod_skills = 1.0 if tier1_count == 0 else max(0.0, 1.0 - tier1_count / 3.0)

    # Signal D: never in a product company
    no_product_co  = 1.0 if product_ratio < 0.10 else max(0.0, 1.0 - product_ratio / 0.3)

    confidence = (
        0.35 * research_title_ratio
      + 0.25 * min(1.0, acad_density / 0.10)
      + 0.25 * no_prod_skills
      + 0.15 * no_product_co
    )
    evidence = {
        "research_title_ratio":  round(research_title_ratio, 3),
        "academic_keyword_hits": acad_hit_count,
        "tier1_skill_count":     tier1_count,
        "product_company_ratio": round(product_ratio, 3),
    }
    parts = []
    if research_title_ratio > 0.5:
        parts.append(f"{research_title_count}/{n_roles} career titles are research/academic")
    if tier1_count == 0:
        parts.append("no tier-1 production ML skills found")
    if product_ratio < 0.10:
        parts.append("no product company experience")
    description = "; ".join(parts) if parts else "No research-only signals detected"

    return _make_signal("research_only", confidence, evidence, description)


def detect_low_quality_profile(flat: dict) -> TrapSignal:
    """
    Detect thin, incomplete, or low-effort profiles.

    Signals
    -------
    - profile_completeness_score < 40
    - skill_count < 3
    - summary length < 20 words
    - tier1_skill_count == 0 AND tier2_skill_count == 0
    - cert_count == 0 AND n_career_roles == 0
    - verified_email == False
    - edu_count == 0
    """
    completeness = float(_g(flat, "profile_completeness_score",  50.0))
    skill_count  = int(_g(flat, "skill_count",                    5))
    tier1        = int(_g(flat, "tier1_skill_count",              0))
    tier2        = int(_g(flat, "tier2_skill_count",              0))
    cert_count   = int(_g(flat, "cert_count",                     0))
    n_roles      = int(_g(flat, "n_career_roles",                 0))
    email_ver    = bool(_g(flat, "verified_email",                True))
    edu_count    = int(_g(flat, "edu_count",                      1))
    summary      = str(_g(flat, "summary",                        ""))
    summary_len  = len(summary.split())

    flags = 0
    evidence: dict[str, Any] = {}

    if completeness < 40:
        flags += 2
        evidence["profile_completeness_score"] = completeness
    elif completeness < 60:
        flags += 1
        evidence["profile_completeness_score"] = completeness

    if skill_count < 3:
        flags += 2
        evidence["skill_count"] = skill_count

    if tier1 == 0 and tier2 == 0:
        flags += 2
        evidence["no_relevant_skills"] = True

    if summary_len < 20:
        flags += 1
        evidence["summary_word_count"] = summary_len

    if cert_count == 0 and n_roles == 0:
        flags += 2
        evidence["no_certs_no_career"] = True

    if not email_ver:
        flags += 1
        evidence["unverified_email"] = True

    if edu_count == 0:
        flags += 1
        evidence["no_education_listed"] = True

    max_flags = 11
    confidence = min(1.0, flags / max_flags)
    parts = []
    if completeness < 40:
        parts.append(f"profile {completeness:.0f}% complete")
    if skill_count < 3:
        parts.append(f"only {skill_count} skills listed")
    if tier1 == 0 and tier2 == 0:
        parts.append("no relevant AI/ML skills")
    if n_roles == 0:
        parts.append("no career history")
    description = "; ".join(parts) if parts else "Profile appears adequate"

    return _make_signal("low_quality_profile", confidence, evidence, description)


def detect_inactive_candidate(flat: dict) -> TrapSignal:
    """
    Detect candidates who are structurally inactive on the platform —
    going beyond the soft behavioral cap in behavioral_signals.py.

    Signals
    -------
    - days_since_last_active > 180 (6 months stale)
    - applications_submitted_30d == 0
    - profile_views_received_30d < 5
    - open_to_work_flag == False
    - search_appearance_30d == 0
    """
    days_inactive = int(_g(flat, "days_since_last_active",      0))
    apps_30d      = int(_g(flat, "applications_submitted_30d",  0))
    views_30d     = int(_g(flat, "profile_views_received_30d",  50))
    open_to_work  = bool(_g(flat, "open_to_work_flag",          True))
    appearances   = int(_g(flat, "search_appearance_30d",       10))

    # Score each signal [0,1]
    staleness_score = min(1.0, max(0.0, (days_inactive - 60) / 300.0))
    no_apps_score   = 1.0 if apps_30d == 0 else max(0.0, 1.0 - apps_30d / 5.0)
    no_views_score  = 1.0 if views_30d < 5 else max(0.0, 1.0 - views_30d / 30.0)
    not_seeking     = 1.0 if not open_to_work else 0.0
    no_appear_score = 1.0 if appearances == 0 else max(0.0, 1.0 - appearances / 20.0)

    confidence = (
        0.35 * staleness_score
      + 0.20 * no_apps_score
      + 0.15 * no_views_score
      + 0.20 * not_seeking
      + 0.10 * no_appear_score
    )
    evidence = {
        "days_since_last_active":      days_inactive,
        "applications_submitted_30d":  apps_30d,
        "profile_views_received_30d":  views_30d,
        "open_to_work_flag":           open_to_work,
        "search_appearance_30d":       appearances,
    }
    parts = []
    if days_inactive > 180:
        parts.append(f"inactive for {days_inactive} days")
    if not open_to_work and days_inactive > 90:
        parts.append("not open to work")
    if apps_30d == 0 and views_30d < 5:
        parts.append("no platform activity in 30 days")
    description = "; ".join(parts) if parts else "No structural inactivity detected"

    return _make_signal("inactive_candidate", confidence, evidence, description)


def detect_inconsistent_career(flat: dict) -> TrapSignal:
    """
    Detect career histories that show implausible progressions.

    Signals
    -------
    - Seniority regression: career titles go from senior → junior
    - High variance in seniority across career (erratic path)
    - n_career_roles / max(years_of_experience, 1) > 1.5 (too many jobs)
    - is_consulting_only with claims of deep ML expertise
    """
    career_titles = list(_g(flat, "career_titles",   []))
    n_roles       = int(_g(flat, "n_career_roles",    0))
    yoe           = int(_g(flat, "years_of_experience", 1))
    is_consulting = bool(_g(flat, "is_consulting_only", False))
    tier1         = int(_g(flat, "tier1_skill_count",   0))

    evidence: dict[str, Any] = {}
    signals: list[float] = []

    # Signal A: seniority regression
    if len(career_titles) >= 2:
        levels = [_title_seniority(t) for t in career_titles]
        # Count drops of > 1 level between consecutive roles
        regressions = sum(
            1 for i in range(1, len(levels))
            if levels[i] < levels[i - 1] - 1
        )
        regression_score = min(1.0, regressions / max(len(levels) - 1, 1))
        if regressions > 0:
            evidence["seniority_regressions"] = regressions
            evidence["seniority_levels"]      = levels
        signals.append(regression_score * 0.40)
    else:
        signals.append(0.0)

    # Signal B: seniority variance (erratic path)
    if len(career_titles) >= 3:
        levels = [_title_seniority(t) for t in career_titles]
        var    = float(np.var(levels))
        var_score = min(1.0, var / 4.0)   # variance of 4 = full score
        if var > 2.0:
            evidence["seniority_variance"] = round(var, 2)
        signals.append(var_score * 0.25)
    else:
        signals.append(0.0)

    # Signal C: too many jobs per year
    jobs_per_year = n_roles / max(yoe, 1)
    hopper_score  = min(1.0, max(0.0, (jobs_per_year - 1.5) / 1.5))
    if jobs_per_year > 1.5:
        evidence["jobs_per_year"] = round(jobs_per_year, 2)
    signals.append(hopper_score * 0.20)

    # Signal D: consulting-only with claimed ML expertise
    if is_consulting and tier1 >= 2:
        evidence["consulting_with_ml_claims"] = True
        signals.append(0.25)
    else:
        signals.append(0.0)

    # Signal E: no career titles despite claiming experience
    if yoe > 3 and n_roles == 0:
        evidence["yoe_no_history"] = yoe
        signals.append(0.15)
    else:
        signals.append(0.0)

    # Extreme job-hopping alone is a strong signal — apply a dominant floor
    confidence = min(1.0, max(sum(signals), hopper_score * 0.50))
    parts = []
    if evidence.get("seniority_regressions", 0) > 0:
        parts.append(f"{evidence['seniority_regressions']} seniority regressions in career path")
    if jobs_per_year > 1.5:
        parts.append(f"{n_roles} roles in {yoe} years ({jobs_per_year:.1f} jobs/yr)")
    if is_consulting and tier1 >= 2:
        parts.append("consulting-only background with production ML skill claims")
    description = "; ".join(parts) if parts else "Career history appears consistent"

    return _make_signal("inconsistent_career", confidence, evidence, description)


def detect_suspicious_timeline(flat: dict) -> TrapSignal:
    """
    Detect experience timeline anomalies.

    Signals
    -------
    - total_career_months > years_of_experience * 12 * 1.25  (impossible overlap)
    - years_of_experience > 0 AND n_career_roles == 0        (no verifiable history)
    - n_career_roles / max(years_of_experience, 0.5) > 2.5   (extreme job-hopping)
    - salary_inverted (honeypot; fabricated profile signal)
    - years_of_experience > 30 (implausible for AI/ML field)
    """
    total_months  = int(_g(flat, "total_career_months",   0))
    yoe           = int(_g(flat, "years_of_experience",   0))
    n_roles       = int(_g(flat, "n_career_roles",         0))
    salary_inv    = bool(_g(flat, "salary_inverted",       False))

    evidence: dict[str, Any] = {}
    score_parts: list[float] = []

    # Signal A: career months exceeds stated YoE by 25%
    claimed_months = yoe * 12
    if claimed_months > 0 and total_months > claimed_months * 1.25:
        overflow_ratio = (total_months - claimed_months) / max(claimed_months, 1)
        overflow_score = min(1.0, overflow_ratio / 0.5)
        evidence["total_career_months"]  = total_months
        evidence["claimed_months"]       = claimed_months
        evidence["overflow_ratio"]       = round(overflow_ratio, 2)
    else:
        overflow_score = 0.0
    score_parts.append(overflow_score * 0.30)

    # Signal B: stated YoE but zero career history
    ghost_score = 0.0
    if yoe >= 3 and n_roles == 0:
        ghost_score = min(1.0, yoe / 10.0)
        evidence["yoe_with_no_roles"] = yoe
    score_parts.append(ghost_score * 0.35)

    # Signal C: extreme job hopping
    jobs_per_year  = n_roles / max(yoe, 0.5)
    hopper_score   = min(1.0, max(0.0, (jobs_per_year - 2.5) / 2.5))
    if jobs_per_year > 2.5:
        evidence["extreme_jobs_per_year"] = round(jobs_per_year, 2)
    score_parts.append(hopper_score * 0.20)

    # Signal D: salary inverted honeypot
    if salary_inv:
        evidence["salary_inverted"] = True
        score_parts.append(0.15)
    else:
        score_parts.append(0.0)

    # Signal E: implausibly long career in AI/ML
    if yoe > 30:
        evidence["implausible_yoe"] = yoe
        score_parts.append(0.0)   # low weight — uncommon but not impossible

    base_conf = sum(score_parts)
    # A single dominant anomaly (impossible overflow, ghost YoE, extreme hopping)
    # should clear the threshold on its own
    dominant  = max(overflow_score, ghost_score, hopper_score)
    confidence = min(1.0, max(base_conf, dominant * 0.60))
    parts = []
    if total_months > claimed_months * 1.25 and claimed_months > 0:
        parts.append(
            f"career spans {total_months} months but YoE implies ≤ {claimed_months}"
        )
    if yoe >= 3 and n_roles == 0:
        parts.append(f"claims {yoe} years experience but no listed positions")
    if jobs_per_year > 2.5:
        parts.append(f"extreme job-hopping ({jobs_per_year:.1f} jobs/year)")
    if salary_inv:
        parts.append("salary range inverted (fabrication signal)")
    description = "; ".join(parts) if parts else "Timeline appears plausible"

    return _make_signal("suspicious_timeline", confidence, evidence, description)


def detect_ai_keywords_no_production(flat: dict) -> TrapSignal:
    """
    Detect profiles rich in AI buzzwords but lacking any evidence of
    deploying models to production systems.

    High AI keyword count + zero production keywords = theory-only candidate.

    Adjusted for seniority: juniors with 0–2 years get a leniency factor.
    """
    text = _text(flat)
    yoe  = int(_g(flat, "years_of_experience", 0))

    ai_hits   = _count_set_hits(text, _AI_BUZZWORDS)
    prod_hits = _count_set_hits(text, _PRODUCTION_KEYWORDS)

    # Normalise AI keyword count (cap at 25 = full score)
    ai_score   = min(1.0, ai_hits / 15.0)
    # Production signal: 0 hits = no evidence; 5+ = strong
    prod_score = min(1.0, prod_hits / 5.0)
    gap        = max(0.0, ai_score - prod_score)

    # Leniency: juniors (< 2 years) are not expected to have prod experience
    leniency   = 0.4 if yoe <= 2 else 0.0
    confidence = max(0.0, gap - leniency)

    evidence = {
        "ai_keyword_hits":         ai_hits,
        "production_keyword_hits": prod_hits,
        "leniency_applied":        leniency > 0,
    }
    if confidence >= _THRESHOLDS["ai_keywords_no_production"]:
        description = (
            f"Profile mentions {ai_hits} AI concepts but only "
            f"{prod_hits} production deployment signals"
        )
    else:
        description = (
            f"AI ({ai_hits}) and production ({prod_hits}) keyword counts balanced"
        )
    return _make_signal("ai_keywords_no_production", confidence, evidence, description)


def detect_behavioral_trust_issues(flat: dict) -> TrapSignal:
    """
    Detect candidates with platform behaviour that undermines trust.

    Signals
    -------
    - recruiter_response_rate < 0.05  (ghosts recruiters)
    - interview_completion_rate < 0.20 (bails on interviews)
    - offer_acceptance_rate < 0.10 AND >= 0 (declines all offers)
    - github == -1 AND linkedin_connected == False  (no verifiable online presence)
    - salary_inverted  (honeypot)
    - saved_by_recruiters_30d == 0 AND years_of_experience > 5 (senior but ignored)
    - connection_count < 10 AND years_of_experience > 3 (oddly isolated)
    """
    response_rate  = float(_g(flat, "recruiter_response_rate",   0.5))
    interview_rate = float(_g(flat, "interview_completion_rate", 0.5))
    acceptance     = float(_g(flat, "offer_acceptance_rate",     -1.0))
    github         = float(_g(flat, "github_activity_score",     -1.0))
    linkedin       = bool(_g(flat, "linkedin_connected",          True))
    salary_inv     = bool(_g(flat, "salary_inverted",             False))
    saved          = int(_g(flat, "saved_by_recruiters_30d",      1))
    connections    = int(_g(flat, "connection_count",             50))
    yoe            = int(_g(flat, "years_of_experience",          0))

    evidence: dict[str, Any] = {}
    flag_score = 0.0

    if response_rate < 0.05:
        evidence["recruiter_response_rate"] = response_rate
        flag_score += 0.25

    if interview_rate < 0.20:
        evidence["interview_completion_rate"] = interview_rate
        flag_score += 0.20

    if acceptance >= 0 and acceptance < 0.10:
        evidence["offer_acceptance_rate"] = acceptance
        flag_score += 0.15

    if github < 0 and not linkedin:
        evidence["no_online_presence"] = True
        flag_score += 0.15

    if salary_inv:
        evidence["salary_inverted"] = True
        flag_score += 0.10

    if yoe > 5 and saved == 0:
        evidence["senior_no_recruiter_saves"] = {"yoe": yoe, "saved": saved}
        flag_score += 0.10

    if yoe > 3 and connections < 10:
        evidence["isolated_senior"] = {"yoe": yoe, "connections": connections}
        flag_score += 0.05

    confidence = min(1.0, flag_score)
    parts = []
    if response_rate < 0.05:
        parts.append(f"recruiter response rate only {response_rate:.0%}")
    if interview_rate < 0.20:
        parts.append(f"interview completion only {interview_rate:.0%}")
    if acceptance >= 0 and acceptance < 0.10:
        parts.append(f"accepts only {acceptance:.0%} of offers")
    if github < 0 and not linkedin:
        parts.append("no GitHub or LinkedIn presence")
    description = "; ".join(parts) if parts else "No behavioral trust issues detected"

    return _make_signal("behavioral_trust_issues", confidence, evidence, description)


# ─────────────────────────────────────────────────────────────────────────────
# § 5  Main scoring functions
# ─────────────────────────────────────────────────────────────────────────────

_DETECTORS = [
    detect_keyword_stuffing,
    detect_fake_ai_profile,
    detect_generic_chatgpt_user,
    detect_research_only,
    detect_low_quality_profile,
    detect_inactive_candidate,
    detect_inconsistent_career,
    detect_suspicious_timeline,
    detect_ai_keywords_no_production,
    detect_behavioral_trust_issues,
]
assert len(_DETECTORS) == len(TRAP_NAMES)


def run_all_detectors(flat: dict[str, Any]) -> list[TrapSignal]:
    """Run all 10 detectors and return the list of TrapSignals."""
    return [fn(flat) for fn in _DETECTORS]


def detect_traps(flat: dict[str, Any]) -> dict[str, Any]:
    """
    Main entry point.  Run all 10 trap detectors on a flattened candidate dict.

    Returns
    -------
    {
        "trap_risk_score" : float [0, 1],   # severity-weighted mean of triggered confidences
        "trap_labels"     : list[str],       # names of triggered traps
        "trap_penalty"    : float [0, 1],   # compound multiplicative penalty
        "explanation"     : list[dict],      # per-trap breakdown (all 10, triggered or not)
    }
    """
    signals  = run_all_detectors(flat)
    triggered = [s for s in signals if s.triggered]

    # ── trap_risk_score ───────────────────────────────────────────────────────
    if triggered:
        total_weight = sum(1.0 - s.penalty_factor for s in triggered)
        if total_weight > 1e-9:
            trap_risk_score = sum(
                s.confidence * (1.0 - s.penalty_factor) for s in triggered
            ) / total_weight
        else:
            trap_risk_score = sum(s.confidence for s in triggered) / len(triggered)
    else:
        trap_risk_score = 0.0

    # ── trap_penalty ──────────────────────────────────────────────────────────
    trap_penalty = 1.0
    for s in triggered:
        trap_penalty *= s.penalty_factor
    trap_penalty = max(MIN_COMPOUND_PENALTY, trap_penalty)

    return {
        "trap_risk_score": round(float(trap_risk_score), 4),
        "trap_labels":     [s.name for s in triggered],
        "trap_penalty":    round(float(trap_penalty),    4),
        "explanation":     [s.as_dict() for s in signals],
    }


# ─────────────────────────────────────────────────────────────────────────────
# § 6  Batch report
# ─────────────────────────────────────────────────────────────────────────────

def build_trap_report(
    flat_rows:      list[dict[str, Any]],
    *,
    show_progress:  bool = False,
) -> pd.DataFrame:
    """
    Run detect_traps on every candidate and return a DataFrame with one row
    per candidate.

    Columns
    -------
    candidate_id, trap_risk_score, trap_penalty, trap_count,
    <trap_name>_triggered  (one bool column per trap)
    """
    rows = []
    iterable = tqdm(flat_rows, desc="trap scan") if show_progress else flat_rows
    for flat in iterable:
        result = detect_traps(flat)
        row: dict[str, Any] = {
            "candidate_id":    flat.get("candidate_id", ""),
            "trap_risk_score": result["trap_risk_score"],
            "trap_penalty":    result["trap_penalty"],
            "trap_count":      len(result["trap_labels"]),
        }
        for name in TRAP_NAMES:
            row[f"{name}_triggered"] = name in result["trap_labels"]
        rows.append(row)

    df = pd.DataFrame(rows)
    if "candidate_id" in df.columns:
        df = df.set_index("candidate_id")
    bool_cols = [c for c in df.columns if c.endswith("_triggered")]
    df[bool_cols] = df[bool_cols].astype(bool)
    return df
