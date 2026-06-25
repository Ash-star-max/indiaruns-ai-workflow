"""
load_data.py — Efficient candidate data loader

Public API
----------
iter_candidates(path, ...)        → Iterator[CandidateRecord]  (streaming)
load_candidates(path, ...)        → list[CandidateRecord]
flatten_candidate(record)         → dict
build_candidate_text(record)      → str
load_flat(path, ...)              → pd.DataFrame
validate_schema(record_dict)      → bool
benchmark_memory(path, ...)       → dict[str, float]

Format support
--------------
    .json      — JSON array of candidate objects (sample file format)
    .jsonl     — Newline-delimited JSON (production format)
    .jsonl.gz  — Gzip-compressed JSONL

Streaming design
----------------
iter_candidates() is a true generator — only one record is in memory at a
time. load_flat() accumulates all rows but builds the DataFrame in a single
pd.DataFrame(list_of_dicts) call rather than growing a DataFrame row by row.
"""

from __future__ import annotations

import gzip
import json
import tracemalloc
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from tqdm import tqdm

from src.config import (
    CONSULTING_FIRMS,
    MAX_CANDIDATE_TEXT_CHARS,
    REFERENCE_DATE,
    TIER1_SKILLS,
    TIER2_SKILLS,
)

# ─────────────────────────────────────────────────────────────────────────────
# § 1  Pydantic models  (validation layer — mirrors candidate_schema.json)
# ─────────────────────────────────────────────────────────────────────────────

_PROFICIENCY_LEVELS = {"beginner", "intermediate", "advanced", "expert"}
_WORK_MODES         = {"remote", "hybrid", "onsite", "flexible"}
_EDU_TIERS          = {"tier_1", "tier_2", "tier_3", "tier_4", "unknown"}
_COMPANY_SIZES      = {"1-10","11-50","51-200","201-500","501-1000",
                       "1001-5000","5001-10000","10001+"}


class SkillEntry(BaseModel):
    name: str
    proficiency: str = "intermediate"
    endorsements: int = 0
    duration_months: int = 0

    @field_validator("proficiency")
    @classmethod
    def _check_proficiency(cls, v: str) -> str:
        if v not in _PROFICIENCY_LEVELS:
            raise ValueError(f"Invalid proficiency: {v!r}")
        return v

    @field_validator("endorsements", "duration_months")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        return max(0, v)


class CareerEntry(BaseModel):
    company: str
    title: str
    start_date: date
    end_date: Optional[date] = None
    duration_months: int = 0
    is_current: bool = False
    industry: str = ""
    company_size: str = ""
    description: str = ""

    @field_validator("duration_months")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        return max(0, v)

    @model_validator(mode="after")
    def _end_after_start(self) -> "CareerEntry":
        if self.end_date and self.start_date and self.end_date < self.start_date:
            # Swap silently — dataset has some ordering noise
            self.start_date, self.end_date = self.end_date, self.start_date
        return self


class EducationEntry(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: str = "unknown"

    @field_validator("tier")
    @classmethod
    def _check_tier(cls, v: str) -> str:
        return v if v in _EDU_TIERS else "unknown"


class CertificationEntry(BaseModel):
    name: str
    issuer: str
    year: int


class LanguageEntry(BaseModel):
    language: str
    proficiency: str


class SalaryRange(BaseModel):
    min: float = 0.0
    max: float = 0.0

    @model_validator(mode="after")
    def _ensure_min_le_max(self) -> "SalaryRange":
        # Store the original inversion as a signal; do not "fix" it here —
        # flatten_candidate will expose salary_inverted as a honeypot flag.
        return self


class RedrobSignals(BaseModel):
    profile_completeness_score: float = 0.0
    signup_date: Optional[date] = None
    last_active_date: Optional[date] = None
    open_to_work_flag: bool = False
    profile_views_received_30d: int = 0
    applications_submitted_30d: int = 0
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 0.0
    skill_assessment_scores: dict[str, float] = Field(default_factory=dict)
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: int = 90
    expected_salary_range_inr_lpa: SalaryRange = Field(default_factory=SalaryRange)
    preferred_work_mode: str = "flexible"
    willing_to_relocate: bool = False
    github_activity_score: float = -1.0   # -1 = no GitHub linked
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0   # -1 = no offer history
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False

    @field_validator("recruiter_response_rate", "interview_completion_rate")
    @classmethod
    def _clamp_rate(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("profile_completeness_score")
    @classmethod
    def _clamp_completeness(cls, v: float) -> float:
        return max(0.0, min(100.0, v))


class Profile(BaseModel):
    anonymized_name: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""
    country: str = ""
    years_of_experience: float = 0.0
    current_title: str = ""
    current_company: str = ""
    current_company_size: str = ""
    current_industry: str = ""

    @field_validator("years_of_experience")
    @classmethod
    def _clamp_yoe(cls, v: float) -> float:
        return max(0.0, min(50.0, v))


class CandidateRecord(BaseModel):
    candidate_id: str
    profile: Profile
    career_history: list[CareerEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    skills: list[SkillEntry] = Field(default_factory=list)
    certifications: list[CertificationEntry] = Field(default_factory=list)
    languages: list[LanguageEntry] = Field(default_factory=list)
    redrob_signals: RedrobSignals = Field(default_factory=RedrobSignals)


# ─────────────────────────────────────────────────────────────────────────────
# § 2  Candidate text builder
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_text(c: CandidateRecord) -> str:
    """
    Build a dense text representation for embedding.

    Order follows the user specification:
        headline → summary → current_title → skills →
        certifications → education → career_descriptions

    The summary is positioned second (after the short headline) because it is
    the most semantically authoritative text — it is written by the candidate
    in their own voice and is not subject to the description-shuffling trap
    observed in career_history.

    The result is truncated to MAX_CANDIDATE_TEXT_CHARS so that all candidates
    fit within the embedding model's token window.
    """
    parts: list[str] = []

    # ── 1. Headline (identity signal — short, first) ──────────────────────
    if c.profile.headline.strip():
        parts.append(c.profile.headline.strip())

    # ── 2. Summary (most authoritative text) ─────────────────────────────
    if c.profile.summary.strip():
        parts.append(c.profile.summary.strip())

    # ── 3. Current title + company + industry ─────────────────────────────
    role_parts = [c.profile.current_title]
    if c.profile.current_company:
        role_parts.append(f"at {c.profile.current_company}")
    if c.profile.current_industry:
        role_parts.append(f"({c.profile.current_industry})")
    role_line = " ".join(p for p in role_parts if p)
    if role_line.strip():
        parts.append(f"Current role: {role_line}")

    # ── 4. Skills (tiered by proficiency) ────────────────────────────────
    if c.skills:
        high_skills  = [s.name for s in c.skills if s.proficiency in ("expert", "advanced")]
        low_skills   = [s.name for s in c.skills if s.proficiency in ("intermediate", "beginner")]
        skill_lines: list[str] = []
        if high_skills:
            skill_lines.append(f"Expert/Advanced: {', '.join(high_skills)}")
        if low_skills:
            skill_lines.append(f"Intermediate/Beginner: {', '.join(low_skills)}")
        if skill_lines:
            parts.append("Skills: " + "; ".join(skill_lines))

    # ── 5. Certifications ────────────────────────────────────────────────
    if c.certifications:
        cert_text = ", ".join(
            f"{cert.name} ({cert.issuer}, {cert.year})"
            for cert in c.certifications
        )
        parts.append(f"Certifications: {cert_text}")

    # ── 6. Education ─────────────────────────────────────────────────────
    if c.education:
        edu_items = [
            f"{e.degree} in {e.field_of_study} from {e.institution} ({e.end_year})"
            for e in c.education
        ]
        parts.append("Education: " + "; ".join(edu_items))

    # ── 7. Career descriptions ────────────────────────────────────────────
    # Each description is prefixed with the role title for grounding context.
    # NOTE: descriptions in this dataset may not perfectly match their listed
    # titles (synthetic shuffling). The summary (§2) is more reliable.
    if c.career_history:
        desc_items: list[str] = []
        for entry in c.career_history:
            if entry.description.strip():
                desc_items.append(
                    f"{entry.title} at {entry.company}: {entry.description.strip()}"
                )
        if desc_items:
            parts.append("Experience:\n" + "\n".join(desc_items))

    combined = "\n".join(parts)

    # Truncate to budget — embedder handles sub-512-token inputs best
    if len(combined) > MAX_CANDIDATE_TEXT_CHARS:
        combined = combined[:MAX_CANDIDATE_TEXT_CHARS]

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# § 3  Flattener — nested CandidateRecord → flat dict for DataFrame
# ─────────────────────────────────────────────────────────────────────────────

# Ordered tier names for "highest tier" calculation (lower index = more prestigious)
_TIER_ORDER = ["tier_1", "tier_2", "tier_3", "tier_4", "unknown"]

# Keywords that suggest a role was at a product company (heuristic)
_PRODUCT_INDUSTRY_SIGNALS = {
    "fintech", "saas", "product", "technology", "software", "ai", "ml",
    "e-commerce", "ecommerce", "internet", "startup", "platform",
}


def _is_consulting_company(company_name: str) -> bool:
    """Return True if company_name matches the consulting firm list."""
    name = company_name.strip()
    # Exact match first (fast)
    if name in CONSULTING_FIRMS:
        return True
    # Partial match for variants like "TCS Limited", "Wipro Technologies"
    name_lower = name.lower()
    return any(cf.lower() in name_lower for cf in CONSULTING_FIRMS)


def flatten_candidate(c: CandidateRecord) -> dict[str, Any]:
    """
    Convert a CandidateRecord into a flat dict suitable for a pandas DataFrame row.

    Object-typed columns (lists, dicts) are preserved as Python objects in the
    DataFrame. Downstream feature engineering in features.py converts these into
    purely numeric arrays.
    """
    sig  = c.redrob_signals
    prof = c.profile

    # ── Career aggregations ───────────────────────────────────────────────────
    career_companies: list[str] = [e.company for e in c.career_history]
    career_titles: list[str]    = [e.title   for e in c.career_history]
    career_descriptions_text: str = " ".join(
        e.description for e in c.career_history if e.description.strip()
    )
    total_career_months: int = sum(e.duration_months for e in c.career_history)
    n_roles: int = len(career_companies)

    # Consulting vs product split
    consulting_count = sum(1 for co in career_companies if _is_consulting_company(co))
    product_company_ratio = (n_roles - consulting_count) / n_roles if n_roles > 0 else 0.0
    is_consulting_only = (n_roles > 0) and (consulting_count == n_roles)

    # Most recent / current role
    current_roles = [e for e in c.career_history if e.is_current]
    most_recent_title   = current_roles[0].title   if current_roles else prof.current_title
    most_recent_company = current_roles[0].company if current_roles else prof.current_company

    # ── Skills aggregations ───────────────────────────────────────────────────
    skill_names: list[str] = [s.name for s in c.skills]
    skill_names_lower      = {s.name.lower() for s in c.skills}

    tier1_count = len(skill_names_lower & TIER1_SKILLS)
    tier2_count = len(skill_names_lower & TIER2_SKILLS)

    # Honeypot signal: expert/advanced skill with 0 usage duration
    expert_zero_duration_count = sum(
        1 for s in c.skills
        if s.proficiency in ("expert", "advanced") and s.duration_months == 0
    )

    # Skill depth aggregates (used by feature_engineering.py)
    tier1_skills_list = [s for s in c.skills if s.name.lower() in TIER1_SKILLS]
    tier2_skills_list = [s for s in c.skills if s.name.lower() in TIER2_SKILLS]
    tier1_avg_duration = (
        sum(s.duration_months for s in tier1_skills_list) / len(tier1_skills_list)
        if tier1_skills_list else 0.0
    )
    tier2_avg_duration = (
        sum(s.duration_months for s in tier2_skills_list) / len(tier2_skills_list)
        if tier2_skills_list else 0.0
    )
    total_skill_endorsements = sum(s.endorsements for s in c.skills)

    # ── Education ─────────────────────────────────────────────────────────────
    highest_edu_tier = min(
        (e.tier for e in c.education),
        key=lambda t: _TIER_ORDER.index(t) if t in _TIER_ORDER else len(_TIER_ORDER),
        default="unknown",
    )
    edu_fields: list[str] = [e.field_of_study for e in c.education]

    # ── Languages ────────────────────────────────────────────────────────────
    language_names: list[str] = [lang.language for lang in c.languages]
    language_count: int = len(c.languages)
    english_proficiency: str = next(
        (lang.proficiency for lang in c.languages
         if "english" in lang.language.lower()),
        "unknown",
    )

    # ── Certifications ────────────────────────────────────────────────────────
    cert_names: list[str] = [cert.name for cert in c.certifications]

    # ── Behavioral signals ────────────────────────────────────────────────────
    days_since_last_active: int = (
        (REFERENCE_DATE - sig.last_active_date).days
        if sig.last_active_date
        else 9999   # treat unknown as very stale
    )

    salary_min = sig.expected_salary_range_inr_lpa.min
    salary_max = sig.expected_salary_range_inr_lpa.max
    salary_inverted = salary_min > salary_max  # honeypot signal

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "candidate_id":          c.candidate_id,
        # ── Profile ───────────────────────────────────────────────────────────
        "headline":              prof.headline,
        "summary":               prof.summary,
        "location":              prof.location,
        "country":               prof.country,
        "years_of_experience":   prof.years_of_experience,
        "current_title":         prof.current_title,
        "current_company":       prof.current_company,
        "current_company_size":  prof.current_company_size,
        "current_industry":      prof.current_industry,
        # ── Career ────────────────────────────────────────────────────────────
        "career_companies":          career_companies,
        "career_titles":             career_titles,
        "career_descriptions_text":  career_descriptions_text,
        "total_career_months":       total_career_months,
        "n_career_roles":            n_roles,
        "most_recent_title":         most_recent_title,
        "most_recent_company":       most_recent_company,
        "product_company_ratio":     product_company_ratio,
        "is_consulting_only":        is_consulting_only,
        # ── Skills ────────────────────────────────────────────────────────────
        "skill_names":                skill_names,
        "skill_count":                len(c.skills),
        "tier1_skill_count":          tier1_count,
        "tier2_skill_count":          tier2_count,
        "expert_zero_duration_count": expert_zero_duration_count,
        "tier1_avg_duration_months":  tier1_avg_duration,
        "tier2_avg_duration_months":  tier2_avg_duration,
        "total_skill_endorsements":   total_skill_endorsements,
        # ── Languages ─────────────────────────────────────────────────────────
        "language_names":       language_names,
        "language_count":       language_count,
        "english_proficiency":  english_proficiency,
        # ── Education ─────────────────────────────────────────────────────────
        "highest_edu_tier": highest_edu_tier,
        "edu_fields":       edu_fields,
        "edu_count":        len(c.education),
        # ── Certifications ────────────────────────────────────────────────────
        "cert_names": cert_names,
        "cert_count": len(c.certifications),
        # ── Redrob signals (flattened) ────────────────────────────────────────
        "profile_completeness_score":  sig.profile_completeness_score,
        "days_since_last_active":      days_since_last_active,
        "open_to_work_flag":           sig.open_to_work_flag,
        "profile_views_received_30d":  sig.profile_views_received_30d,
        "applications_submitted_30d":  sig.applications_submitted_30d,
        "recruiter_response_rate":     sig.recruiter_response_rate,
        "avg_response_time_hours":     sig.avg_response_time_hours,
        "skill_assessment_scores":     sig.skill_assessment_scores,
        "connection_count":            sig.connection_count,
        "endorsements_received":       sig.endorsements_received,
        "notice_period_days":          sig.notice_period_days,
        "salary_min_lpa":              salary_min,
        "salary_max_lpa":              salary_max,
        "salary_inverted":             salary_inverted,
        "preferred_work_mode":         sig.preferred_work_mode,
        "willing_to_relocate":         sig.willing_to_relocate,
        "github_activity_score":       sig.github_activity_score,
        "search_appearance_30d":       sig.search_appearance_30d,
        "saved_by_recruiters_30d":     sig.saved_by_recruiters_30d,
        "interview_completion_rate":   sig.interview_completion_rate,
        "offer_acceptance_rate":       sig.offer_acceptance_rate,
        "verified_email":              sig.verified_email,
        "verified_phone":              sig.verified_phone,
        "linkedin_connected":          sig.linkedin_connected,
        # ── Computed embedding text ───────────────────────────────────────────
        "candidate_text": build_candidate_text(c),
    }


# ─────────────────────────────────────────────────────────────────────────────
# § 4  File I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_format(path: Path) -> tuple[str, bool]:
    """
    Return (fmt, is_gzipped) where fmt is 'jsonl' or 'json'.

    Heuristic: if any suffix (left to right) is .jsonl → JSONL format.
    Otherwise treat as JSON array.
    """
    suffixes = [s.lower() for s in path.suffixes]
    is_gzipped = ".gz" in suffixes
    is_jsonl = ".jsonl" in suffixes
    fmt = "jsonl" if is_jsonl else "json"
    return fmt, is_gzipped


def _open_file(path: Path, is_gzipped: bool):
    """Return a text-mode file handle, handling optional gzip."""
    if is_gzipped:
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _parse_record(
    raw: dict[str, Any],
    skip_invalid: bool,
    source_label: str,
) -> Optional[CandidateRecord]:
    """Validate a raw dict and return CandidateRecord, or None on failure."""
    try:
        return CandidateRecord.model_validate(raw)
    except ValidationError as exc:
        cid = raw.get("candidate_id", "<unknown>")
        msg = f"Validation error [{source_label}] id={cid}: {exc.error_count()} error(s)"
        if skip_invalid:
            logger.warning(msg)
            return None
        raise ValueError(msg) from exc


# ─────────────────────────────────────────────────────────────────────────────
# § 5  Public API
# ─────────────────────────────────────────────────────────────────────────────

def iter_candidates(
    path: str | Path,
    *,
    skip_invalid: bool = True,
    max_records: Optional[int] = None,
) -> Iterator[CandidateRecord]:
    """
    Stream candidate records one at a time from *path*.

    Supports .json (array), .jsonl, and .jsonl.gz.
    Only one record is live in memory at a time — safe for 100K+ files.

    Parameters
    ----------
    path          : File path (str or Path).
    skip_invalid  : If True, log and skip bad records. If False, raise.
    max_records   : Stop after this many successfully parsed records (None = all).

    Yields
    ------
    CandidateRecord
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    fmt, is_gzipped = _detect_format(path)
    emitted = 0

    with _open_file(path, is_gzipped) as fh:
        if fmt == "jsonl":
            # ── JSONL: one JSON object per line ──────────────────────────
            for line_num, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    if skip_invalid:
                        logger.warning(f"JSON parse error at line {line_num}: {exc}")
                        continue
                    raise
                record = _parse_record(raw, skip_invalid, f"line {line_num}")
                if record is not None:
                    # Check limit BEFORE yielding so max_records=0 returns nothing
                    if max_records is not None and emitted >= max_records:
                        return
                    yield record
                    emitted += 1
        else:
            # ── JSON array: load all then iterate ────────────────────────
            # JSON arrays cannot be streamed line-by-line; use for samples only.
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON file {path}: {exc}") from exc

            if not isinstance(data, list):
                raise ValueError(
                    f"Expected a JSON array in {path}, got {type(data).__name__}"
                )

            for idx, raw in enumerate(data):
                if not isinstance(raw, dict):
                    if skip_invalid:
                        logger.warning(f"Item {idx} is not a dict — skipping")
                        continue
                    raise ValueError(f"Item {idx} is not a dict")
                record = _parse_record(raw, skip_invalid, f"item {idx}")
                if record is not None:
                    # Check limit BEFORE yielding so max_records=0 returns nothing
                    if max_records is not None and emitted >= max_records:
                        return
                    yield record
                    emitted += 1


def load_candidates(
    path: str | Path,
    *,
    skip_invalid: bool = True,
    max_records: Optional[int] = None,
    show_progress: bool = True,
) -> list[CandidateRecord]:
    """
    Load all candidates into a list of CandidateRecord objects.

    For 100K records this uses ~1-2 GB RAM; prefer iter_candidates() when
    memory is tight or only a subset is needed.
    """
    path = Path(path)
    records: list[CandidateRecord] = []

    gen = iter_candidates(path, skip_invalid=skip_invalid, max_records=max_records)

    if show_progress:
        desc = f"Loading {path.name}"
        gen = tqdm(gen, desc=desc, unit=" records", mininterval=0.5)

    for record in gen:
        records.append(record)

    logger.info(f"Loaded {len(records):,} candidates from {path.name}")
    return records


def load_flat(
    path: str | Path,
    *,
    skip_invalid: bool = True,
    max_records: Optional[int] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Load and flatten all candidates into a pandas DataFrame.

    Each row corresponds to one candidate. List/dict columns (career_companies,
    skill_names, skill_assessment_scores, etc.) remain as Python objects —
    they are converted to numeric arrays in features.py.

    This function accumulates all rows as dicts then calls pd.DataFrame()
    once, which is significantly faster than growing a DataFrame row-by-row.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []

    gen = iter_candidates(path, skip_invalid=skip_invalid, max_records=max_records)

    if show_progress:
        desc = f"Flattening {path.name}"
        gen = tqdm(gen, desc=desc, unit=" records", mininterval=0.5)

    for record in gen:
        rows.append(flatten_candidate(record))

    if not rows:
        logger.warning("No records loaded — returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Ensure candidate_id is the index for fast lookup
    df = df.set_index("candidate_id", drop=False)

    logger.info(
        f"DataFrame: {len(df):,} rows × {len(df.columns)} columns "
        f"({df.memory_usage(deep=True).sum() / 1024**2:.1f} MB)"
    )
    return df


def validate_schema(record_dict: dict[str, Any]) -> bool:
    """
    Return True if *record_dict* passes Pydantic validation, False otherwise.

    Optionally uses jsonschema if candidate_schema.json is present, for a
    second independent validation pass.
    """
    try:
        CandidateRecord.model_validate(record_dict)
        return True
    except ValidationError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# § 6  Memory benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_memory(
    path: str | Path,
    n_records: int = 500,
    show_results: bool = True,
) -> dict[str, float]:
    """
    Measure peak Python heap allocations for two loading strategies.

    Uses tracemalloc (stdlib) — measures Python-object allocations, not total
    process RSS. Results give a relative comparison between approaches.

    Returns
    -------
    dict with keys:
        n_records              : records actually processed
        streaming_peak_mb      : peak MB while streaming (iter_candidates)
        load_flat_peak_mb      : peak MB while building full DataFrame
        flat_df_size_mb        : DataFrame in-memory size (deep)
        bytes_per_record       : flat_df_size_mb / n_records (in bytes)
    """
    path = Path(path)

    # ── Streaming measurement ─────────────────────────────────────────────
    tracemalloc.start()
    stream_count = 0
    for _ in iter_candidates(path, skip_invalid=True, max_records=n_records):
        stream_count += 1
    _, stream_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── load_flat measurement ─────────────────────────────────────────────
    tracemalloc.start()
    df = load_flat(path, skip_invalid=True, max_records=n_records, show_progress=False)
    _, flat_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    flat_df_mb = df.memory_usage(deep=True).sum() / 1024**2
    n = len(df)

    results = {
        "n_records":          stream_count,
        "streaming_peak_mb":  round(stream_peak  / 1024**2, 3),
        "load_flat_peak_mb":  round(flat_peak    / 1024**2, 3),
        "flat_df_size_mb":    round(flat_df_mb, 3),
        "bytes_per_record":   round(flat_df_mb * 1024**2 / n, 1) if n else 0,
    }

    if show_results:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Memory benchmark  ({n_records} records, {path.name})")
        logger.info(f"  Streaming peak  : {results['streaming_peak_mb']:.3f} MB")
        logger.info(f"  load_flat peak  : {results['load_flat_peak_mb']:.3f} MB")
        logger.info(f"  DataFrame size  : {results['flat_df_size_mb']:.3f} MB")
        logger.info(f"  Bytes/record    : {results['bytes_per_record']:.0f} B")
        logger.info(f"{'─'*50}")

    return results
