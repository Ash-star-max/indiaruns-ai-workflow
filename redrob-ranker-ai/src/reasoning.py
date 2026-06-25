"""
reasoning.py — Natural-language explanations for candidate rankings.

Generates 1–2 sentence summaries that are:
  • grounded only in verifiable profile facts (no hallucinations)
  • tone-calibrated to rank position (top-10 vs borderline)
  • naming the candidate's strongest attributes relative to the JD
  • surfacing concerns (trap signals, inactivity, gaps) when relevant
  • naturally varied across candidates via deterministic template selection

RANK-BAND STRATEGY
────────────────────────────────────────────────────────────────────────────────
  Rank  1–10  : Lead with specific tech or system named; pure-strength framing.
  Rank 11–30  : Strength + one concern (even mild; why they didn't crack top 10).
  Rank 31–60  : Partial match framing + clearest skill or profile gap.
  Rank 61–100 : Honest borderline framing + disqualifying or limiting signal.
  No rank     : Uses general strength/concern logic.

FACT-GROUNDING CONTRACT
────────────────────────────────────────────────────────────────────────────────
Every claim in the generated text must be backed by one of:
  • A specific value from flat (years_of_experience, skill_names, location, …)
  • A score from score_result.score_breakdown (used only to SELECT which fact
    to highlight, never to assert capability beyond what the score measures)
  • A trap_label from score_result.score_breakdown["trap_detail"]["trap_labels"]

Claims that are NEVER made:
  • Specific performance numbers ("improved latency by 40%") — not in flat
  • Named companies or universities (only tiers are scored)
  • Team sizes, funding stages, or headcount
  • Anything absent from the flat-dict schema

PUBLIC API
────────────────────────────────────────────────────────────────────────────────
    generate_explanation(flat, score_result, *, rank) -> str
    generate_explanations(flat_rows, score_results, *, ranks) -> list[str]
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from src.config import PREFERRED_CITIES
from src.scoring import CandidateScore, score_candidate

# ─────────────────────────────────────────────────────────────────────────────
# § 1  JD-relevant skill vocabulary
#      Matched case-insensitively against flat["skill_names"] to surface the
#      most informative skills in explanations.
# ─────────────────────────────────────────────────────────────────────────────

_JD_SKILLS: frozenset[str] = frozenset({
    # Vector databases
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma", "chromadb",
    "elasticsearch", "opensearch", "vespa",
    # Embedding / dense retrieval
    "sentence transformers", "sentence-transformers", "dense retrieval",
    "embeddings", "vector search", "semantic search", "bi-encoder", "cross-encoder",
    "colbert", "dpr",
    # LLMs and RAG
    "llm", "llms", "large language models", "gpt", "llama", "mistral", "gemma",
    "fine-tuning", "finetuning", "lora", "qlora",
    "rag", "retrieval augmented generation", "langchain", "llamaindex",
    # ML frameworks
    "pytorch", "tensorflow", "jax", "huggingface", "transformers",
    # Ranking and evaluation
    "ranking", "learning to rank", "ltr", "reranking", "re-ranking",
    "bm25", "ndcg", "mrr",
    # MLOps / serving
    "mlflow", "kubeflow", "ray", "triton", "torchserve", "onnx", "mlops",
    # Data infra
    "spark", "kafka", "airflow",
    # Experimentation
    "a/b testing", "experimentation",
})

# ─────────────────────────────────────────────────────────────────────────────
# § 2  Concern and gap text
# ─────────────────────────────────────────────────────────────────────────────

# Trap → recruiter-readable concern phrase
_CONCERN_TEXT: dict[str, str] = {
    "keyword_stuffing":          "skill list may be inflated—verify depth in technical screen",
    "fake_ai_profile":           "zero-duration expert claims detected; credential check recommended",
    "generic_chatgpt_user":      "profile language appears AI-generated; verify experience authenticity",
    "research_only":             "background appears research-focused; confirm production deployment experience",
    "low_quality_profile":       "thin profile with limited verifiable evidence",
    "inactive_candidate":        "candidate has been inactive recently; confirm current availability",
    "inconsistent_career":       "frequent role changes detected; assess long-term stability",
    "suspicious_timeline":       "experience timeline has anomalies; verify dates at screen",
    "ai_keywords_no_production": "strong AI vocabulary without clear production evidence",
    "behavioral_trust_issues":   "platform signals warrant additional verification",
}

# Most-severe-first ordering when multiple traps are triggered
_CONCERN_PRIORITY: list[str] = [
    "fake_ai_profile", "generic_chatgpt_user", "suspicious_timeline",
    "keyword_stuffing", "ai_keywords_no_production", "research_only",
    "inconsistent_career", "inactive_candidate",
    "low_quality_profile", "behavioral_trust_issues",
]

# Weakest sub-score → readable gap phrase (for rank 31-100)
_GAP_TEXT: dict[str, str] = {
    "retrieval_ranking_score": "limited depth in vector search and ranking",
    "production_ml_score":     "limited evidence of ML deployed to production users",
    "must_have_skill_score":   "gaps in the JD's must-have skill coverage",
    "jd_semantic_score":       "low overall alignment with JD requirements",
    "product_shipper_score":   "limited product-company or startup experience",
    "behavioral_signal_score": "low platform engagement and availability signals",
    "location_score":          "not based in a preferred India location",
    "salary_score":            "salary expectations outside the target range",
    "experience_score":        "years of experience outside the JD's target range",
    "education_score":         "education background less aligned with the role",
}

# ─────────────────────────────────────────────────────────────────────────────
# § 3  Sentence template pools
#      6 variants each; selected via _pick(pool, seed, offset).
#      All {yoe_yrs} usage follows "as a {title}" or "with {yoe_yrs} of ..."
#      patterns—never bare "{yoe_yrs} {title}" which is ungrammatical.
# ─────────────────────────────────────────────────────────────────────────────

# ── Rank 1-10: specific tech named, pure strength ─────────────────────────────

_S1_TOP_RETRIEVAL: list[str] = [
    "A {title} with {yoe_yrs} of specialist depth in vector search and ranking{skill_clause}—exactly the core technical profile the JD targets.",
    "Retrieval engineering background{skill_clause} spanning {yoe_yrs} as a {title}; covers the JD's most critical must-have skills with evidence from career text.",
    "{yoe_yrs} as a {title} with demonstrated depth in dense retrieval and vector search{skill_clause}—the closest technical match to this role's core requirements.",
    "Specialist in the JD's primary domain—vector search, ranking, and production retrieval{skill_clause}; {yoe_yrs} as a {title}.",
    "This {title} brings {yoe_yrs} of focused retrieval and ranking expertise{skill_clause}—directly meeting the JD's primary technical ask.",
    "Vector search and ranking depth{skill_clause} paired with {yoe_yrs} as a {title} maps precisely onto the role's senior-level requirements.",
]

_S1_TOP_PRODUCTION: list[str] = [
    "A {title} with {yoe_yrs} building and shipping ML systems to real users{skill_clause}—directly aligned with the JD's production-first emphasis.",
    "Production ML track record spanning {yoe_yrs}{skill_clause}; this {title} has deployed at scale rather than only prototyped.",
    "With {yoe_yrs} shipping ML to production{skill_clause}, this {title} matches the JD's explicit preference for builders over architects.",
    "{yoe_yrs} as a {title} with hands-on production ML evidence{skill_clause}—aligned with the role's core requirement to deploy real systems.",
    "End-to-end ML delivery track record{skill_clause} over {yoe_yrs} as a {title}—the JD's production emphasis is clearly met.",
    "This {title} has spent {yoe_yrs} taking ML systems from prototype to real users{skill_clause}; the delivery breadth the JD specifies.",
]

# ── General strength (rank 11-30 and no-rank path) ───────────────────────────

_S1_SKILLS: list[str] = [
    "Strong must-have skill coverage{skill_clause} and {yoe_yrs} of applied ML experience make this {title} a solid candidate for the role.",
    "A {title} with {yoe_yrs} and solid alignment to the JD's required skills{skill_clause}.",
    "Skill profile{skill_clause} shows good coverage of the role's must-have requirements, backed by {yoe_yrs} in applied ML.",
    "A {title} with {yoe_yrs} of relevant ML experience{skill_clause} and broad coverage of the JD's stated requirements.",
    "JD skill alignment is strong{skill_clause}; {yoe_yrs} as a {title} gives breadth across the role's must-have domains.",
    "This {title}'s skill set{skill_clause} matches the JD's must-have groups well, supported by {yoe_yrs} of applied experience.",
]

_S1_BEHAVIORAL: list[str] = [
    "Highly engaged and actively seeking roles—a {title} with strong platform signals and rapid recruiter responsiveness after {yoe_yrs} in ML.",
    "Strong availability and engagement indicators; a {title} with {yoe_yrs} of ML experience actively looking and responsive to outreach{skill_clause}.",
    "Platform signals indicate high intent: actively available {title} with {yoe_yrs} of ML experience{skill_clause}.",
    "Active candidate with {yoe_yrs} in ML{skill_clause} and strong engagement signals—likely to respond and convert quickly.",
    "Open-to-work status and high platform activity set this {title} apart{skill_clause}; {yoe_yrs} of ML experience backs the intent signals.",
    "Recruiter responsiveness and availability signals are strong—a {title} with {yoe_yrs} actively positioned for the right role{skill_clause}.",
]

# ── Rank 31-60: partial match framing ────────────────────────────────────────

_S1_PARTIAL: list[str] = [
    "Partial profile match—a {title} with {yoe_yrs} of relevant ML experience{skill_clause} but not full coverage of the JD's must-have requirements.",
    "A {title} with {yoe_yrs} of ML background{skill_clause}; covers some JD requirements but shows gaps in the role's core domain.",
    "A {title} with {yoe_yrs}{skill_clause}—reasonable ML foundation but limited alignment with the retrieval and ranking focus of the role.",
    "Moderate JD fit: {yoe_yrs} of experience{skill_clause} as a {title}, with partial coverage of the stated requirements.",
    "Some relevant ML depth{skill_clause} from {yoe_yrs} as a {title}, but the core retrieval and vector search skills show gaps.",
    "Reasonable ML background across {yoe_yrs} as a {title}{skill_clause}; coverage of the JD's primary technical domains is incomplete.",
]

# ── Rank 61-100: honest borderline framing ────────────────────────────────────

_S1_BORDERLINE: list[str] = [
    "Borderline fit—a {title} with {yoe_yrs} of experience and limited alignment to the JD's core vector search and ranking requirements{skill_clause}.",
    "Weak overall match: {yoe_yrs} as a {title}{skill_clause}, with significant gaps relative to the role's must-have skills.",
    "Profile shows some ML exposure{skill_clause} but does not strongly match the {title} requirements outlined in the JD; {yoe_yrs} of experience.",
    "Limited JD alignment—{yoe_yrs} as a {title}{skill_clause}; included at this rank but core retrieval and ranking gaps persist.",
    "This {title} brings {yoe_yrs} of general ML experience{skill_clause} but lacks the specialist depth in vector search the JD centres on.",
    "At the borderline of the top 100—{yoe_yrs} as a {title}{skill_clause}—the profile shows ML exposure without the required specialist depth.",
]

# ── General fallback ──────────────────────────────────────────────────────────

_S1_GENERAL: list[str] = [
    "A {title} with {yoe_yrs} of experience in applied ML{skill_clause}, showing reasonable alignment with the JD.",
    "A {title} with {yoe_yrs} of relevant experience{skill_clause}—profile covers a portion of the role's technical requirements.",
    "Candidate brings {yoe_yrs} as a {title}{skill_clause} with skills relevant to the Senior AI Engineer role.",
    "{yoe_yrs} of ML experience{skill_clause}; profile shows moderate fit with the JD's core requirements.",
    "This {title} offers {yoe_yrs} of ML experience{skill_clause} with reasonable alignment across the JD's stated requirements.",
    "ML background of {yoe_yrs}{skill_clause} positions this {title} as a reasonable match for the role's general requirements.",
]

# ── Second sentences — location / logistics ───────────────────────────────────

_S2_LOCATION: list[str] = [
    "Based in {location} with a {notice_str} notice period—minimal logistics friction for an offer.",
    "Located in {location}; {notice_str} notice period if a hire decision is made.",
    "{location}-based with a {notice_str} availability window.",
    "In {location} with {notice_str} notice—operationally straightforward to move forward.",
    "{notice_str} notice from {location} makes the onboarding timeline predictable.",
    "India-based in {location} with {notice_str} notice—no relocation complexity.",
]

# ── Second sentences — product-company background ─────────────────────────────

_S2_PRODUCT: list[str] = [
    "Product-company career ({product_pct}% of roles) suggests comfort with the fast-moving startup environment described in the JD.",
    "Predominantly product-company background ({product_pct}%) aligns with the JD's emphasis on shipping over architecture.",
    "Startup and product experience ({product_pct}% of career) indicates the builder mindset the role values.",
    "A product-company track record ({product_pct}% of career) fits the JD's preference for engineers who ship.",
    "{product_pct}% of career at product companies signals the startup operating rhythm the JD explicitly calls for.",
    "Product-focused background ({product_pct}% of career) over consulting aligns with the JD's preference for engineers who ship working systems.",
]

# ── Second sentences — flagged concern ───────────────────────────────────────

_S2_CONCERN: list[str] = [
    "Flag: {concern}.",
    "Note for review: {concern}.",
    "Flagged concern: {concern}.",
    "Recruiter note: {concern}.",
    "Screening note: {concern}.",
    "Worth verifying before advancing: {concern}.",
]

# ── Second sentences — identified gap (for rank 31-100) ──────────────────────

_S2_GAP: list[str] = [
    "Primary gap: {gap}.",
    "Key limitation: {gap}.",
    "Notable gap: {gap}.",
    "Skill gap to flag: {gap}.",
    "Gap relative to JD: {gap}.",
    "Main missing element: {gap}.",
]

# ── Template renderer with artifact cleanup ───────────────────────────────────
# Handles artifacts that arise when {yoe_yrs} is empty (candidate has no YoE data).

_CLEANUP_RE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\bwith\s+of\b'),     'with'),      # "with  of" from empty yoe
    (re.compile(r'\bwith\s+and\b'),    'with'),      # "with  and" from empty yoe
    (re.compile(r'\bwith\s+as\b'),     'as'),        # "with  as" from empty yoe
    (re.compile(r'\bafter\s+in\b'),    'in'),        # "after  in ML" from empty yoe
    (re.compile(r'\bover\s+as\b'),     'as'),        # "over  as" from empty yoe
    (re.compile(r'\bspanning\s+as\b'), 'as'),        # "spanning  as" from empty yoe
    (re.compile(r'\boffers\s+of\b'),   'offers'),   # "offers  of" from empty yoe
    (re.compile(r'\s{2,}'),            ' '),         # collapse multiple spaces
    (re.compile(r'\s+\.'),             '.'),         # remove space before period
]


def _apply_template(template: str, **kwargs: object) -> str:
    """Format a template and clean up artifacts from empty optional fields."""
    text = template.format(**kwargs)
    for pattern, repl in _CLEANUP_RE:
        text = pattern.sub(repl, text)
    return text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# § 4  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _seed(candidate_id: str) -> int:
    """Deterministic integer seed from candidate_id for template selection."""
    h = hashlib.md5(candidate_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(h, 16)


def _pick(pool: list[str], seed: int, offset: int = 0) -> str:
    """Select deterministically from pool using seed + offset."""
    return pool[(seed + offset) % len(pool)]


def _jd_relevant_skills(skill_names: list[str], n: int = 3) -> list[str]:
    """
    Return up to n skills from skill_names that appear in _JD_SKILLS.
    Falls back to the first n skills when fewer than 2 JD-relevant ones exist.
    """
    relevant = [s for s in skill_names if s.lower() in _JD_SKILLS]
    return relevant[:n] if len(relevant) >= 2 else skill_names[:n]


def _skill_clause(skills: list[str]) -> str:
    """
    Build a natural-language skill clause.
    Returns "" → no skills; " (FAISS)" → one; " (FAISS and RAG)" → two;
    " (FAISS, RAG, and PyTorch)" → three.
    """
    if not skills:
        return ""
    if len(skills) == 1:
        return f" ({skills[0]})"
    if len(skills) == 2:
        return f" ({skills[0]} and {skills[1]})"
    return f" ({skills[0]}, {skills[1]}, and {skills[2]})"


def _yoe_str(yoe: float | None) -> str:
    """Render years of experience as a noun phrase; "" when unavailable."""
    if yoe is None or yoe < 0.5:
        return ""
    y = int(yoe)
    if y == 0:
        return "under 1 year"
    return "1 year" if y == 1 else f"{y} years"


def _notice_str(notice_days: int | None) -> str:
    """Render notice period as a descriptive adjective."""
    if notice_days is None or notice_days <= 0:
        return "flexible"
    if notice_days <= 15:
        return "immediate"
    if notice_days <= 30:
        return "30-day"
    if notice_days <= 60:
        return "60-day"
    if notice_days <= 90:
        return "90-day"
    return f"{notice_days}-day"


def _safe_title(flat: dict[str, Any]) -> str:
    """Return the candidate's most specific known title, or 'candidate'."""
    title = flat.get("current_title") or flat.get("most_recent_title") or ""
    return str(title).strip() or "candidate"


def _in_preferred_location(flat: dict[str, Any]) -> bool:
    """True when the candidate is based in India + a JD-preferred city."""
    country  = str(flat.get("country") or "").lower()
    location = str(flat.get("location") or "").lower()
    if "india" not in country:
        return False
    return any(c.lower() in location for c in PREFERRED_CITIES)


def _primary_concern(trap_labels: list[str]) -> str | None:
    """Return the highest-priority concern text or None."""
    for trap_name in _CONCERN_PRIORITY:
        if trap_name in trap_labels:
            return _CONCERN_TEXT.get(trap_name)
    return None


def _inactivity_concern(flat: dict[str, Any]) -> str | None:
    """Return an inactivity concern string when relevant."""
    days = int(flat.get("days_since_last_active") or 0)
    open_to_work = bool(flat.get("open_to_work_flag") or False)
    if days > 90 and not open_to_work:
        return f"inactive for {days} days with open-to-work off; confirm current interest"
    return None


def _weakest_sub_score(sub_scores: dict[str, float], exclude: list[str] | None = None) -> str | None:
    """Return the name of the sub-score with the lowest value."""
    exclude_set = set(exclude or [])
    candidates = {k: v for k, v in sub_scores.items() if k not in exclude_set}
    if not candidates:
        return None
    return min(candidates, key=lambda k: candidates[k])


# ─────────────────────────────────────────────────────────────────────────────
# § 5  Sentence builders
# ─────────────────────────────────────────────────────────────────────────────

_RETRIEVAL_THRESHOLD  = 0.68
_PRODUCTION_THRESHOLD = 0.62
_SKILLS_THRESHOLD     = 0.60
_BEHAVIORAL_THRESHOLD = 0.65
_PRODUCT_PCT_THRESHOLD = 70   # % of career at product companies


def _build_sentence_1(
    flat: dict[str, Any],
    sub_scores: dict[str, float],
    seed: int,
    rank: int | None,
    yoe_yrs: str,
    title: str,
    skill_clause: str,
) -> str:
    fmt = {"yoe_yrs": yoe_yrs, "title": title, "skill_clause": skill_clause}

    rr = sub_scores.get("retrieval_ranking_score", 0.0)
    pm = sub_scores.get("production_ml_score",     0.0)
    jd = sub_scores.get("jd_semantic_score",       0.0)
    mh = sub_scores.get("must_have_skill_score",   0.0)
    bh = sub_scores.get("behavioral_signal_score", 0.0)

    # Rank 61-100: always use borderline framing
    if rank is not None and rank > 60:
        return _apply_template(_pick(_S1_BORDERLINE, seed, 4), **fmt)

    # Rank 31-60: partial match framing
    if rank is not None and rank > 30:
        return _apply_template(_pick(_S1_PARTIAL, seed, 4), **fmt)

    # Rank 1-10: emphasise specific tech — use more precise pools
    if rank is not None and rank <= 10:
        if rr >= _RETRIEVAL_THRESHOLD:
            return _apply_template(_pick(_S1_TOP_RETRIEVAL, seed, 0), **fmt)
        if pm >= _PRODUCTION_THRESHOLD:
            return _apply_template(_pick(_S1_TOP_PRODUCTION, seed, 1), **fmt)
        # Fall through to general strength pools

    # Rank 11-30 and no-rank: strength-based selection
    if rr >= _RETRIEVAL_THRESHOLD:
        return _apply_template(_pick(_S1_TOP_RETRIEVAL, seed, 0), **fmt)
    if pm >= _PRODUCTION_THRESHOLD:
        return _apply_template(_pick(_S1_TOP_PRODUCTION, seed, 1), **fmt)
    if max(jd, mh) >= _SKILLS_THRESHOLD:
        return _apply_template(_pick(_S1_SKILLS, seed, 2), **fmt)
    if bh >= _BEHAVIORAL_THRESHOLD:
        return _apply_template(_pick(_S1_BEHAVIORAL, seed, 3), **fmt)
    return _apply_template(_pick(_S1_GENERAL, seed, 4), **fmt)


def _build_sentence_2(
    flat: dict[str, Any],
    sub_scores: dict[str, float],
    seed: int,
    rank: int | None,
    concern: str | None,
) -> str | None:
    """
    Return a secondary sentence or None.

    Priority logic varies by rank band:
      Always: concern > gap (if rank > 10) > location > product background
    """
    # Concern overrides everything
    if concern:
        return _pick(_S2_CONCERN, seed, 5).format(concern=concern)

    # For rank 11-100: surface the weakest sub-score gap even without a trap
    if rank is not None and rank > 10:
        weak = _weakest_sub_score(sub_scores)
        if weak and sub_scores.get(weak, 1.0) < 0.55:
            gap_text = _GAP_TEXT.get(weak)
            if gap_text:
                return _pick(_S2_GAP, seed, 8).format(gap=gap_text)

    # Location note — only if in a preferred India city
    if _in_preferred_location(flat):
        location = str(flat.get("location") or "").strip()
        notice_days = flat.get("notice_period_days")
        notice_str = _notice_str(int(notice_days) if notice_days is not None else None)
        if location:
            return _pick(_S2_LOCATION, seed, 6).format(
                location=location, notice_str=notice_str
            )

    # Product-company background
    product_pct = int(round(float(flat.get("product_company_ratio") or 0.0) * 100))
    if product_pct >= _PRODUCT_PCT_THRESHOLD:
        return _pick(_S2_PRODUCT, seed, 7).format(product_pct=product_pct)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# § 6  Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation(
    flat: dict[str, Any],
    score_result: CandidateScore | None = None,
    *,
    rank: int | None = None,
) -> str:
    """
    Generate a 1–2 sentence explanation for a single candidate.

    Parameters
    ----------
    flat : dict
        Flattened candidate dict from load_data.flatten_candidate().

    score_result : CandidateScore | None
        Pre-computed score from scoring.score_candidate().
        If None, score_candidate(flat) is called internally.

    rank : int | None
        1-based rank position.  Controls tone and framing per rank-band strategy:
          1–10  : strong-signal framing, specific tech named
          11–30 : strength + mild concern or gap
          31–60 : partial-match framing + gap
          61–100: borderline framing + disqualifier named
          None  : general strength/concern logic (no rank-band tone shift)

    Returns
    -------
    str — 1–2 natural-language sentences, fully fact-grounded, no placeholders.
    """
    if score_result is None:
        score_result = score_candidate(flat)

    bd         = score_result.score_breakdown
    cid        = score_result.candidate_id or ""
    seed_val   = _seed(cid)
    sub_scores = bd.get("sub_scores", {})

    # Gather display facts from flat
    yoe_raw    = flat.get("years_of_experience")
    yoe        = float(yoe_raw) if yoe_raw is not None else None
    yoe_yrs    = _yoe_str(yoe)
    title      = _safe_title(flat)
    all_skills = [s for s in (flat.get("skill_names") or []) if s]
    s_clause   = _skill_clause(_jd_relevant_skills(all_skills))

    # Identify primary concern
    trap_labels = bd.get("trap_detail", {}).get("trap_labels", [])
    concern: str | None = _primary_concern(trap_labels) or _inactivity_concern(flat)

    s1 = _build_sentence_1(flat, sub_scores, seed_val, rank, yoe_yrs, title, s_clause)
    s2 = _build_sentence_2(flat, sub_scores, seed_val, rank, concern)

    return f"{s1} {s2}" if s2 else s1


def generate_explanations(
    flat_rows: list[dict[str, Any]],
    score_results: list[CandidateScore] | None = None,
    *,
    ranks: list[int] | None = None,
) -> list[str]:
    """
    Generate explanations for all candidates.

    Parameters
    ----------
    flat_rows : list[dict]
        One flat dict per candidate.

    score_results : list[CandidateScore] | None
        Pre-computed scores aligned with flat_rows.
        If None, each candidate is scored inline.

    ranks : list[int] | None
        1-based ranks aligned with flat_rows.

    Returns
    -------
    list[str] — one explanation per candidate, same order as flat_rows.
    """
    n = len(flat_rows)
    if score_results is None:
        score_results = [None] * n  # type: ignore[list-item]
    if ranks is None:
        ranks = [None] * n  # type: ignore[list-item]

    return [
        generate_explanation(flat, sr, rank=r)
        for flat, sr, r in zip(flat_rows, score_results, ranks)
    ]
