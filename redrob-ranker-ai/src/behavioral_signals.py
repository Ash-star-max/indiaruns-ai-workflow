"""
behavioral_signals.py — Behavioral signal scoring from Redrob platform data.

Converts 18 raw Redrob platform signals into a single composite
behavioral_signal_score in [0, 1].  Higher = more behaviourally available
and engaged.  This score feeds into WEIGHT_BEHAVIORAL (0.20) in scorer.py.

Scoring model
-------------
Three dimensions weighted by config:

    availability_score  (0.45) — is the candidate reachable right now?
        open_to_work_flag      (0.40)
        days_since_last_active  (0.35)  exponential decay, half-life 30 d
        notice_period_days     (0.25)

    engagement_score    (0.35) — do they actually respond?
        recruiter_response_rate (0.40)
        avg_response_time_hours (0.25)  lower = faster = better
        applications_submitted_30d (0.20)
        interview_completion_rate  (0.15)

    trust_score         (0.20) — is the profile credible?
        verification (email + phone + linkedin)  (0.35)
        profile_completeness_score               (0.30)
        network (connections + endorsements)     (0.20)
        github_activity_score                    (0.15)

Modifiers (applied after weighted sum, before hard cap)
-------------------------------------------------------
    market_demand_bonus  — additive bonus ≤ 0.08 from
                           saved_by_recruiters_30d · 0.50
                           + profile_views_received_30d · 0.30
                           + search_appearance_30d · 0.20

    hiring_history_multiplier — multiplicative [0.90, 1.10] from
                                offer_acceptance_rate

Hard cap
--------
If days_since_last_active > 90 AND open_to_work_flag == False:
    behavioral_signal_score = min(score, INACTIVE_CAP_VALUE)

Public API
----------
    SIGNAL_WEIGHTS                          all weights as a nested dict
    normalize_signals(flat) -> dict         all 18 signals normalised to [0,1]
    compute_availability_score(flat) -> float
    compute_engagement_score(flat) -> float
    compute_trust_score(flat) -> float
    compute_market_demand_score(flat) -> float
    compute_hiring_history_multiplier(flat) -> float
    compute_behavioral_signal_score(flat) -> float   ← main entry point
    explain_behavioral_score(flat) -> dict           ← complete explanation
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.config import (
    AVAILABILITY_DECAY_HALFLIFE_DAYS,
    BEHAVIORAL_AVAILABILITY_WEIGHT,
    BEHAVIORAL_ENGAGEMENT_WEIGHT,
    BEHAVIORAL_TRUST_WEIGHT,
    INACTIVE_CAP_THRESHOLD_DAYS,
    INACTIVE_CAP_VALUE,
)

# ─────────────────────────────────────────────────────────────────────────────
# § 1  Weight registry
# ─────────────────────────────────────────────────────────────────────────────

# ── Intra-dimension weights (each set sums to 1.0) ───────────────────────────
_AVAIL_OPEN_TO_WORK: float = 0.40
_AVAIL_ACTIVITY:     float = 0.35
_AVAIL_NOTICE:       float = 0.25
assert abs(_AVAIL_OPEN_TO_WORK + _AVAIL_ACTIVITY + _AVAIL_NOTICE - 1.0) < 1e-9

_ENGAGE_RESPONSE_RATE: float = 0.40
_ENGAGE_RESPONSE_TIME: float = 0.25
_ENGAGE_APPLICATIONS:  float = 0.20
_ENGAGE_INTERVIEW:     float = 0.15
assert abs(_ENGAGE_RESPONSE_RATE + _ENGAGE_RESPONSE_TIME
           + _ENGAGE_APPLICATIONS + _ENGAGE_INTERVIEW - 1.0) < 1e-9

_TRUST_VERIFICATION:  float = 0.35
_TRUST_COMPLETENESS:  float = 0.30
_TRUST_NETWORK:       float = 0.20
_TRUST_GITHUB:        float = 0.15
assert abs(_TRUST_VERIFICATION + _TRUST_COMPLETENESS
           + _TRUST_NETWORK + _TRUST_GITHUB - 1.0) < 1e-9

# ── Market demand modifier ────────────────────────────────────────────────────
_DEMAND_SAVED:       float = 0.50
_DEMAND_VIEWS:       float = 0.30
_DEMAND_APPEARANCES: float = 0.20
MAX_DEMAND_BONUS:    float = 0.08   # max additive bonus from market signals

# ── Hiring history modifier ────────────────────────────────────────────────────
HIRING_HISTORY_MIN_MULT: float = 0.90
HIRING_HISTORY_MAX_MULT: float = 1.10

# ── Log-normalisation caps (N items → 1.0 score) ─────────────────────────────
_LOG_CAP_CONNECTIONS  = 501    # 500+ LinkedIn connections
_LOG_CAP_ENDORSEMENTS = 101    # 100+ platform endorsements
_LOG_CAP_SAVED        = 21     # 20+ recruiter saves in 30 d
_LOG_CAP_VIEWS        = 101    # 100+ profile views in 30 d
_LOG_CAP_APPEARANCES  = 201    # 200+ search appearances in 30 d

# ── Response time: hours at which score = 0.5 ────────────────────────────────
_RESPONSE_TIME_HALF_HOURS = 8.0

INACTIVE_DAYS_THRESHOLD = INACTIVE_CAP_THRESHOLD_DAYS
INACTIVE_CAP            = INACTIVE_CAP_VALUE

# ── Exported weight registry ──────────────────────────────────────────────────
SIGNAL_WEIGHTS: dict[str, Any] = {
    "dimensions": {
        "availability": BEHAVIORAL_AVAILABILITY_WEIGHT,
        "engagement":   BEHAVIORAL_ENGAGEMENT_WEIGHT,
        "trust":        BEHAVIORAL_TRUST_WEIGHT,
    },
    "availability_signals": {
        "open_to_work":    _AVAIL_OPEN_TO_WORK,
        "activity_recency": _AVAIL_ACTIVITY,
        "notice_period":   _AVAIL_NOTICE,
    },
    "engagement_signals": {
        "response_rate": _ENGAGE_RESPONSE_RATE,
        "response_time": _ENGAGE_RESPONSE_TIME,
        "applications":  _ENGAGE_APPLICATIONS,
        "interview_rate":_ENGAGE_INTERVIEW,
    },
    "trust_signals": {
        "verification":   _TRUST_VERIFICATION,
        "completeness":   _TRUST_COMPLETENESS,
        "network":        _TRUST_NETWORK,
        "github_activity":_TRUST_GITHUB,
    },
    "modifiers": {
        "demand_saved":       _DEMAND_SAVED,
        "demand_views":       _DEMAND_VIEWS,
        "demand_appearances": _DEMAND_APPEARANCES,
        "max_demand_bonus":   MAX_DEMAND_BONUS,
        "hiring_min_mult":    HIRING_HISTORY_MIN_MULT,
        "hiring_max_mult":    HIRING_HISTORY_MAX_MULT,
    },
    "caps": {
        "inactive_threshold_days": INACTIVE_DAYS_THRESHOLD,
        "inactive_cap_value":      INACTIVE_CAP,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# § 2  Low-level normalisers  (all return float in [0, 1])
# ─────────────────────────────────────────────────────────────────────────────

def _norm_log(count: int, cap: int) -> float:
    """Log-normalise a count: 0→0, cap-1→1.  Avoids division by zero."""
    if count <= 0:
        return 0.0
    return min(1.0, math.log(1.0 + count) / math.log(cap))


def _norm_activity_recency(days_inactive: int) -> float:
    """
    Exponential decay with half-life from config (default 30 days).

    days  = 0   → 1.000
    days  = 30  → 0.500
    days  = 60  → 0.250
    days  = 90  → 0.125  (triggers inactive cap consideration)
    days  = 9999 → ~0.00 (sentinel: no last_active_date)
    """
    if days_inactive >= 9999:
        return 0.02   # effectively stale
    decay = -math.log(2.0) / max(AVAILABILITY_DECAY_HALFLIFE_DAYS, 1)
    return max(0.0, math.exp(decay * days_inactive))


def _norm_notice_period(days: int) -> float:
    """
    Map notice period to availability score. Shorter = more available.

      ≤ 15 days  → 1.00  (immediate)
      ≤ 30 days  → 0.90
      ≤ 60 days  → 0.75
      ≤ 90 days  → 0.55
      ≤ 120 days → 0.35
      > 120 days → 0.20
    """
    if days <= 15:  return 1.00
    if days <= 30:  return 0.90
    if days <= 60:  return 0.75
    if days <= 90:  return 0.55
    if days <= 120: return 0.35
    return 0.20


def _norm_response_time(hours: float, response_rate: float) -> float:
    """
    Score response speed.  Uses hyperbolic decay:
        score = 1 / (1 + hours / _RESPONSE_TIME_HALF_HOURS)

    Special cases:
    - hours == 0 AND response_rate < 0.01 → 0.50 (never responded; unknown speed)
    - hours == 0 AND response_rate ≥ 0.01 → 1.00 (near-instant responder)
    """
    if hours <= 0.0:
        return 0.50 if response_rate < 0.01 else 1.00
    return 1.0 / (1.0 + hours / _RESPONSE_TIME_HALF_HOURS)


def _norm_github(score: float) -> float:
    """
    Normalise github_activity_score.

    -1  → 0.50  (no GitHub linked; neutral — not penalised)
     0  → 0.00
    100 → 1.00
    """
    if score < 0:
        return 0.50
    return float(np.clip(score / 100.0, 0.0, 1.0))


def _norm_offer_acceptance(rate: float) -> float:
    """
    Normalise offer_acceptance_rate.

    -1  → 0.50  (no offer history; neutral)
     0  → 0.10  (accepted 0 of N offers; mild concern)
     1  → 1.00
    """
    if rate < 0:
        return 0.50
    # Re-scale so 0 → 0.10 (not 0.0) to avoid heavily penalising candidates
    # who received but declined all offers (legitimate choice)
    return 0.10 + 0.90 * float(np.clip(rate, 0.0, 1.0))


def _norm_verification(email: bool, phone: bool, linkedin: bool) -> float:
    """Each verification signal worth 1/3 of the combined score."""
    return (float(email) + float(phone) + float(linkedin)) / 3.0


def _norm_network(connections: int, endorsements: int) -> float:
    """Combine log-normalised connections and endorsements (equal weight)."""
    conn_score  = _norm_log(connections,  _LOG_CAP_CONNECTIONS)
    endorse_score = _norm_log(endorsements, _LOG_CAP_ENDORSEMENTS)
    return (conn_score + endorse_score) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# § 3  normalize_signals — master normalisation table
# ─────────────────────────────────────────────────────────────────────────────

_ALL_SIGNAL_KEYS = (
    "open_to_work", "activity_recency", "notice_period",
    "response_rate", "response_time", "applications", "interview_rate",
    "verification", "profile_completeness", "network", "github_activity",
    "saved_count", "profile_views", "search_appearances",
    "acceptance_rate",
    # raw booleans kept for explanation
    "email_verified", "phone_verified", "linkedin_connected",
)


def _si(val, default: int) -> int:
    """Safe int coercion — returns default when val is None."""
    return default if val is None else int(val)


def _sf(val, default: float) -> float:
    """Safe float coercion — returns default when val is None."""
    return default if val is None else float(val)


def normalize_signals(flat: dict[str, Any]) -> dict[str, float]:
    """
    Normalise all 18 behavioural signals to [0, 1].

    Returns a flat dict keyed by signal name.  Missing / sentinel values
    are handled gracefully with neutral defaults.
    """
    g = flat.get

    days_inactive  = _si(g("days_since_last_active"),       9999)
    open_to_work   = bool(g("open_to_work_flag")  or False)
    notice_days    = _si(g("notice_period_days"),            90)
    response_rate  = float(np.clip(_sf(g("recruiter_response_rate"), 0.0), 0, 1))
    response_hours = max(0.0, _sf(g("avg_response_time_hours"), 0.0))
    github_raw     = _sf(g("github_activity_score"),        -1.0)
    interview_rate = float(np.clip(_sf(g("interview_completion_rate"), 0.0), 0, 1))
    saved_count    = max(0, _si(g("saved_by_recruiters_30d"), 0))
    acceptance_raw = _sf(g("offer_acceptance_rate"),        -1.0)
    email_ver      = bool(g("verified_email")   or False)
    phone_ver      = bool(g("verified_phone")   or False)
    linkedin_con   = bool(g("linkedin_connected") or False)
    completeness   = float(np.clip(_sf(g("profile_completeness_score"), 0.0) / 100.0, 0, 1))
    connections    = max(0, _si(g("connection_count"),           0))
    endorsements   = max(0, _si(g("endorsements_received"),      0))
    profile_views  = max(0, _si(g("profile_views_received_30d"), 0))
    applications   = max(0, _si(g("applications_submitted_30d"), 0))
    search_appear  = max(0, _si(g("search_appearance_30d"),      0))

    return {
        # ── Availability ──────────────────────────────────────────────────────
        "open_to_work":        1.0 if open_to_work else 0.0,
        "activity_recency":    _norm_activity_recency(days_inactive),
        "notice_period":       _norm_notice_period(notice_days),
        # ── Engagement ────────────────────────────────────────────────────────
        "response_rate":       response_rate,
        "response_time":       _norm_response_time(response_hours, response_rate),
        "applications":        min(1.0, applications / 8.0),
        "interview_rate":      interview_rate,
        # ── Trust ─────────────────────────────────────────────────────────────
        "verification":        _norm_verification(email_ver, phone_ver, linkedin_con),
        "profile_completeness":completeness,
        "network":             _norm_network(connections, endorsements),
        "github_activity":     _norm_github(github_raw),
        # ── Market demand (modifier) ─────────────────────────────────────────
        "saved_count":         _norm_log(saved_count,   _LOG_CAP_SAVED),
        "profile_views":       _norm_log(profile_views, _LOG_CAP_VIEWS),
        "search_appearances":  _norm_log(search_appear, _LOG_CAP_APPEARANCES),
        # ── Hiring history (modifier) ─────────────────────────────────────────
        "acceptance_rate":     _norm_offer_acceptance(acceptance_raw),
        # ── Raw booleans (kept for explain / cap logic) ───────────────────────
        "email_verified":      1.0 if email_ver  else 0.0,
        "phone_verified":      1.0 if phone_ver  else 0.0,
        "linkedin_connected":  1.0 if linkedin_con else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# § 4  Dimension scorers
# ─────────────────────────────────────────────────────────────────────────────

def compute_availability_score(flat: dict[str, Any]) -> float:
    """
    Availability score in [0, 1].

    Captures whether the candidate is reachable and can start soon.
    """
    ns = normalize_signals(flat)
    score = (
        ns["open_to_work"]    * _AVAIL_OPEN_TO_WORK
      + ns["activity_recency"]* _AVAIL_ACTIVITY
      + ns["notice_period"]   * _AVAIL_NOTICE
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_engagement_score(flat: dict[str, Any]) -> float:
    """
    Engagement score in [0, 1].

    Captures whether the candidate engages with recruiter outreach.
    """
    ns = normalize_signals(flat)
    score = (
        ns["response_rate"] * _ENGAGE_RESPONSE_RATE
      + ns["response_time"] * _ENGAGE_RESPONSE_TIME
      + ns["applications"]  * _ENGAGE_APPLICATIONS
      + ns["interview_rate"]* _ENGAGE_INTERVIEW
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_trust_score(flat: dict[str, Any]) -> float:
    """
    Trust score in [0, 1].

    Captures identity credibility and professional network depth.
    """
    ns = normalize_signals(flat)
    score = (
        ns["verification"]         * _TRUST_VERIFICATION
      + ns["profile_completeness"] * _TRUST_COMPLETENESS
      + ns["network"]              * _TRUST_NETWORK
      + ns["github_activity"]      * _TRUST_GITHUB
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_market_demand_score(flat: dict[str, Any]) -> float:
    """
    Market demand signal in [0, 1].

    Reflects organic recruiter interest in the past 30 days.
    Used as an additive bonus (max MAX_DEMAND_BONUS) not a dimension.
    """
    ns = normalize_signals(flat)
    score = (
        ns["saved_count"]        * _DEMAND_SAVED
      + ns["profile_views"]      * _DEMAND_VIEWS
      + ns["search_appearances"] * _DEMAND_APPEARANCES
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_hiring_history_multiplier(flat: dict[str, Any]) -> float:
    """
    Hiring history multiplier in [HIRING_HISTORY_MIN_MULT, HIRING_HISTORY_MAX_MULT].

    Uses offer_acceptance_rate:
      - No history (-1)  → neutral (1.00)
      - High acceptance  → slight boost (up to 1.10)
      - Low acceptance   → slight penalty (down to 0.90)
    """
    ns     = normalize_signals(flat)
    raw    = _sf(flat.get("offer_acceptance_rate"), -1.0)
    if raw < 0:
        return 1.00   # no hiring history → neutral

    # acceptance_rate in [0, 1] → multiplier centred at 0.5 acceptance
    # deviation from neutral: (rate - 0.5) * 0.20 → [-0.10, +0.10]
    deviation = (float(np.clip(raw, 0.0, 1.0)) - 0.50) * 0.20
    multiplier = 1.00 + deviation
    return float(np.clip(multiplier, HIRING_HISTORY_MIN_MULT, HIRING_HISTORY_MAX_MULT))


# ─────────────────────────────────────────────────────────────────────────────
# § 5  Main scorer
# ─────────────────────────────────────────────────────────────────────────────

def compute_behavioral_signal_score(flat: dict[str, Any]) -> float:
    """
    Compute the composite behavioral_signal_score in [0, 1].

    Formula
    -------
    base   = availability * 0.45 + engagement * 0.35 + trust * 0.20
    bonus  = market_demand_score * MAX_DEMAND_BONUS
    pre    = clip(base + bonus, 0, 1) * hiring_history_multiplier
    final  = clip(pre, 0, 1)
    if (days_inactive > 90 AND NOT open_to_work):
        final = min(final, INACTIVE_CAP)
    """
    avail   = compute_availability_score(flat)
    engage  = compute_engagement_score(flat)
    trust   = compute_trust_score(flat)

    base = (
        avail  * BEHAVIORAL_AVAILABILITY_WEIGHT
      + engage * BEHAVIORAL_ENGAGEMENT_WEIGHT
      + trust  * BEHAVIORAL_TRUST_WEIGHT
    )

    demand_bonus = compute_market_demand_score(flat) * MAX_DEMAND_BONUS
    pre_mult     = float(np.clip(base + demand_bonus, 0.0, 1.0))

    hiring_mult  = compute_hiring_history_multiplier(flat)
    score        = float(np.clip(pre_mult * hiring_mult, 0.0, 1.0))

    # Hard cap: inactive + not seeking → capped at INACTIVE_CAP
    days_inactive = int(flat.get("days_since_last_active") or 9999)
    open_to_work  = bool(flat.get("open_to_work_flag") or False)
    if days_inactive > INACTIVE_DAYS_THRESHOLD and not open_to_work:
        score = min(score, INACTIVE_CAP)

    return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# § 6  Complete scoring explanation
# ─────────────────────────────────────────────────────────────────────────────

def explain_behavioral_score(flat: dict[str, Any]) -> dict[str, Any]:
    """
    Return a fully structured explanation of the behavioral_signal_score.

    Output schema
    -------------
    {
        "behavioral_signal_score": float,
        "is_capped": bool,
        "cap_reason": str | None,
        "signals_normalized": dict[str, float],   # all 18 signals → [0,1]
        "dimensions": {
            "availability": {
                "score": float,
                "weight": float,
                "weighted_contribution": float,
                "signal_weights": dict,
            },
            "engagement": { ... },
            "trust": { ... },
        },
        "modifiers": {
            "market_demand_score": float,
            "market_demand_bonus": float,
            "hiring_history_multiplier": float,
        },
        "pre_cap_score": float,
        "narrative": str,
    }
    """
    ns = normalize_signals(flat)

    # ── Compute dimension scores ───────────────────────────────────────────────
    avail_score  = compute_availability_score(flat)
    engage_score = compute_engagement_score(flat)
    trust_score  = compute_trust_score(flat)

    base = (
        avail_score  * BEHAVIORAL_AVAILABILITY_WEIGHT
      + engage_score * BEHAVIORAL_ENGAGEMENT_WEIGHT
      + trust_score  * BEHAVIORAL_TRUST_WEIGHT
    )

    demand_score  = compute_market_demand_score(flat)
    demand_bonus  = demand_score * MAX_DEMAND_BONUS
    pre_mult      = float(np.clip(base + demand_bonus, 0.0, 1.0))

    hiring_mult   = compute_hiring_history_multiplier(flat)
    pre_cap_score = float(np.clip(pre_mult * hiring_mult, 0.0, 1.0))

    days_inactive = int(flat.get("days_since_last_active") or 9999)
    open_to_work  = bool(flat.get("open_to_work_flag") or False)
    is_capped     = days_inactive > INACTIVE_DAYS_THRESHOLD and not open_to_work
    final_score   = min(pre_cap_score, INACTIVE_CAP) if is_capped else pre_cap_score
    final_score   = round(final_score, 4)

    # ── Narrative ─────────────────────────────────────────────────────────────
    narrative = _build_narrative(flat, ns, avail_score, engage_score,
                                 trust_score, is_capped)

    return {
        "behavioral_signal_score": final_score,
        "is_capped":  is_capped,
        "cap_reason": (
            f"days_since_last_active ({days_inactive}) > {INACTIVE_DAYS_THRESHOLD} "
            f"AND open_to_work_flag == False"
        ) if is_capped else None,
        "signals_normalized": {k: round(v, 4) for k, v in ns.items()},
        "dimensions": {
            "availability": {
                "score":                round(avail_score, 4),
                "weight":               BEHAVIORAL_AVAILABILITY_WEIGHT,
                "weighted_contribution":round(avail_score * BEHAVIORAL_AVAILABILITY_WEIGHT, 4),
                "signal_weights": {
                    "open_to_work":     _AVAIL_OPEN_TO_WORK,
                    "activity_recency": _AVAIL_ACTIVITY,
                    "notice_period":    _AVAIL_NOTICE,
                },
                "signal_scores": {
                    "open_to_work":     round(ns["open_to_work"], 4),
                    "activity_recency": round(ns["activity_recency"], 4),
                    "notice_period":    round(ns["notice_period"], 4),
                },
            },
            "engagement": {
                "score":                round(engage_score, 4),
                "weight":               BEHAVIORAL_ENGAGEMENT_WEIGHT,
                "weighted_contribution":round(engage_score * BEHAVIORAL_ENGAGEMENT_WEIGHT, 4),
                "signal_weights": {
                    "response_rate": _ENGAGE_RESPONSE_RATE,
                    "response_time": _ENGAGE_RESPONSE_TIME,
                    "applications":  _ENGAGE_APPLICATIONS,
                    "interview_rate":_ENGAGE_INTERVIEW,
                },
                "signal_scores": {
                    "response_rate": round(ns["response_rate"], 4),
                    "response_time": round(ns["response_time"], 4),
                    "applications":  round(ns["applications"],  4),
                    "interview_rate":round(ns["interview_rate"],4),
                },
            },
            "trust": {
                "score":                round(trust_score, 4),
                "weight":               BEHAVIORAL_TRUST_WEIGHT,
                "weighted_contribution":round(trust_score * BEHAVIORAL_TRUST_WEIGHT, 4),
                "signal_weights": {
                    "verification":    _TRUST_VERIFICATION,
                    "completeness":    _TRUST_COMPLETENESS,
                    "network":         _TRUST_NETWORK,
                    "github_activity": _TRUST_GITHUB,
                },
                "signal_scores": {
                    "verification":    round(ns["verification"],         4),
                    "completeness":    round(ns["profile_completeness"], 4),
                    "network":         round(ns["network"],              4),
                    "github_activity": round(ns["github_activity"],      4),
                },
            },
        },
        "modifiers": {
            "market_demand_score":      round(demand_score,  4),
            "market_demand_bonus":      round(demand_bonus,  4),
            "hiring_history_multiplier":round(hiring_mult,   4),
        },
        "pre_cap_score": round(pre_cap_score, 4),
        "narrative":     narrative,
    }


def _build_narrative(
    flat: dict[str, Any],
    ns:   dict[str, float],
    avail: float,
    engage: float,
    trust: float,
    is_capped: bool,
) -> str:
    """Generate a human-readable scoring narrative for the explain dict."""
    parts: list[str] = []

    # ── Availability ──────────────────────────────────────────────────────────
    otw   = bool(flat.get("open_to_work_flag") or False)
    days  = int(flat.get("days_since_last_active") or 9999)
    notice = int(flat.get("notice_period_days") or 90)

    if otw and days <= 7:
        parts.append("Actively seeking and highly engaged on platform (last active ≤ 7 days).")
    elif otw and days <= 30:
        parts.append(f"Open to work. Active recently ({days}d ago).")
    elif otw:
        parts.append(f"Marked open-to-work but last active {days} days ago — may be stale.")
    elif days <= 14:
        parts.append("Recently active but not explicitly open to work — possibly passively open.")
    elif days <= 60:
        parts.append(f"Not open-to-work. Last active {days} days ago.")
    else:
        parts.append(f"Appears behaviorally inactive ({days} days since last activity).")

    if notice <= 15:
        parts.append("Can join immediately (notice ≤ 15 days).")
    elif notice <= 30:
        parts.append(f"Short notice period ({notice} days).")
    elif notice <= 60:
        parts.append(f"Manageable notice period ({notice} days).")
    else:
        parts.append(f"Long notice period ({notice} days) — may delay start date.")

    # ── Engagement ────────────────────────────────────────────────────────────
    rr    = float(flat.get("recruiter_response_rate") or 0.0)
    rt    = float(flat.get("avg_response_time_hours") or 0.0)
    apps  = int(flat.get("applications_submitted_30d") or 0)
    icr   = float(flat.get("interview_completion_rate") or 0.0)

    if rr >= 0.75:
        parts.append(f"Highly responsive to recruiter outreach ({rr:.0%} response rate).")
    elif rr >= 0.40:
        parts.append(f"Moderately responsive ({rr:.0%} response rate).")
    elif rr > 0:
        parts.append(f"Low recruiter response rate ({rr:.0%}) — outreach may go unanswered.")
    else:
        parts.append("No recruiter response history on platform.")

    if rt > 0 and rr >= 0.10:
        if rt <= 4:
            parts.append(f"Very fast responder (avg {rt:.0f}h).")
        elif rt <= 12:
            parts.append(f"Responds within a business day (avg {rt:.0f}h).")
        else:
            parts.append(f"Slow responder (avg {rt:.0f}h).")

    if apps >= 5:
        parts.append(f"Actively applying ({apps} applications in 30 days).")
    elif apps > 0:
        parts.append(f"Some job search activity ({apps} applications in 30 days).")

    if icr >= 0.8:
        parts.append(f"Strong interview completion ({icr:.0%}).")
    elif icr < 0.3 and icr > 0:
        parts.append(f"Low interview completion rate ({icr:.0%}) — possible drop-off risk.")

    # ── Trust ─────────────────────────────────────────────────────────────────
    n_verified = sum([
        bool(flat.get("verified_email") or False),
        bool(flat.get("verified_phone") or False),
        bool(flat.get("linkedin_connected") or False),
    ])
    completeness = float(flat.get("profile_completeness_score") or 0.0)
    github_raw   = float(flat.get("github_activity_score") or -1.0)

    parts.append(f"Identity verification: {n_verified}/3 signals confirmed.")

    if completeness >= 80:
        parts.append(f"Well-completed profile ({completeness:.0f}%).")
    elif completeness >= 50:
        parts.append(f"Partial profile ({completeness:.0f}% complete).")
    else:
        parts.append(f"Thin profile ({completeness:.0f}% complete) — reduces trust signal.")

    if github_raw >= 70:
        parts.append(f"Strong GitHub presence (activity score {github_raw:.0f}/100).")
    elif github_raw >= 30:
        parts.append(f"Moderate GitHub activity ({github_raw:.0f}/100).")
    elif github_raw < 0:
        parts.append("No GitHub account linked — open-source activity unknown.")

    # ── Market demand ─────────────────────────────────────────────────────────
    saved = int(flat.get("saved_by_recruiters_30d") or 0)
    if saved >= 10:
        parts.append(f"High demand — saved by {saved} recruiters in 30 days.")
    elif saved >= 3:
        parts.append(f"Some recruiter interest ({saved} saves in 30 days).")

    # ── Offer acceptance ──────────────────────────────────────────────────────
    oar = float(flat.get("offer_acceptance_rate") or -1.0)
    if oar >= 0.7:
        parts.append(f"Strong hiring track record (accepted {oar:.0%} of offers).")
    elif oar >= 0 and oar < 0.3:
        parts.append(f"Declines most offers ({oar:.0%} acceptance) — retention risk.")

    # ── Cap warning ───────────────────────────────────────────────────────────
    if is_capped:
        parts.append(
            f"⚠ Behavioral score capped at {INACTIVE_CAP} — "
            f"inactive ({days}d) and not open-to-work."
        )

    return " ".join(parts)
