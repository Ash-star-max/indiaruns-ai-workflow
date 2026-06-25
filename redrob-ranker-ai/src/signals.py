"""
signals.py — Behavioral signal integration (Redrob platform signals)

Converts the 23 redrob_signals fields into a behavioral multiplier (0–1).
This multiplier is applied on top of the base skill/career score.

The four composite dimensions:

    availability_score:
        open_to_work_flag, last_active_date (decay), notice_period_days

    engagement_score:
        recruiter_response_rate, avg_response_time_hours,
        applications_submitted_30d, interview_completion_rate

    trust_score:
        verified_email, verified_phone, linkedin_connected,
        profile_completeness_score, connection_count, endorsements_received

    market_demand_score:
        saved_by_recruiters_30d, profile_views_received_30d,
        search_appearance_30d

    hiring_history_bonus:
        offer_acceptance_rate (−1 = no history → neutral),
        github_activity_score (−1 = no GitHub → neutral)

Hard rule: if inactive (last_active > 90d) AND open_to_work == False,
cap the entire behavioral multiplier at INACTIVE_CAP_VALUE (0.30).
"""

from __future__ import annotations

# TODO: implement in Phase 2
# Planned exports:
#   - compute_behavioral_multiplier(signals: dict) -> float   (0–1)
#   - compute_availability_score(signals: dict) -> float
#   - compute_engagement_score(signals: dict) -> float
#   - compute_trust_score(signals: dict) -> float
