"""
tests/test_behavioral_signals.py — comprehensive tests for behavioral_signals.py

§ A  Module structure & exports         (11 tests)
§ B  Low-level normalisers               (18 tests)
§ C  normalize_signals                   (12 tests)
§ D  compute_availability_score          (10 tests)
§ E  compute_engagement_score            (10 tests)
§ F  compute_trust_score                  (9 tests)
§ G  compute_market_demand_score          (6 tests)
§ H  compute_hiring_history_multiplier    (7 tests)
§ I  Hard inactive cap                    (8 tests)
§ J  compute_behavioral_signal_score     (10 tests)
§ K  explain_behavioral_score            (14 tests)
§ L  Edge cases & sentinels              (10 tests)
§ M  Fixture comparisons                  (7 tests)

Total: 132 tests
"""

import math
import pytest
import numpy as np

from src.behavioral_signals import (
    # module-level exports
    SIGNAL_WEIGHTS,
    INACTIVE_DAYS_THRESHOLD,
    INACTIVE_CAP,
    MAX_DEMAND_BONUS,
    HIRING_HISTORY_MIN_MULT,
    HIRING_HISTORY_MAX_MULT,
    # low-level normalisers
    _norm_log,
    _norm_activity_recency,
    _norm_notice_period,
    _norm_response_time,
    _norm_github,
    _norm_offer_acceptance,
    _norm_verification,
    _norm_network,
    # public API
    normalize_signals,
    compute_availability_score,
    compute_engagement_score,
    compute_trust_score,
    compute_market_demand_score,
    compute_hiring_history_multiplier,
    compute_behavioral_signal_score,
    explain_behavioral_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

ENGAGED = {
    "open_to_work_flag":            True,
    "days_since_last_active":       5,
    "notice_period_days":           15,
    "recruiter_response_rate":      0.92,
    "avg_response_time_hours":      2.5,
    "applications_submitted_30d":   6,
    "interview_completion_rate":    0.90,
    "verified_email":               True,
    "verified_phone":               True,
    "linkedin_connected":           True,
    "profile_completeness_score":   95.0,
    "connection_count":             480,
    "endorsements_received":        70,
    "github_activity_score":        82.0,
    "saved_by_recruiters_30d":      14,
    "profile_views_received_30d":   90,
    "search_appearance_30d":        160,
    "offer_acceptance_rate":        0.80,
}

DISENGAGED = {
    "open_to_work_flag":            False,
    "days_since_last_active":       180,
    "notice_period_days":           120,
    "recruiter_response_rate":      0.04,
    "avg_response_time_hours":      80.0,
    "applications_submitted_30d":   0,
    "interview_completion_rate":    0.15,
    "verified_email":               False,
    "verified_phone":               False,
    "linkedin_connected":           False,
    "profile_completeness_score":   18.0,
    "connection_count":             8,
    "endorsements_received":        0,
    "github_activity_score":        -1.0,
    "saved_by_recruiters_30d":      0,
    "profile_views_received_30d":   3,
    "search_appearance_30d":        0,
    "offer_acceptance_rate":        -1.0,
}

EMPTY: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# § A  Module structure & exports
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleStructure:
    def test_signal_weights_is_dict(self):
        assert isinstance(SIGNAL_WEIGHTS, dict)

    def test_signal_weights_has_dimensions_key(self):
        assert "dimensions" in SIGNAL_WEIGHTS

    def test_dimension_weights_sum_to_one(self):
        dims = SIGNAL_WEIGHTS["dimensions"]
        total = sum(dims.values())
        assert abs(total - 1.0) < 1e-9

    def test_signal_weights_has_all_sections(self):
        expected = {"dimensions", "availability_signals", "engagement_signals",
                    "trust_signals", "modifiers", "caps"}
        assert expected.issubset(SIGNAL_WEIGHTS.keys())

    def test_availability_signals_sum_to_one(self):
        s = SIGNAL_WEIGHTS["availability_signals"]
        assert abs(sum(s.values()) - 1.0) < 1e-9

    def test_engagement_signals_sum_to_one(self):
        s = SIGNAL_WEIGHTS["engagement_signals"]
        assert abs(sum(s.values()) - 1.0) < 1e-9

    def test_trust_signals_sum_to_one(self):
        s = SIGNAL_WEIGHTS["trust_signals"]
        assert abs(sum(s.values()) - 1.0) < 1e-9

    def test_inactive_cap_value_is_float(self):
        assert isinstance(INACTIVE_CAP, float)

    def test_inactive_threshold_is_positive(self):
        assert INACTIVE_DAYS_THRESHOLD > 0

    def test_max_demand_bonus_small(self):
        # Demand bonus should not dominate the score
        assert 0.0 < MAX_DEMAND_BONUS <= 0.15

    def test_hiring_history_bounds_order(self):
        assert HIRING_HISTORY_MIN_MULT < 1.0 < HIRING_HISTORY_MAX_MULT


# ─────────────────────────────────────────────────────────────────────────────
# § B  Low-level normalisers
# ─────────────────────────────────────────────────────────────────────────────

class TestNormLog:
    def test_zero_count(self):
        assert _norm_log(0, 101) == 0.0

    def test_negative_count(self):
        assert _norm_log(-5, 101) == 0.0

    def test_at_cap_minus_one(self):
        # log(cap) / log(cap) = 1.0
        assert _norm_log(100, 101) == pytest.approx(1.0, abs=1e-4)

    def test_above_cap_clipped_to_one(self):
        assert _norm_log(1000, 101) == 1.0

    def test_monotone_increasing(self):
        scores = [_norm_log(c, 101) for c in [0, 10, 50, 100]]
        assert scores == sorted(scores)


class TestNormActivityRecency:
    def test_zero_days(self):
        assert _norm_activity_recency(0) == pytest.approx(1.0, abs=1e-6)

    def test_halflife_days(self):
        from src.config import AVAILABILITY_DECAY_HALFLIFE_DAYS as HL
        score = _norm_activity_recency(HL)
        assert abs(score - 0.5) < 0.01

    def test_stale_sentinel(self):
        assert _norm_activity_recency(9999) < 0.10

    def test_monotone_decreasing(self):
        scores = [_norm_activity_recency(d) for d in [0, 15, 30, 60, 90]]
        assert scores == sorted(scores, reverse=True)


class TestNormNoticePeriod:
    def test_immediate(self):
        assert _norm_notice_period(0) == 1.00

    def test_fifteen_days(self):
        assert _norm_notice_period(15) == 1.00

    def test_thirty_days(self):
        assert _norm_notice_period(30) == pytest.approx(0.90)

    def test_long_notice(self):
        assert _norm_notice_period(150) == pytest.approx(0.20)

    def test_monotone_decreasing(self):
        scores = [_norm_notice_period(d) for d in [0, 30, 60, 90, 120, 150]]
        assert scores == sorted(scores, reverse=True)


class TestNormResponseTime:
    def test_no_responses_at_all(self):
        # rate 0, hours 0 → unknown → neutral
        assert _norm_response_time(0.0, 0.0) == pytest.approx(0.50)

    def test_instant_with_responses(self):
        # hours 0, rate > 0 → instant
        assert _norm_response_time(0.0, 0.8) == pytest.approx(1.00)

    def test_eight_hours_half_score(self):
        score = _norm_response_time(8.0, 0.8)
        assert abs(score - 0.5) < 0.01

    def test_slow_response(self):
        assert _norm_response_time(48.0, 0.5) < 0.20

    def test_monotone_decreasing(self):
        scores = [_norm_response_time(h, 0.7) for h in [0, 2, 8, 24, 48]]
        assert scores == sorted(scores, reverse=True)


class TestNormGithub:
    def test_sentinel_minus_one_is_neutral(self):
        assert _norm_github(-1.0) == 0.50

    def test_zero_activity(self):
        assert _norm_github(0.0) == 0.0

    def test_full_activity(self):
        assert _norm_github(100.0) == 1.0

    def test_mid_range(self):
        assert _norm_github(50.0) == pytest.approx(0.50)


class TestNormOfferAcceptance:
    def test_no_history_is_neutral(self):
        assert _norm_offer_acceptance(-1.0) == 0.50

    def test_full_acceptance(self):
        assert _norm_offer_acceptance(1.0) == 1.0

    def test_zero_acceptance_not_zero(self):
        # Should be > 0 (mild concern, not full penalty)
        assert _norm_offer_acceptance(0.0) > 0.0


class TestNormVerification:
    def test_all_verified(self):
        assert _norm_verification(True, True, True) == pytest.approx(1.0)

    def test_none_verified(self):
        assert _norm_verification(False, False, False) == pytest.approx(0.0)

    def test_two_of_three(self):
        assert _norm_verification(True, True, False) == pytest.approx(2 / 3)


class TestNormNetwork:
    def test_empty_network(self):
        assert _norm_network(0, 0) == 0.0

    def test_rich_network(self):
        score = _norm_network(500, 100)
        assert score >= 0.90

    def test_midrange(self):
        score = _norm_network(100, 20)
        assert 0.2 < score < 0.9


# ─────────────────────────────────────────────────────────────────────────────
# § C  normalize_signals
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeSignals:
    def test_returns_dict(self):
        assert isinstance(normalize_signals(ENGAGED), dict)

    def test_all_values_in_zero_one(self):
        ns = normalize_signals(ENGAGED)
        for k, v in ns.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_empty_flat_no_crash(self):
        ns = normalize_signals(EMPTY)
        assert isinstance(ns, dict)

    def test_empty_flat_all_in_range(self):
        ns = normalize_signals(EMPTY)
        for k, v in ns.items():
            assert 0.0 <= v <= 1.0, f"{k}={v}"

    def test_open_to_work_true(self):
        ns = normalize_signals({"open_to_work_flag": True})
        assert ns["open_to_work"] == 1.0

    def test_open_to_work_false(self):
        ns = normalize_signals({"open_to_work_flag": False})
        assert ns["open_to_work"] == 0.0

    def test_github_sentinel_is_neutral(self):
        ns = normalize_signals({"github_activity_score": -1.0})
        assert ns["github_activity"] == 0.50

    def test_acceptance_sentinel_is_neutral(self):
        ns = normalize_signals({"offer_acceptance_rate": -1.0})
        assert ns["acceptance_rate"] == 0.50

    def test_stale_sentinel_activity(self):
        ns = normalize_signals({"days_since_last_active": 9999})
        assert ns["activity_recency"] < 0.05

    def test_contains_all_expected_keys(self):
        ns = normalize_signals(ENGAGED)
        for key in ("open_to_work", "activity_recency", "notice_period",
                    "response_rate", "response_time", "applications",
                    "interview_rate", "verification", "profile_completeness",
                    "network", "github_activity", "saved_count",
                    "profile_views", "search_appearances", "acceptance_rate",
                    "email_verified", "phone_verified", "linkedin_connected"):
            assert key in ns, f"Missing key: {key}"

    def test_engaged_higher_than_disengaged(self):
        ns_e = normalize_signals(ENGAGED)
        ns_d = normalize_signals(DISENGAGED)
        keys = ("open_to_work", "activity_recency", "response_rate",
                "interview_rate", "verification", "profile_completeness")
        for k in keys:
            assert ns_e[k] >= ns_d[k], f"{k}: engaged {ns_e[k]} < disengaged {ns_d[k]}"

    def test_none_values_handled(self):
        flat = {k: None for k in ENGAGED}
        ns = normalize_signals(flat)
        for v in ns.values():
            assert 0.0 <= v <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# § D  compute_availability_score
# ─────────────────────────────────────────────────────────────────────────────

class TestAvailabilityScore:
    def test_returns_float(self):
        assert isinstance(compute_availability_score(ENGAGED), float)

    def test_in_zero_one(self):
        for flat in (ENGAGED, DISENGAGED, EMPTY):
            s = compute_availability_score(flat)
            assert 0.0 <= s <= 1.0

    def test_engaged_high(self):
        assert compute_availability_score(ENGAGED) > 0.70

    def test_disengaged_low(self):
        assert compute_availability_score(DISENGAGED) < 0.35

    def test_open_to_work_boosts_availability(self):
        base = {"open_to_work_flag": False, "days_since_last_active": 5, "notice_period_days": 15}
        seeking = {**base, "open_to_work_flag": True}
        assert compute_availability_score(seeking) > compute_availability_score(base)

    def test_recent_activity_boosts_availability(self):
        a = {"open_to_work_flag": True, "days_since_last_active": 1,   "notice_period_days": 30}
        b = {"open_to_work_flag": True, "days_since_last_active": 180,  "notice_period_days": 30}
        assert compute_availability_score(a) > compute_availability_score(b)

    def test_short_notice_boosts_availability(self):
        a = {"open_to_work_flag": False, "days_since_last_active": 10, "notice_period_days": 0}
        b = {"open_to_work_flag": False, "days_since_last_active": 10, "notice_period_days": 180}
        assert compute_availability_score(a) > compute_availability_score(b)

    def test_empty_gives_low_score(self):
        # No open_to_work, unknown days → low
        s = compute_availability_score(EMPTY)
        assert s < 0.50

    def test_perfect_candidate(self):
        flat = {"open_to_work_flag": True, "days_since_last_active": 0, "notice_period_days": 0}
        assert compute_availability_score(flat) > 0.90

    def test_worst_case_candidate(self):
        flat = {"open_to_work_flag": False, "days_since_last_active": 9999, "notice_period_days": 365}
        assert compute_availability_score(flat) < 0.20


# ─────────────────────────────────────────────────────────────────────────────
# § E  compute_engagement_score
# ─────────────────────────────────────────────────────────────────────────────

class TestEngagementScore:
    def test_returns_float(self):
        assert isinstance(compute_engagement_score(ENGAGED), float)

    def test_in_zero_one(self):
        for flat in (ENGAGED, DISENGAGED, EMPTY):
            s = compute_engagement_score(flat)
            assert 0.0 <= s <= 1.0

    def test_engaged_high(self):
        assert compute_engagement_score(ENGAGED) > 0.65

    def test_disengaged_low(self):
        assert compute_engagement_score(DISENGAGED) < 0.30

    def test_high_response_rate_boosts(self):
        base = {"recruiter_response_rate": 0.0}
        good = {"recruiter_response_rate": 1.0}
        assert compute_engagement_score(good) > compute_engagement_score(base)

    def test_fast_response_boosts(self):
        a = {"recruiter_response_rate": 0.8, "avg_response_time_hours": 1.0}
        b = {"recruiter_response_rate": 0.8, "avg_response_time_hours": 48.0}
        assert compute_engagement_score(a) > compute_engagement_score(b)

    def test_applications_contribute(self):
        a = {"applications_submitted_30d": 0}
        b = {"applications_submitted_30d": 10}
        assert compute_engagement_score(b) > compute_engagement_score(a)

    def test_interview_completion_contributes(self):
        a = {"interview_completion_rate": 0.0}
        b = {"interview_completion_rate": 1.0}
        assert compute_engagement_score(b) > compute_engagement_score(a)

    def test_perfect_engagement(self):
        flat = {
            "recruiter_response_rate": 1.0,
            "avg_response_time_hours": 0.0,
            "applications_submitted_30d": 20,
            "interview_completion_rate": 1.0,
        }
        assert compute_engagement_score(flat) > 0.90

    def test_zero_engagement(self):
        flat = {
            "recruiter_response_rate": 0.0,
            "avg_response_time_hours": 0.0,
            "applications_submitted_30d": 0,
            "interview_completion_rate": 0.0,
        }
        assert compute_engagement_score(flat) < 0.30


# ─────────────────────────────────────────────────────────────────────────────
# § F  compute_trust_score
# ─────────────────────────────────────────────────────────────────────────────

class TestTrustScore:
    def test_returns_float(self):
        assert isinstance(compute_trust_score(ENGAGED), float)

    def test_in_zero_one(self):
        for flat in (ENGAGED, DISENGAGED, EMPTY):
            s = compute_trust_score(flat)
            assert 0.0 <= s <= 1.0

    def test_engaged_high(self):
        assert compute_trust_score(ENGAGED) > 0.65

    def test_disengaged_low(self):
        assert compute_trust_score(DISENGAGED) < 0.35

    def test_full_verification_boosts(self):
        no_ver  = {"verified_email": False, "verified_phone": False, "linkedin_connected": False}
        all_ver = {"verified_email": True,  "verified_phone": True,  "linkedin_connected": True}
        assert compute_trust_score(all_ver) > compute_trust_score(no_ver)

    def test_completeness_contributes(self):
        low  = {"profile_completeness_score": 10.0}
        high = {"profile_completeness_score": 100.0}
        assert compute_trust_score(high) > compute_trust_score(low)

    def test_github_contributes(self):
        no_git = {"github_activity_score": 0.0}
        has_git = {"github_activity_score": 100.0}
        assert compute_trust_score(has_git) > compute_trust_score(no_git)

    def test_github_sentinel_neutral(self):
        sent = {"github_activity_score": -1.0}
        zero = {"github_activity_score": 0.0}
        # sentinel (0.50) should produce higher score than zero activity
        assert compute_trust_score(sent) > compute_trust_score(zero)

    def test_network_contributes(self):
        a = {"connection_count": 0,   "endorsements_received": 0}
        b = {"connection_count": 500, "endorsements_received": 100}
        assert compute_trust_score(b) > compute_trust_score(a)


# ─────────────────────────────────────────────────────────────────────────────
# § G  compute_market_demand_score
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketDemandScore:
    def test_returns_float(self):
        assert isinstance(compute_market_demand_score(ENGAGED), float)

    def test_in_zero_one(self):
        for flat in (ENGAGED, DISENGAGED, EMPTY):
            s = compute_market_demand_score(flat)
            assert 0.0 <= s <= 1.0

    def test_empty_gives_zero(self):
        assert compute_market_demand_score(EMPTY) == 0.0

    def test_high_saves_raises_score(self):
        a = {"saved_by_recruiters_30d": 0}
        b = {"saved_by_recruiters_30d": 20}
        assert compute_market_demand_score(b) > compute_market_demand_score(a)

    def test_views_contribute(self):
        a = {"profile_views_received_30d": 0}
        b = {"profile_views_received_30d": 100}
        assert compute_market_demand_score(b) > compute_market_demand_score(a)

    def test_engaged_higher_than_disengaged(self):
        assert compute_market_demand_score(ENGAGED) > compute_market_demand_score(DISENGAGED)


# ─────────────────────────────────────────────────────────────────────────────
# § H  compute_hiring_history_multiplier
# ─────────────────────────────────────────────────────────────────────────────

class TestHiringHistoryMultiplier:
    def test_returns_float(self):
        assert isinstance(compute_hiring_history_multiplier(ENGAGED), float)

    def test_no_history_is_neutral(self):
        flat = {"offer_acceptance_rate": -1.0}
        assert compute_hiring_history_multiplier(flat) == 1.0

    def test_empty_is_neutral(self):
        assert compute_hiring_history_multiplier(EMPTY) == 1.0

    def test_high_acceptance_gives_bonus(self):
        flat = {"offer_acceptance_rate": 1.0}
        assert compute_hiring_history_multiplier(flat) > 1.0

    def test_low_acceptance_gives_penalty(self):
        flat = {"offer_acceptance_rate": 0.0}
        assert compute_hiring_history_multiplier(flat) < 1.0

    def test_within_bounds(self):
        for rate in (-1.0, 0.0, 0.5, 1.0):
            m = compute_hiring_history_multiplier({"offer_acceptance_rate": rate})
            assert HIRING_HISTORY_MIN_MULT <= m <= HIRING_HISTORY_MAX_MULT

    def test_monotone_increasing_with_acceptance(self):
        rates = [0.0, 0.25, 0.5, 0.75, 1.0]
        mults = [compute_hiring_history_multiplier({"offer_acceptance_rate": r}) for r in rates]
        assert mults == sorted(mults)


# ─────────────────────────────────────────────────────────────────────────────
# § I  Hard inactive cap
# ─────────────────────────────────────────────────────────────────────────────

class TestInactiveCap:
    def _make(self, days, open_to_work):
        return {**ENGAGED, "days_since_last_active": days, "open_to_work_flag": open_to_work}

    def test_cap_triggered_inactive_not_seeking(self):
        flat = self._make(days=180, open_to_work=False)
        score = compute_behavioral_signal_score(flat)
        assert score <= INACTIVE_CAP + 1e-6

    def test_cap_not_triggered_inactive_but_seeking(self):
        flat = self._make(days=180, open_to_work=True)
        score = compute_behavioral_signal_score(flat)
        # Score should NOT be capped — can exceed INACTIVE_CAP
        assert score > INACTIVE_CAP

    def test_cap_not_triggered_active_not_seeking(self):
        flat = self._make(days=10, open_to_work=False)
        score = compute_behavioral_signal_score(flat)
        # Active candidate should score freely
        assert score > INACTIVE_CAP

    def test_cap_not_triggered_at_threshold(self):
        # Exactly at threshold should NOT trigger cap
        flat = self._make(days=INACTIVE_DAYS_THRESHOLD, open_to_work=False)
        score = compute_behavioral_signal_score(flat)
        # Threshold is exclusive (>90, not >=90)
        assert score > INACTIVE_CAP

    def test_cap_triggered_just_above_threshold(self):
        flat = self._make(days=INACTIVE_DAYS_THRESHOLD + 1, open_to_work=False)
        score = compute_behavioral_signal_score(flat)
        assert score <= INACTIVE_CAP + 1e-6

    def test_explain_is_capped_flag_set(self):
        flat = self._make(days=180, open_to_work=False)
        exp = explain_behavioral_score(flat)
        assert exp["is_capped"] is True
        assert exp["cap_reason"] is not None

    def test_explain_not_capped_when_seeking(self):
        flat = self._make(days=180, open_to_work=True)
        exp = explain_behavioral_score(flat)
        assert exp["is_capped"] is False
        assert exp["cap_reason"] is None

    def test_cap_value_matches_config(self):
        from src.config import INACTIVE_CAP_VALUE
        assert INACTIVE_CAP == INACTIVE_CAP_VALUE


# ─────────────────────────────────────────────────────────────────────────────
# § J  compute_behavioral_signal_score
# ─────────────────────────────────────────────────────────────────────────────

class TestBehavioralSignalScore:
    def test_returns_float(self):
        assert isinstance(compute_behavioral_signal_score(ENGAGED), float)

    def test_in_zero_one(self):
        for flat in (ENGAGED, DISENGAGED, EMPTY):
            s = compute_behavioral_signal_score(flat)
            assert 0.0 <= s <= 1.0, f"score {s} out of [0,1]"

    def test_engaged_higher_than_disengaged(self):
        assert compute_behavioral_signal_score(ENGAGED) > compute_behavioral_signal_score(DISENGAGED)

    def test_engaged_scores_high(self):
        assert compute_behavioral_signal_score(ENGAGED) > 0.70

    def test_disengaged_scores_low(self):
        assert compute_behavioral_signal_score(DISENGAGED) <= INACTIVE_CAP + 1e-6

    def test_empty_scores_below_half(self):
        assert compute_behavioral_signal_score(EMPTY) < 0.50

    def test_score_rounded_to_four_decimals(self):
        s = compute_behavioral_signal_score(ENGAGED)
        assert s == round(s, 4)

    def test_deterministic(self):
        s1 = compute_behavioral_signal_score(ENGAGED)
        s2 = compute_behavioral_signal_score(ENGAGED)
        assert s1 == s2

    def test_market_demand_provides_bonus(self):
        no_demand = {**ENGAGED, "saved_by_recruiters_30d": 0,
                     "profile_views_received_30d": 0, "search_appearance_30d": 0}
        with_demand = ENGAGED
        assert compute_behavioral_signal_score(with_demand) >= compute_behavioral_signal_score(no_demand)

    def test_good_acceptance_rate_provides_bonus(self):
        no_history = {**ENGAGED, "offer_acceptance_rate": -1.0}
        good_history = {**ENGAGED, "offer_acceptance_rate": 1.0}
        assert compute_behavioral_signal_score(good_history) >= compute_behavioral_signal_score(no_history)


# ─────────────────────────────────────────────────────────────────────────────
# § K  explain_behavioral_score
# ─────────────────────────────────────────────────────────────────────────────

class TestExplainBehavioralScore:
    def test_returns_dict(self):
        assert isinstance(explain_behavioral_score(ENGAGED), dict)

    def test_has_all_top_level_keys(self):
        exp = explain_behavioral_score(ENGAGED)
        expected = {"behavioral_signal_score", "is_capped", "cap_reason",
                    "signals_normalized", "dimensions", "modifiers",
                    "pre_cap_score", "narrative"}
        assert expected.issubset(exp.keys())

    def test_score_matches_direct_computation(self):
        exp = explain_behavioral_score(ENGAGED)
        direct = compute_behavioral_signal_score(ENGAGED)
        assert exp["behavioral_signal_score"] == direct

    def test_dimensions_has_three_keys(self):
        exp = explain_behavioral_score(ENGAGED)
        assert set(exp["dimensions"].keys()) == {"availability", "engagement", "trust"}

    def test_dimension_scores_in_range(self):
        exp = explain_behavioral_score(ENGAGED)
        for dim, data in exp["dimensions"].items():
            s = data["score"]
            assert 0.0 <= s <= 1.0, f"{dim}.score={s}"

    def test_weighted_contributions_sum_to_base(self):
        exp = explain_behavioral_score(ENGAGED)
        base = sum(
            exp["dimensions"][d]["weighted_contribution"]
            for d in ("availability", "engagement", "trust")
        )
        demand_bonus = exp["modifiers"]["market_demand_bonus"]
        hiring_mult  = exp["modifiers"]["hiring_history_multiplier"]
        # pre_cap_score is clipped to [0, 1] before the hard cap step
        expected_pre_cap = min(1.0, max(0.0, (base + demand_bonus) * hiring_mult))
        assert abs(exp["pre_cap_score"] - expected_pre_cap) < 0.01

    def test_narrative_is_nonempty_string(self):
        exp = explain_behavioral_score(ENGAGED)
        assert isinstance(exp["narrative"], str)
        assert len(exp["narrative"]) > 20

    def test_signals_normalized_all_in_range(self):
        exp = explain_behavioral_score(ENGAGED)
        for k, v in exp["signals_normalized"].items():
            assert 0.0 <= v <= 1.0, f"signals_normalized[{k}]={v}"

    def test_cap_scenario_sets_flag(self):
        flat = {**DISENGAGED}  # days=180, open_to_work=False → capped
        exp = explain_behavioral_score(flat)
        assert exp["is_capped"] is True

    def test_cap_scenario_score_at_most_cap(self):
        exp = explain_behavioral_score(DISENGAGED)
        assert exp["behavioral_signal_score"] <= INACTIVE_CAP + 1e-6

    def test_no_cap_scenario(self):
        exp = explain_behavioral_score(ENGAGED)
        assert exp["is_capped"] is False
        assert exp["cap_reason"] is None

    def test_modifiers_has_expected_keys(self):
        exp = explain_behavioral_score(ENGAGED)
        mods = exp["modifiers"]
        assert "market_demand_score"       in mods
        assert "market_demand_bonus"       in mods
        assert "hiring_history_multiplier" in mods

    def test_empty_flat_no_crash(self):
        exp = explain_behavioral_score(EMPTY)
        assert isinstance(exp, dict)
        assert 0.0 <= exp["behavioral_signal_score"] <= 1.0

    def test_dimension_signal_weights_in_explain(self):
        exp = explain_behavioral_score(ENGAGED)
        assert "signal_weights" in exp["dimensions"]["availability"]
        assert "signal_weights" in exp["dimensions"]["engagement"]
        assert "signal_weights" in exp["dimensions"]["trust"]


# ─────────────────────────────────────────────────────────────────────────────
# § L  Edge cases & sentinels
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_none_days_inactive(self):
        flat = {"days_since_last_active": None}
        s = compute_behavioral_signal_score(flat)
        assert 0.0 <= s <= 1.0

    def test_none_response_rate(self):
        flat = {"recruiter_response_rate": None}
        s = compute_engagement_score(flat)
        assert 0.0 <= s <= 1.0

    def test_none_github(self):
        flat = {"github_activity_score": None}
        # None should be treated as -1 (sentinel)
        ns = normalize_signals(flat)
        assert 0.0 <= ns["github_activity"] <= 1.0

    def test_negative_applications(self):
        flat = {"applications_submitted_30d": -5}
        ns = normalize_signals(flat)
        assert ns["applications"] == 0.0

    def test_over_100_profile_completeness(self):
        flat = {"profile_completeness_score": 200.0}
        ns = normalize_signals(flat)
        assert ns["profile_completeness"] <= 1.0

    def test_over_100_github_score(self):
        flat = {"github_activity_score": 150.0}
        ns = normalize_signals(flat)
        assert ns["github_activity"] <= 1.0

    def test_zero_notice_period(self):
        flat = {"notice_period_days": 0}
        ns = normalize_signals(flat)
        assert ns["notice_period"] == 1.0

    def test_acceptance_rate_boundary_zero(self):
        flat = {"offer_acceptance_rate": 0.0}
        mult = compute_hiring_history_multiplier(flat)
        assert HIRING_HISTORY_MIN_MULT <= mult

    def test_acceptance_rate_boundary_one(self):
        flat = {"offer_acceptance_rate": 1.0}
        mult = compute_hiring_history_multiplier(flat)
        assert mult <= HIRING_HISTORY_MAX_MULT

    def test_very_large_connection_count(self):
        flat = {"connection_count": 100_000, "endorsements_received": 5000}
        s = compute_trust_score(flat)
        assert 0.0 <= s <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# § M  Fixture comparisons
# ─────────────────────────────────────────────────────────────────────────────

class TestFixtureComparisons:
    def test_engaged_vs_disengaged_availability(self):
        assert compute_availability_score(ENGAGED) > compute_availability_score(DISENGAGED)

    def test_engaged_vs_disengaged_engagement(self):
        assert compute_engagement_score(ENGAGED) > compute_engagement_score(DISENGAGED)

    def test_engaged_vs_disengaged_trust(self):
        assert compute_trust_score(ENGAGED) > compute_trust_score(DISENGAGED)

    def test_engaged_vs_disengaged_market_demand(self):
        assert compute_market_demand_score(ENGAGED) > compute_market_demand_score(DISENGAGED)

    def test_engaged_vs_disengaged_final_score(self):
        assert compute_behavioral_signal_score(ENGAGED) > compute_behavioral_signal_score(DISENGAGED)

    def test_score_gap_significant(self):
        gap = compute_behavioral_signal_score(ENGAGED) - compute_behavioral_signal_score(DISENGAGED)
        assert gap > 0.30  # should be meaningfully discriminative

    def test_explain_narrative_mentions_cap_for_disengaged(self):
        exp = explain_behavioral_score(DISENGAGED)
        assert "cap" in exp["narrative"].lower() or exp["is_capped"]
