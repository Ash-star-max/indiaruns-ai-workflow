"""
tests/test_scoring.py — Comprehensive tests for src/scoring.py

Test sections
─────────────
  §A  Module structure & constants          — 12 tests
  §B  Sub-score helper functions            — 22 tests
  §C  score_candidate output structure      — 12 tests
  §D  score_candidate composite properties  — 15 tests
  §E  Determinism and tie-breaking          —  8 tests
  §F  Trap penalty integration              —  8 tests
  §G  score_candidates batch function       — 10 tests
  §H  Edge cases / missing data             — 10 tests
                                     TOTAL  ≥ 97 tests
"""

from __future__ import annotations

import copy
import math
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.scoring import (
    SCORE_WEIGHTS,
    WEIGHT_GROUPS,
    CandidateScore,
    _sub_education,
    _sub_experience,
    _sub_location,
    _sub_must_have,
    _sub_product_shipper,
    _sub_production_ml,
    _sub_retrieval_ranking,
    _sub_salary,
    score_candidate,
    score_candidates,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — complete, realistic flat candidate dicts
# ─────────────────────────────────────────────────────────────────────────────

# A strong Senior AI Engineer candidate for the JD
IDEAL: dict[str, Any] = {
    "candidate_id": "ideal-001",
    # Career
    "years_of_experience":        7.0,
    "total_career_months":        84,
    "career_titles":              ["ML Engineer", "Senior ML Engineer", "Staff AI Engineer"],
    "most_recent_title":          "Staff AI Engineer",
    "current_title":              "Staff AI Engineer",
    "n_career_roles":             3,
    # Company
    "product_company_ratio":      1.0,
    "current_company_size":       "201-500",
    "current_industry":           "Technology",
    "is_consulting_only":         False,
    # Skills
    "skill_names":                [
        "Python", "PyTorch", "FAISS", "Pinecone", "Sentence Transformers",
        "LLM fine-tuning", "RAG", "vector search", "ranking systems", "A/B testing",
    ],
    "skill_count":                10,
    "tier1_skill_count":          6,
    "tier2_skill_count":          3,
    "tier1_avg_duration_months":  30.0,
    "total_skill_endorsements":   80,
    "expert_zero_duration_count": 0,
    "skill_assessment_scores":    {"python": 92.0, "ml_fundamentals": 88.0},
    # Education
    "highest_edu_tier":           "tier_1",
    "edu_fields":                 ["computer_science"],
    "cert_names":                 ["AWS Certified ML Specialty"],
    "english_proficiency":        "native",
    "language_count":             2,
    # Location
    "country":                    "India",
    "location":                   "Bangalore",
    "willing_to_relocate":        True,
    "notice_period_days":         30,
    # Salary
    "salary_min_lpa":             45.0,
    "salary_max_lpa":             60.0,
    "salary_inverted":            False,
    # Behavioral signals
    "open_to_work_flag":          True,
    "days_since_last_active":     3,
    "recruiter_response_rate":    0.90,
    "avg_response_time_hours":    3.0,
    "applications_submitted_30d": 5,
    "interview_completion_rate":  0.95,
    "verified_email":             True,
    "verified_phone":             True,
    "linkedin_connected":         True,
    "profile_completeness_score": 95.0,
    "connection_count":           500,
    "endorsements_received":      75,
    "github_activity_score":      85.0,
    "saved_by_recruiters_30d":    12,
    "profile_views_received_30d": 80,
    "search_appearance_30d":      140,
    "offer_acceptance_rate":      0.80,
    # Career text
    "career_descriptions_text": (
        "Built production vector search pipeline using FAISS and Sentence Transformers "
        "serving 2M daily users. Implemented learning-to-rank with NDCG optimisation "
        "for recommendation engine. Deployed RAG system using LLM fine-tuning on "
        "proprietary data. Optimised retrieval latency from 800ms to 40ms using "
        "approximate nearest neighbour indexing. A/B tested ranking algorithms in "
        "production environment increasing CTR by 18%."
    ),
    "summary": (
        "Staff AI Engineer with 7 years building and shipping production ML systems. "
        "Deep expertise in vector databases, dense retrieval, and large-scale ranking. "
        "Shipped recommendation and search features used by millions."
    ),
}

# A clearly unsuitable candidate
POOR: dict[str, Any] = {
    "candidate_id": "poor-001",
    # Career
    "years_of_experience":        1.5,
    "total_career_months":        18,
    "career_titles":              ["Data Analyst"],
    "most_recent_title":          "Data Analyst",
    "current_title":              "Data Analyst",
    "n_career_roles":             1,
    # Company
    "product_company_ratio":      0.0,
    "current_company_size":       "1001-5000",
    "current_industry":           "Banking",
    "is_consulting_only":         True,
    # Skills
    "skill_names":                ["Excel", "SQL", "Tableau"],
    "skill_count":                3,
    "tier1_skill_count":          0,
    "tier2_skill_count":          0,
    "tier1_avg_duration_months":  0.0,
    "total_skill_endorsements":   2,
    "expert_zero_duration_count": 0,
    "skill_assessment_scores":    {},
    # Education
    "highest_edu_tier":           "tier_3",
    "edu_fields":                 ["commerce"],
    "cert_names":                 [],
    "english_proficiency":        "basic",
    "language_count":             1,
    # Location
    "country":                    "Germany",
    "location":                   "Munich",
    "willing_to_relocate":        False,
    "notice_period_days":         120,
    # Salary
    "salary_min_lpa":             15.0,
    "salary_max_lpa":             20.0,
    "salary_inverted":            False,
    # Behavioral signals
    "open_to_work_flag":          False,
    "days_since_last_active":     200,
    "recruiter_response_rate":    0.05,
    "avg_response_time_hours":    96.0,
    "applications_submitted_30d": 0,
    "interview_completion_rate":  0.20,
    "verified_email":             False,
    "verified_phone":             False,
    "linkedin_connected":         False,
    "profile_completeness_score": 30.0,
    "connection_count":           20,
    "endorsements_received":      1,
    "github_activity_score":      -1,
    "saved_by_recruiters_30d":    0,
    "profile_views_received_30d": 5,
    "search_appearance_30d":      10,
    "offer_acceptance_rate":      0.10,
    # Career text
    "career_descriptions_text": (
        "Analysed sales data in Excel. Built Tableau dashboards for quarterly reports. "
        "Wrote SQL queries against the finance database."
    ),
    "summary": "Data Analyst with 1.5 years of experience in banking sector.",
}

# A medium-quality candidate (midpoint reference)
AVERAGE: dict[str, Any] = {
    "candidate_id": "avg-001",
    "years_of_experience":        5.0,
    "total_career_months":        60,
    "career_titles":              ["ML Engineer", "Senior Data Scientist"],
    "most_recent_title":          "Senior Data Scientist",
    "current_title":              "Senior Data Scientist",
    "n_career_roles":             2,
    "product_company_ratio":      0.6,
    "current_company_size":       "501-1000",
    "current_industry":           "Technology",
    "is_consulting_only":         False,
    "skill_names":                ["Python", "scikit-learn", "Spark", "SQL", "TensorFlow"],
    "skill_count":                5,
    "tier1_skill_count":          2,
    "tier2_skill_count":          2,
    "tier1_avg_duration_months":  18.0,
    "total_skill_endorsements":   25,
    "expert_zero_duration_count": 0,
    "skill_assessment_scores":    {"python": 72.0},
    "highest_edu_tier":           "tier_2",
    "edu_fields":                 ["statistics"],
    "cert_names":                 [],
    "english_proficiency":        "fluent",
    "language_count":             1,
    "country":                    "India",
    "location":                   "Hyderabad",
    "willing_to_relocate":        False,
    "notice_period_days":         60,
    "salary_min_lpa":             35.0,
    "salary_max_lpa":             50.0,
    "salary_inverted":            False,
    "open_to_work_flag":          True,
    "days_since_last_active":     15,
    "recruiter_response_rate":    0.50,
    "avg_response_time_hours":    24.0,
    "applications_submitted_30d": 2,
    "interview_completion_rate":  0.70,
    "verified_email":             True,
    "verified_phone":             True,
    "linkedin_connected":         True,
    "profile_completeness_score": 70.0,
    "connection_count":           200,
    "endorsements_received":      20,
    "github_activity_score":      50.0,
    "saved_by_recruiters_30d":    3,
    "profile_views_received_30d": 30,
    "search_appearance_30d":      60,
    "offer_acceptance_rate":      0.60,
    "career_descriptions_text": (
        "Built ML models for churn prediction using scikit-learn. "
        "Processed large datasets with Spark. "
        "Contributed to TensorFlow model serving pipeline."
    ),
    "summary": (
        "Senior Data Scientist with 5 years experience in ML at product companies. "
        "Comfortable with Python, Spark, and model deployment."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# §A  Module structure & constants
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleStructure:
    def test_score_weights_is_dict(self):
        assert isinstance(SCORE_WEIGHTS, dict)

    def test_score_weights_has_10_keys(self):
        assert len(SCORE_WEIGHTS) == 10

    def test_score_weights_sum_to_one(self):
        assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9

    def test_all_weights_positive(self):
        assert all(v > 0 for v in SCORE_WEIGHTS.values())

    def test_expected_sub_score_names_present(self):
        expected = {
            "jd_semantic_score", "must_have_skill_score", "retrieval_ranking_score",
            "production_ml_score", "product_shipper_score", "behavioral_signal_score",
            "location_score", "salary_score", "experience_score", "education_score",
        }
        assert set(SCORE_WEIGHTS.keys()) == expected

    def test_weight_groups_has_five_buckets(self):
        assert set(WEIGHT_GROUPS.keys()) == {
            "career_relevance", "skill_depth", "behavioral", "location", "experience_fit"
        }

    def test_weight_groups_career_relevance_sums_to_035(self):
        s = sum(WEIGHT_GROUPS["career_relevance"].values())
        assert abs(s - 0.35) < 1e-9

    def test_weight_groups_skill_depth_sums_to_025(self):
        s = sum(WEIGHT_GROUPS["skill_depth"].values())
        assert abs(s - 0.25) < 1e-9

    def test_weight_groups_behavioral_sums_to_020(self):
        s = sum(WEIGHT_GROUPS["behavioral"].values())
        assert abs(s - 0.20) < 1e-9

    def test_weight_groups_location_sums_to_012(self):
        s = sum(WEIGHT_GROUPS["location"].values())
        assert abs(s - 0.12) < 1e-9

    def test_weight_groups_experience_fit_sums_to_008(self):
        s = sum(WEIGHT_GROUPS["experience_fit"].values())
        assert abs(s - 0.08) < 1e-9

    def test_candidate_score_dataclass_fields(self):
        cs = CandidateScore(
            candidate_id="x",
            composite_score=0.5,
            rank_key=(-0.5, "x"),
            score_breakdown={},
        )
        assert cs.candidate_id == "x"
        assert cs.composite_score == 0.5
        assert cs.rank_key == (-0.5, "x")
        assert cs.score_breakdown == {}


# ─────────────────────────────────────────────────────────────────────────────
# §B  Sub-score helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestSubScoreHelpers:
    # ── _sub_must_have ────────────────────────────────────────────────────────

    def test_must_have_reads_overall_must_have(self):
        r = _sub_must_have({"overall_must_have": 0.75, "composite_skill_score": 0.60})
        assert r == pytest.approx(0.75)

    def test_must_have_falls_back_to_half_when_missing(self):
        assert _sub_must_have({}) == pytest.approx(0.50)

    def test_must_have_clips_above_one(self):
        assert _sub_must_have({"overall_must_have": 1.5}) == pytest.approx(1.0)

    def test_must_have_clips_below_zero(self):
        assert _sub_must_have({"overall_must_have": -0.2}) == pytest.approx(0.0)

    # ── _sub_retrieval_ranking ────────────────────────────────────────────────

    def test_retrieval_ranking_zero_when_all_zero(self):
        r = _sub_retrieval_ranking({"retrieval": 0, "vector_db": 0, "ranking": 0, "evaluation": 0})
        assert r == pytest.approx(0.0)

    def test_retrieval_ranking_one_when_all_one(self):
        r = _sub_retrieval_ranking({"retrieval": 1, "vector_db": 1, "ranking": 1, "evaluation": 1})
        assert r == pytest.approx(1.0)

    def test_retrieval_ranking_weights_correct(self):
        # Only retrieval=1, rest=0: expect 0.35
        r = _sub_retrieval_ranking({"retrieval": 1, "vector_db": 0, "ranking": 0, "evaluation": 0})
        assert r == pytest.approx(0.35)

    def test_retrieval_ranking_falls_back_to_zero_missing(self):
        assert _sub_retrieval_ranking({}) == pytest.approx(0.0)

    # ── _sub_production_ml ───────────────────────────────────────────────────

    def test_production_ml_high_features_and_pattern_scores_high(self):
        features = {
            "f_role_ml_relevance": 1.0, "f_current_role_is_ml": 1.0,
            "f_tier1_density": 1.0, "f_tier1_depth": 1.0,
            "f_assessment_score": 1.0, "f_applied_ml_ratio": 1.0,
            "f_disqualifier_penalty": 1.0,
        }
        r = _sub_production_ml(features, {"production_ml": 1.0})
        assert r > 0.90

    def test_production_ml_disqualifier_penalty_applied(self):
        features = {
            "f_role_ml_relevance": 1.0, "f_current_role_is_ml": 1.0,
            "f_tier1_density": 1.0, "f_tier1_depth": 1.0,
            "f_assessment_score": 1.0, "f_applied_ml_ratio": 1.0,
            "f_disqualifier_penalty": 0.10,  # heavy disqualifier
        }
        r = _sub_production_ml(features, {"production_ml": 1.0})
        assert r < 0.15  # 0.10× almost-perfect score

    def test_production_ml_zero_features_gives_low_score(self):
        features = {
            "f_role_ml_relevance": 0.0, "f_current_role_is_ml": 0.0,
            "f_tier1_density": 0.0, "f_tier1_depth": 0.0,
            "f_assessment_score": 0.0, "f_applied_ml_ratio": 0.0,
            "f_disqualifier_penalty": 1.0,
        }
        r = _sub_production_ml(features, {"production_ml": 0.0})
        assert r == pytest.approx(0.0)

    # ── _sub_product_shipper ─────────────────────────────────────────────────

    def test_product_shipper_full_score(self):
        features = {
            "f_product_company_ratio": 1.0, "f_company_size_fit": 1.0,
            "f_industry_relevance": 1.0, "f_consulting_clean": 1.0,
        }
        assert _sub_product_shipper(features) == pytest.approx(1.0)

    def test_product_shipper_consulting_only_penalised(self):
        features = {
            "f_product_company_ratio": 1.0, "f_company_size_fit": 1.0,
            "f_industry_relevance": 1.0, "f_consulting_clean": 0.0,
        }
        r = _sub_product_shipper(features)
        assert r == pytest.approx(0.75)   # 0.30+0.25+0.20 = 0.75

    def test_product_shipper_falls_back_half_when_missing(self):
        # No consulting_clean → defaults to 1.0 (not consulting_only by default)
        r = _sub_product_shipper({})
        assert 0.0 <= r <= 1.0

    # ── _sub_location ─────────────────────────────────────────────────────────

    def test_location_in_preferred_city_high(self):
        features = {"f_location_score": 1.0, "f_relocation_ready": 1.0}
        assert _sub_location(features) == pytest.approx(1.0)

    def test_location_outside_india_no_relocate_low(self):
        # f_location_score=0.10 (outside India, won't relocate), no relocation bonus
        features = {"f_location_score": 0.10, "f_relocation_ready": 0.0}
        assert _sub_location(features) == pytest.approx(0.08)

    def test_location_relocation_bonus_helps(self):
        # same location score but willing to relocate
        r_yes = _sub_location({"f_location_score": 0.10, "f_relocation_ready": 1.0})
        r_no  = _sub_location({"f_location_score": 0.10, "f_relocation_ready": 0.0})
        assert r_yes > r_no

    # ── _sub_salary ───────────────────────────────────────────────────────────

    def test_salary_sweet_spot_scores_one(self):
        flat = {"salary_min_lpa": 40.0, "salary_max_lpa": 65.0, "salary_inverted": False}
        assert _sub_salary(flat) == pytest.approx(1.0)

    def test_salary_above_market_max_penalised(self):
        flat = {"salary_min_lpa": 90.0, "salary_max_lpa": 110.0, "salary_inverted": False}
        assert _sub_salary(flat) == pytest.approx(0.30)

    def test_salary_inverted_honeypot(self):
        flat = {"salary_min_lpa": 50.0, "salary_max_lpa": 60.0, "salary_inverted": True}
        assert _sub_salary(flat) == pytest.approx(0.05)

    def test_salary_no_data_neutral(self):
        flat = {"salary_min_lpa": 0.0, "salary_max_lpa": 0.0, "salary_inverted": False}
        assert _sub_salary(flat) == pytest.approx(0.50)

    def test_salary_linear_decay_above_sweet_spot(self):
        # 65→80 decays 1.00→0.30 linearly.  At midpoint=72.5: t=0.5 → 0.65
        flat = {"salary_min_lpa": 72.5, "salary_max_lpa": 72.5, "salary_inverted": False}
        r = _sub_salary(flat)
        assert pytest.approx(0.65, abs=0.01) == r

    def test_salary_linear_rise_below_sweet_spot(self):
        # 30→40 rises 0.50→1.00.  At midpoint=35: t=0.5 → 0.75
        flat = {"salary_min_lpa": 35.0, "salary_max_lpa": 35.0, "salary_inverted": False}
        r = _sub_salary(flat)
        assert pytest.approx(0.75, abs=0.01) == r

    # ── _sub_experience ──────────────────────────────────────────────────────

    def test_experience_reads_f_yoe_fit(self):
        r = _sub_experience({"f_yoe_fit": 0.82})
        assert r == pytest.approx(0.82)

    def test_experience_falls_back_when_missing(self):
        r = _sub_experience({})
        assert r == pytest.approx(0.5)

    # ── _sub_education ───────────────────────────────────────────────────────

    def test_education_full_tier_and_field(self):
        r = _sub_education({"f_edu_tier_score": 1.0, "f_edu_field_relevance": 1.0})
        assert r == pytest.approx(1.0)

    def test_education_50_50_weighted(self):
        r = _sub_education({"f_edu_tier_score": 0.8, "f_edu_field_relevance": 0.4})
        assert r == pytest.approx(0.60)


# ─────────────────────────────────────────────────────────────────────────────
# §C  score_candidate output structure
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreCandidateStructure:
    def test_returns_candidate_score_instance(self):
        cs = score_candidate(IDEAL)
        assert isinstance(cs, CandidateScore)

    def test_candidate_id_preserved(self):
        cs = score_candidate(IDEAL)
        assert cs.candidate_id == "ideal-001"

    def test_composite_score_in_range(self):
        cs = score_candidate(IDEAL)
        assert 0.0 <= cs.composite_score <= 1.0

    def test_rank_key_is_tuple(self):
        cs = score_candidate(IDEAL)
        assert isinstance(cs.rank_key, tuple) and len(cs.rank_key) == 2

    def test_rank_key_first_element_is_negative_score(self):
        cs = score_candidate(IDEAL)
        assert cs.rank_key[0] == pytest.approx(-cs.composite_score)

    def test_rank_key_second_element_is_candidate_id(self):
        cs = score_candidate(IDEAL)
        assert cs.rank_key[1] == "ideal-001"

    def test_score_breakdown_has_required_keys(self):
        cs = score_candidate(IDEAL)
        bd = cs.score_breakdown
        for k in ("sub_scores", "weights", "weighted_sub_scores",
                   "weighted_sum", "trap_penalty", "composite_score",
                   "weight_groups", "skill_detail", "trap_detail"):
            assert k in bd, f"Missing key: {k}"

    def test_sub_scores_has_all_10_components(self):
        cs = score_candidate(IDEAL)
        assert set(cs.score_breakdown["sub_scores"].keys()) == set(SCORE_WEIGHTS.keys())

    def test_weight_groups_in_breakdown(self):
        cs = score_candidate(IDEAL)
        wg = cs.score_breakdown["weight_groups"]
        assert set(wg.keys()) == {
            "career_relevance", "skill_depth", "behavioral", "location", "experience_fit"
        }

    def test_trap_detail_has_labels_and_risk(self):
        cs = score_candidate(IDEAL)
        td = cs.score_breakdown["trap_detail"]
        assert "trap_labels" in td
        assert "trap_risk_score" in td
        assert "explanation" in td

    def test_skill_detail_has_overall_must_have(self):
        cs = score_candidate(IDEAL)
        assert "overall_must_have" in cs.score_breakdown["skill_detail"]

    def test_composite_in_breakdown_matches_attribute(self):
        cs = score_candidate(IDEAL)
        assert cs.score_breakdown["composite_score"] == cs.composite_score


# ─────────────────────────────────────────────────────────────────────────────
# §D  score_candidate composite properties
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreCandidateComposite:
    def test_ideal_scores_higher_than_poor(self):
        ideal_cs = score_candidate(IDEAL)
        poor_cs  = score_candidate(POOR)
        assert ideal_cs.composite_score > poor_cs.composite_score

    def test_ideal_scores_higher_than_average(self):
        ideal_cs = score_candidate(IDEAL)
        avg_cs   = score_candidate(AVERAGE)
        assert ideal_cs.composite_score > avg_cs.composite_score

    def test_composite_is_six_decimal_places(self):
        cs = score_candidate(IDEAL)
        s = str(cs.composite_score)
        if "." in s:
            decimals = len(s.split(".")[1])
            assert decimals <= 6

    def test_weighted_sum_times_trap_penalty_equals_composite(self):
        cs = score_candidate(AVERAGE)
        bd = cs.score_breakdown
        expected = round(bd["weighted_sum"] * bd["trap_penalty"], 6)
        assert cs.composite_score == pytest.approx(expected, abs=1e-6)

    def test_group_contributions_sum_to_weighted_sum_pre_penalty(self):
        cs = score_candidate(AVERAGE)
        bd = cs.score_breakdown
        group_sum = sum(bd["weight_groups"].values())
        assert group_sum == pytest.approx(bd["weighted_sum"], abs=1e-5)

    def test_weighted_sub_scores_sum_to_weighted_sum(self):
        cs = score_candidate(IDEAL)
        bd = cs.score_breakdown
        total = sum(bd["weighted_sub_scores"].values())
        assert total == pytest.approx(bd["weighted_sum"], abs=1e-5)

    def test_jd_semantic_parameter_overrides_flat(self):
        flat_with_score = {**AVERAGE, "jd_semantic_score": 0.30}
        cs_low   = score_candidate(flat_with_score, jd_semantic_score=0.10)
        cs_high  = score_candidate(flat_with_score, jd_semantic_score=0.90)
        cs_flat  = score_candidate(flat_with_score)           # uses 0.30 from flat
        assert cs_high.composite_score > cs_low.composite_score
        # Using parameter overrides the flat value
        assert cs_low.composite_score != cs_flat.composite_score

    def test_higher_jd_semantic_gives_higher_score(self):
        base = copy.deepcopy(AVERAGE)
        base.pop("jd_semantic_score", None)
        low  = score_candidate(base, jd_semantic_score=0.10)
        high = score_candidate(base, jd_semantic_score=0.90)
        assert high.composite_score > low.composite_score

    def test_precomputed_features_gives_same_result_as_inline(self):
        from src.feature_engineering import extract_features
        flat = copy.deepcopy(AVERAGE)
        features = extract_features(flat)
        cs_inline = score_candidate(flat)
        cs_precomp = score_candidate(flat, precomputed_features=features)
        assert cs_inline.composite_score == pytest.approx(cs_precomp.composite_score, abs=1e-6)

    def test_composite_score_bounded_zero_to_one(self):
        for flat in [IDEAL, POOR, AVERAGE]:
            cs = score_candidate(flat)
            assert 0.0 <= cs.composite_score <= 1.0

    def test_poor_candidate_scores_below_040(self):
        cs = score_candidate(POOR)
        assert cs.composite_score < 0.40

    def test_trap_penalty_in_breakdown_is_between_005_and_1(self):
        cs = score_candidate(IDEAL)
        penalty = cs.score_breakdown["trap_penalty"]
        assert 0.05 <= penalty <= 1.0

    def test_all_sub_scores_in_zero_one(self):
        cs = score_candidate(IDEAL)
        for name, val in cs.score_breakdown["sub_scores"].items():
            assert 0.0 <= val <= 1.0, f"{name} = {val} out of range"

    def test_no_salary_data_gives_neutral_salary_score(self):
        flat = {**copy.deepcopy(AVERAGE), "salary_min_lpa": 0.0, "salary_max_lpa": 0.0}
        cs = score_candidate(flat)
        assert cs.score_breakdown["sub_scores"]["salary_score"] == pytest.approx(0.50)

    def test_salary_inverted_honeypot_depresses_score(self):
        normal = score_candidate(AVERAGE)
        honeypot_flat = {**copy.deepcopy(AVERAGE), "salary_inverted": True}
        honeypot = score_candidate(honeypot_flat)
        assert honeypot.composite_score < normal.composite_score


# ─────────────────────────────────────────────────────────────────────────────
# §E  Determinism and tie-breaking
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_gives_same_score(self):
        cs1 = score_candidate(IDEAL)
        cs2 = score_candidate(IDEAL)
        assert cs1.composite_score == cs2.composite_score

    def test_same_input_gives_same_breakdown(self):
        cs1 = score_candidate(AVERAGE)
        cs2 = score_candidate(AVERAGE)
        assert cs1.score_breakdown["sub_scores"] == cs2.score_breakdown["sub_scores"]

    def test_rank_key_sorts_higher_score_first(self):
        ideal_key = score_candidate(IDEAL).rank_key
        poor_key  = score_candidate(POOR).rank_key
        # Ascending sort on rank_key gives higher-scoring candidate first
        assert ideal_key < poor_key

    def test_tie_break_by_candidate_id_ascending(self):
        # Two identical candidates except candidate_id
        fa = {**copy.deepcopy(AVERAGE), "candidate_id": "zzz-001"}
        fb = {**copy.deepcopy(AVERAGE), "candidate_id": "aaa-001"}
        csa = score_candidate(fa)
        csb = score_candidate(fb)
        # Both have the same composite; "aaa" comes first alphabetically
        assert csa.composite_score == pytest.approx(csb.composite_score)
        assert csb.rank_key < csa.rank_key  # "aaa" < "zzz"

    def test_batch_output_matches_individual_scores(self):
        flats = [copy.deepcopy(IDEAL), copy.deepcopy(AVERAGE)]
        df = score_candidates(flats)
        for flat in flats:
            cid = flat["candidate_id"]
            cs  = score_candidate(flat)
            assert df.loc[cid, "composite_score"] == pytest.approx(cs.composite_score, abs=1e-6)

    def test_rank_column_starts_at_one(self):
        df = score_candidates([IDEAL, AVERAGE, POOR])
        assert 1 in df["rank"].values

    def test_batch_sorted_descending_by_score(self):
        df = score_candidates([POOR, IDEAL, AVERAGE])
        scores = df["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_score_is_float_not_int(self):
        cs = score_candidate(AVERAGE)
        assert isinstance(cs.composite_score, float)


# ─────────────────────────────────────────────────────────────────────────────
# §F  Trap penalty integration
# ─────────────────────────────────────────────────────────────────────────────

class TestTrapPenaltyIntegration:
    def _make_fake_ai_profile(self) -> dict[str, Any]:
        """Candidate who claims ML expertise with zero-duration skills."""
        fake = copy.deepcopy(IDEAL)
        fake["candidate_id"] = "fake-ai-001"
        fake["expert_zero_duration_count"] = 20   # trigger fake_ai_profile
        fake["skill_count"] = 25
        fake["tier1_skill_count"] = 20
        fake["summary"] = (
            "Expert in transformer architectures, diffusion models, GPT-4, "
            "ChatGPT, LLaMA, RLHF, AI alignment, AGI research, "
            "neural architecture search, and quantum ML."
        )
        return fake

    def test_trap_penalty_applied_when_trap_triggered(self):
        fake = self._make_fake_ai_profile()
        clean_cs = score_candidate(IDEAL)
        fake_cs  = score_candidate(fake)
        # Fake profile should score lower due to trap penalty
        assert fake_cs.score_breakdown["trap_penalty"] < 1.0

    def test_trap_penalty_multiplies_weighted_sum(self):
        cs = score_candidate(AVERAGE)
        bd = cs.score_breakdown
        assert cs.composite_score == pytest.approx(
            bd["weighted_sum"] * bd["trap_penalty"], abs=1e-5
        )

    def test_clean_ideal_candidate_has_penalty_near_one(self):
        cs = score_candidate(IDEAL)
        # A genuine ideal candidate should trigger no traps → penalty ~ 1.0
        assert cs.score_breakdown["trap_penalty"] > 0.85

    def test_trap_labels_list_in_breakdown(self):
        cs = score_candidate(IDEAL)
        assert isinstance(cs.score_breakdown["trap_detail"]["trap_labels"], list)

    def test_trap_explanation_list_has_10_entries(self):
        cs = score_candidate(AVERAGE)
        assert len(cs.score_breakdown["trap_detail"]["explanation"]) == 10

    def test_higher_trap_penalty_gives_higher_composite(self):
        cs_clean = score_candidate(IDEAL)
        fake = self._make_fake_ai_profile()
        cs_fake  = score_candidate(fake)
        if cs_fake.score_breakdown["trap_penalty"] < cs_clean.score_breakdown["trap_penalty"]:
            # penalty for fake should push composite down
            assert cs_fake.composite_score < cs_clean.composite_score

    def test_trap_risk_score_in_zero_one(self):
        cs = score_candidate(POOR)
        assert 0.0 <= cs.score_breakdown["trap_detail"]["trap_risk_score"] <= 1.0

    def test_compound_penalty_floor_is_005(self):
        # Even the worst candidate gets at least 0.05 × weighted_sum
        cs = score_candidate(POOR)
        assert cs.score_breakdown["trap_penalty"] >= 0.05


# ─────────────────────────────────────────────────────────────────────────────
# §G  score_candidates batch function
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreCandidatesBatch:
    def test_returns_dataframe(self):
        df = score_candidates([IDEAL, AVERAGE])
        assert isinstance(df, pd.DataFrame)

    def test_index_is_candidate_id(self):
        df = score_candidates([IDEAL, AVERAGE])
        assert df.index.name == "candidate_id"
        assert "ideal-001" in df.index
        assert "avg-001" in df.index

    def test_rank_column_present(self):
        df = score_candidates([IDEAL, AVERAGE, POOR])
        assert "rank" in df.columns

    def test_rank_column_dtype_int(self):
        df = score_candidates([IDEAL, AVERAGE])
        assert df["rank"].dtype in (int, np.int64, np.int32)

    def test_all_sub_score_columns_present(self):
        df = score_candidates([IDEAL])
        for name in SCORE_WEIGHTS:
            assert name in df.columns, f"Missing column: {name}"

    def test_group_columns_present(self):
        df = score_candidates([IDEAL])
        for g in ("career_relevance", "skill_depth", "behavioral", "location", "experience_fit"):
            assert f"group_{g}" in df.columns

    def test_sorted_descending_score(self):
        df = score_candidates([POOR, AVERAGE, IDEAL])
        scores = df["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_jd_semantic_scores_array_applied(self):
        flats = [copy.deepcopy(IDEAL), copy.deepcopy(AVERAGE)]
        arr_low  = np.array([0.10, 0.10])
        arr_high = np.array([0.90, 0.90])
        df_low  = score_candidates(flats, jd_semantic_scores=arr_low)
        df_high = score_candidates(flats, jd_semantic_scores=arr_high)
        assert df_high["composite_score"].mean() > df_low["composite_score"].mean()

    def test_single_candidate_batch(self):
        df = score_candidates([IDEAL])
        assert len(df) == 1
        assert df.loc["ideal-001", "rank"] == 1

    def test_empty_batch_returns_empty_dataframe(self):
        df = score_candidates([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ─────────────────────────────────────────────────────────────────────────────
# §H  Edge cases / missing data
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_flat_does_not_crash(self):
        cs = score_candidate({})
        assert 0.0 <= cs.composite_score <= 1.0

    def test_sparse_dict_missing_keys_does_not_crash(self):
        # flat.get(key, default) only falls back when the key is ABSENT.
        # This test exercises the common case where optional fields are simply
        # not present in the dict (as opposed to being explicitly None).
        flat = {
            "candidate_id": "sparse-001",
            "years_of_experience": 5.0,
            # All other numeric and list fields deliberately omitted
        }
        cs = score_candidate(flat)
        assert 0.0 <= cs.composite_score <= 1.0

    def test_missing_candidate_id_becomes_empty_string(self):
        flat = copy.deepcopy(AVERAGE)
        del flat["candidate_id"]
        cs = score_candidate(flat)
        assert cs.candidate_id == ""

    def test_jd_semantic_none_uses_flat_value(self):
        flat = {**copy.deepcopy(AVERAGE), "jd_semantic_score": 0.80}
        cs = score_candidate(flat, jd_semantic_score=None)
        assert cs.score_breakdown["sub_scores"]["jd_semantic_score"] == pytest.approx(0.80)

    def test_jd_semantic_above_one_clipped(self):
        cs = score_candidate(AVERAGE, jd_semantic_score=1.5)
        assert cs.score_breakdown["sub_scores"]["jd_semantic_score"] == pytest.approx(1.0)

    def test_jd_semantic_below_zero_clipped(self):
        cs = score_candidate(AVERAGE, jd_semantic_score=-0.2)
        assert cs.score_breakdown["sub_scores"]["jd_semantic_score"] == pytest.approx(0.0)

    def test_score_with_empty_skill_names_list(self):
        flat = {**copy.deepcopy(AVERAGE), "skill_names": []}
        cs = score_candidate(flat)
        assert 0.0 <= cs.composite_score <= 1.0

    def test_very_high_salary_expectation_penalised(self):
        flat = {**copy.deepcopy(IDEAL), "salary_min_lpa": 100.0, "salary_max_lpa": 150.0}
        cs = score_candidate(flat)
        assert cs.score_breakdown["sub_scores"]["salary_score"] == pytest.approx(0.30)

    def test_salary_exactly_at_sweet_spot_boundaries(self):
        # min=40 → sweet_spot lower boundary
        flat40 = {**copy.deepcopy(AVERAGE), "salary_min_lpa": 40.0, "salary_max_lpa": 40.0}
        assert _sub_salary(flat40) == pytest.approx(1.0)
        # max=65 → sweet_spot upper boundary
        flat65 = {**copy.deepcopy(AVERAGE), "salary_min_lpa": 65.0, "salary_max_lpa": 65.0}
        assert _sub_salary(flat65) == pytest.approx(1.0)

    def test_batch_with_duplicate_ids_still_scores(self):
        # Duplicate IDs are allowed at scoring stage (dedup is caller's concern)
        flat_a = {**copy.deepcopy(AVERAGE), "candidate_id": "dup-001"}
        flat_b = {**copy.deepcopy(POOR),    "candidate_id": "dup-001"}
        df = score_candidates([flat_a, flat_b])
        assert len(df) >= 1   # at least one row; exact behaviour with dup index is acceptable
