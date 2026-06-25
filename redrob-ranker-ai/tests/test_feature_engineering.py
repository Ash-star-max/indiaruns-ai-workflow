"""
Tests for src/feature_engineering.py

§A  Module structure — FEATURE_NAMES, extract_features keys/types/range
§B  Experience features
§C  Current role features
§D  Company features
§E  Skill features
§F  Education features
§G  Certification features
§H  Language features
§I  Location & availability features
§J  Disqualifier features
§K  build_feature_matrix — shape, columns, index
§L  Integration — ideal vs disqualified candidate
§M  Edge cases — missing fields, empty strings, zeros
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FEATURE_NAMES,
    build_feature_matrix,
    extract_features,
    score_activity,
    score_cert_relevance,
    score_company_size,
    score_edu_field,
    score_edu_tier,
    score_english_proficiency,
    score_industry,
    score_role_ml_relevance,
    score_seniority,
    score_yoe_fit,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

# A rich, JD-aligned candidate flat dict — mirrors the MINIMAL_DICT from test_load_data.py
_IDEAL_FLAT: dict[str, Any] = {
    "candidate_id":              "IDEAL_001",
    "years_of_experience":       7.0,
    "current_title":             "Senior ML Engineer",
    "current_company":           "FinTech AI Startup",
    "current_company_size":      "201-500",
    "current_industry":          "Technology",
    "career_titles":             ["Senior ML Engineer", "ML Engineer", "Data Scientist"],
    "career_companies":          ["FinTech AI Startup", "ProductCo", "DataCo"],
    "total_career_months":       84,
    "n_career_roles":            3,
    "most_recent_title":         "Senior ML Engineer",
    "most_recent_company":       "FinTech AI Startup",
    "product_company_ratio":     1.0,
    "is_consulting_only":        False,
    "skill_names":               ["FAISS", "Python", "NLP", "PyTorch", "NDCG", "Pinecone"],
    "skill_count":               6,
    "tier1_skill_count":         5,
    "tier2_skill_count":         2,
    "expert_zero_duration_count":0,
    "tier1_avg_duration_months": 30.0,
    "tier2_avg_duration_months": 18.0,
    "total_skill_endorsements":  80,
    "highest_edu_tier":          "tier_1",
    "edu_fields":                ["Computer Science"],
    "edu_count":                 1,
    "cert_names":                ["AWS Machine Learning Specialty"],
    "cert_count":                1,
    "skill_assessment_scores":   {"Python": 92.0, "NLP": 85.0},
    "language_names":            ["English", "Hindi"],
    "language_count":            2,
    "english_proficiency":       "professional",
    "country":                   "India",
    "location":                  "Noida",
    "willing_to_relocate":       True,
    "notice_period_days":        30,
    "days_since_last_active":    5,
    "open_to_work_flag":         True,
    "recruiter_response_rate":   0.85,
    "profile_completeness_score": 90.0,
    # other signals used by compute_disqualifier_penalty
    "career_descriptions_text":  "Built FAISS retrieval at scale",
}

# Disqualified candidate — consulting-only, HR Manager title, LangChain only
_BAD_FLAT: dict[str, Any] = {
    "candidate_id":              "BAD_001",
    "years_of_experience":       8.0,
    "current_title":             "HR Manager",
    "current_company":           "TCS",
    "current_company_size":      "10001+",
    "current_industry":          "IT Services",
    "career_titles":             ["HR Manager", "HR Executive", "Recruiter"],
    "career_companies":          ["TCS", "Infosys", "Wipro"],
    "total_career_months":       96,
    "n_career_roles":            3,
    "most_recent_title":         "HR Manager",
    "most_recent_company":       "TCS",
    "product_company_ratio":     0.0,
    "is_consulting_only":        True,
    "skill_names":               ["langchain"],
    "skill_count":               1,
    "tier1_skill_count":         0,
    "tier2_skill_count":         0,
    "expert_zero_duration_count":0,
    "tier1_avg_duration_months": 0.0,
    "tier2_avg_duration_months": 0.0,
    "total_skill_endorsements":  0,
    "highest_edu_tier":          "tier_4",
    "edu_fields":                ["Business Administration"],
    "edu_count":                 1,
    "cert_names":                ["HR Certification"],
    "cert_count":                1,
    "skill_assessment_scores":   {},
    "language_names":            ["English"],
    "language_count":            1,
    "english_proficiency":       "intermediate",
    "country":                   "United States",
    "location":                  "New York",
    "willing_to_relocate":       False,
    "notice_period_days":        120,
    "days_since_last_active":    200,
    "open_to_work_flag":         False,
    "recruiter_response_rate":   0.05,
    "profile_completeness_score": 30.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# §A  Module structure
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleStructure:

    def test_feature_names_is_list(self):
        assert isinstance(FEATURE_NAMES, list)

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 27

    def test_feature_names_are_strings(self):
        assert all(isinstance(n, str) for n in FEATURE_NAMES)

    def test_feature_names_start_with_f_prefix(self):
        assert all(n.startswith("f_") for n in FEATURE_NAMES)

    def test_no_duplicate_feature_names(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_extract_features_returns_dict(self):
        result = extract_features(_IDEAL_FLAT)
        assert isinstance(result, dict)

    def test_extract_features_has_all_keys(self):
        result = extract_features(_IDEAL_FLAT)
        missing = set(FEATURE_NAMES) - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_extract_features_no_extra_keys(self):
        result = extract_features(_IDEAL_FLAT)
        extra = set(result.keys()) - set(FEATURE_NAMES)
        assert not extra, f"Unexpected keys: {extra}"

    def test_all_values_are_floats(self):
        result = extract_features(_IDEAL_FLAT)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} = {v!r} is not float"

    def test_all_values_in_0_1(self):
        result = extract_features(_IDEAL_FLAT)
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"{k} = {v:.4f} out of [0,1]"

    def test_all_values_in_0_1_bad_candidate(self):
        result = extract_features(_BAD_FLAT)
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"{k} = {v:.4f} out of [0,1]"


# ─────────────────────────────────────────────────────────────────────────────
# §B  Experience features
# ─────────────────────────────────────────────────────────────────────────────

class TestExperienceFeatures:

    def test_yoe_fit_peaks_at_seven(self):
        assert score_yoe_fit(7.0) > 0.95

    def test_yoe_fit_lower_at_two_than_seven(self):
        assert score_yoe_fit(2.0) < score_yoe_fit(7.0)

    def test_yoe_fit_lower_at_fifteen_than_seven(self):
        assert score_yoe_fit(15.0) < score_yoe_fit(7.0)

    def test_yoe_fit_floor_at_zero(self):
        assert score_yoe_fit(0.0) == pytest.approx(0.05)

    def test_yoe_fit_symmetric_around_peak(self):
        # 7-2=5 and 7+2=9 should give same score (Gaussian)
        assert score_yoe_fit(5.0) == pytest.approx(score_yoe_fit(9.0), abs=0.01)

    def test_f_yoe_normalized_cap(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["years_of_experience"] = 15.0
        result = extract_features(flat)
        assert result["f_yoe_normalized"] == pytest.approx(1.0)

    def test_f_applied_ml_ratio_all_ml_titles(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["career_titles"] = ["ML Engineer", "Data Scientist", "NLP Engineer"]
        result = extract_features(flat)
        assert result["f_applied_ml_ratio"] == pytest.approx(1.0)

    def test_f_applied_ml_ratio_no_ml_titles(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["career_titles"] = ["Sales Manager", "HR Executive", "Accountant"]
        result = extract_features(flat)
        assert result["f_applied_ml_ratio"] == pytest.approx(0.0)

    def test_f_applied_ml_ratio_mixed_titles(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["career_titles"] = ["ML Engineer", "Sales Manager"]
        result = extract_features(flat)
        assert result["f_applied_ml_ratio"] == pytest.approx(0.5)

    def test_f_career_seniority_senior_title(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["most_recent_title"] = "Senior ML Engineer"
        result = extract_features(flat)
        assert result["f_career_seniority"] >= 0.65

    def test_f_career_seniority_intern_lower_than_senior(self):
        flat_intern = deepcopy(_IDEAL_FLAT)
        flat_intern["most_recent_title"] = "ML Intern"
        flat_senior = deepcopy(_IDEAL_FLAT)
        flat_senior["most_recent_title"] = "Senior ML Engineer"
        intern_score  = extract_features(flat_intern)["f_career_seniority"]
        senior_score  = extract_features(flat_senior)["f_career_seniority"]
        assert intern_score < senior_score

    def test_f_career_length_score_capped(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["total_career_months"] = 200  # > 120
        result = extract_features(flat)
        assert result["f_career_length_score"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# §C  score_seniority & score_role_ml_relevance
# ─────────────────────────────────────────────────────────────────────────────

class TestRoleScorers:

    def test_seniority_senior_ml_engineer(self):
        assert score_seniority("Senior ML Engineer") == pytest.approx(0.70)

    def test_seniority_staff_engineer(self):
        assert score_seniority("Staff ML Engineer") == pytest.approx(0.80)

    def test_seniority_principal(self):
        assert score_seniority("Principal Engineer") == pytest.approx(0.85)

    def test_seniority_intern_lowest(self):
        assert score_seniority("ML Intern") == pytest.approx(0.10)

    def test_seniority_junior(self):
        assert score_seniority("Junior Data Scientist") == pytest.approx(0.25)

    def test_seniority_vp_penalised(self):
        assert score_seniority("VP of Engineering") <= 0.40

    def test_seniority_empty_returns_midpoint(self):
        assert score_seniority("") == pytest.approx(0.45)

    def test_role_ml_relevance_ml_title_full_score(self):
        assert score_role_ml_relevance("Senior ML Engineer") == pytest.approx(1.00)

    def test_role_ml_relevance_data_scientist_full_score(self):
        assert score_role_ml_relevance("Data Scientist") == pytest.approx(1.00)

    def test_role_ml_relevance_swe_partial(self):
        s = score_role_ml_relevance("Software Engineer")
        assert 0.30 <= s <= 0.50

    def test_role_ml_relevance_hr_low(self):
        assert score_role_ml_relevance("HR Manager") < 0.30

    def test_role_ml_relevance_empty_zero(self):
        assert score_role_ml_relevance("") == pytest.approx(0.00)

    def test_f_current_role_is_ml_binary(self):
        # ML title → 1.0
        ideal = extract_features(_IDEAL_FLAT)
        assert ideal["f_current_role_is_ml"] == pytest.approx(1.0)
        # HR Manager → 0.0
        bad = extract_features(_BAD_FLAT)
        assert bad["f_current_role_is_ml"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# §D  Company features
# ─────────────────────────────────────────────────────────────────────────────

class TestCompanyFeatures:

    def test_company_size_sweet_spot(self):
        assert score_company_size("201-500") == pytest.approx(1.00)

    def test_company_size_tiny(self):
        assert score_company_size("1-10") < score_company_size("201-500")

    def test_company_size_huge(self):
        assert score_company_size("10001+") < score_company_size("201-500")

    def test_company_size_unknown_fallback(self):
        s = score_company_size("unknown_band")
        assert 0.0 <= s <= 1.0

    def test_industry_technology_max(self):
        assert score_industry("Technology") == pytest.approx(1.00)

    def test_industry_saas_max(self):
        assert score_industry("SaaS platform") == pytest.approx(1.00)

    def test_industry_banking_adjacent(self):
        s = score_industry("Banking & Financial Services")
        assert 0.50 <= s < 1.00

    def test_industry_manufacturing_low(self):
        assert score_industry("Manufacturing") < 0.50

    def test_f_product_company_ratio_consulting_only(self):
        bad = extract_features(_BAD_FLAT)
        assert bad["f_product_company_ratio"] == pytest.approx(0.0)

    def test_f_product_company_ratio_product_only(self):
        ideal = extract_features(_IDEAL_FLAT)
        assert ideal["f_product_company_ratio"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# §E  Skill features
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillFeatures:

    def test_f_tier1_density_zero_skills(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["tier1_skill_count"] = 0
        flat["skill_count"]       = 0
        result = extract_features(flat)
        assert result["f_tier1_density"] == pytest.approx(0.0)

    def test_f_tier1_density_five_tier1_gives_full_score(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["tier1_skill_count"] = 5
        flat["skill_count"] = 5
        result = extract_features(flat)
        assert result["f_tier1_density"] == pytest.approx(1.0)

    def test_f_tier1_density_capped_at_1(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["tier1_skill_count"] = 10
        flat["skill_count"] = 10
        result = extract_features(flat)
        assert result["f_tier1_density"] == pytest.approx(1.0)

    def test_f_tier1_depth_zero_duration(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["tier1_avg_duration_months"] = 0.0
        result = extract_features(flat)
        assert result["f_tier1_depth"] == pytest.approx(0.0)

    def test_f_tier1_depth_24_months_full(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["tier1_avg_duration_months"] = 24.0
        result = extract_features(flat)
        assert result["f_tier1_depth"] == pytest.approx(1.0)

    def test_f_tier1_depth_caps_at_1(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["tier1_avg_duration_months"] = 48.0
        result = extract_features(flat)
        assert result["f_tier1_depth"] == pytest.approx(1.0)

    def test_f_tier2_breadth_eight_gives_full(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["tier2_skill_count"] = 8
        result = extract_features(flat)
        assert result["f_tier2_breadth"] == pytest.approx(1.0)

    def test_f_skill_endorsement_score_zero(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["total_skill_endorsements"] = 0
        result = extract_features(flat)
        assert result["f_skill_endorsement_score"] == pytest.approx(0.0)

    def test_f_assessment_score_no_assessments_neutral(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["skill_assessment_scores"] = {}
        result = extract_features(flat)
        assert result["f_assessment_score"] == pytest.approx(0.50)

    def test_f_assessment_score_perfect(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["skill_assessment_scores"] = {"Python": 100.0, "NLP": 100.0}
        result = extract_features(flat)
        assert result["f_assessment_score"] == pytest.approx(1.00)

    def test_f_assessment_score_average_computed(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["skill_assessment_scores"] = {"A": 80.0, "B": 60.0}
        result = extract_features(flat)
        assert result["f_assessment_score"] == pytest.approx(0.70, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# §F  Education features
# ─────────────────────────────────────────────────────────────────────────────

class TestEducationFeatures:

    @pytest.mark.parametrize("tier, expected", [
        ("tier_1",  1.00),
        ("tier_2",  0.80),
        ("tier_3",  0.55),
        ("tier_4",  0.35),
        ("unknown", 0.20),
    ])
    def test_edu_tier_score(self, tier, expected):
        assert score_edu_tier(tier) == pytest.approx(expected)

    def test_edu_tier_invalid_maps_to_unknown(self):
        assert score_edu_tier("tier_99") == pytest.approx(0.20)

    @pytest.mark.parametrize("field, expected_min", [
        ("Computer Science",     0.99),
        ("Data Science",         0.99),
        ("Mathematics",          0.99),
        ("Physics",              0.60),
        ("Electrical Engineering", 0.60),
        ("Business Administration", 0.15),
    ])
    def test_edu_field_relevance(self, field, expected_min):
        s = score_edu_field(field)
        assert s >= expected_min, f"{field} → {s:.2f}, expected >= {expected_min}"

    def test_f_edu_tier_score_tier1_in_result(self):
        result = extract_features(_IDEAL_FLAT)
        assert result["f_edu_tier_score"] == pytest.approx(1.00)

    def test_f_edu_field_relevance_cs(self):
        result = extract_features(_IDEAL_FLAT)
        assert result["f_edu_field_relevance"] > 0.90

    def test_f_edu_field_best_of_multiple_fields(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["edu_fields"] = ["Business Administration", "Computer Science"]
        result = extract_features(flat)
        # Should take the best (CS = ~1.0)
        assert result["f_edu_field_relevance"] > 0.90

    def test_f_edu_no_education_defaults(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["edu_fields"]       = []
        flat["highest_edu_tier"] = "unknown"
        result = extract_features(flat)
        assert result["f_edu_field_relevance"] == pytest.approx(0.20)
        assert result["f_edu_tier_score"] == pytest.approx(0.20)


# ─────────────────────────────────────────────────────────────────────────────
# §G  Certification features
# ─────────────────────────────────────────────────────────────────────────────

class TestCertificationFeatures:

    def test_no_certs_returns_zeros(self):
        ml_rel, count = score_cert_relevance([])
        assert ml_rel == 0.0
        assert count == 0.0

    def test_ml_cert_detected(self):
        ml_rel, _ = score_cert_relevance(["AWS Machine Learning Specialty"])
        assert ml_rel == pytest.approx(1.0)

    def test_non_ml_cert_low_relevance(self):
        ml_rel, _ = score_cert_relevance(["PMP Certification", "Six Sigma Green Belt"])
        assert ml_rel == pytest.approx(0.0)

    def test_mixed_certs_partial_relevance(self):
        ml_rel, _ = score_cert_relevance([
            "AWS Machine Learning Specialty",
            "PMP Certification",
        ])
        assert ml_rel == pytest.approx(0.5)

    def test_count_score_five_certs_full(self):
        _, count = score_cert_relevance(["c1", "c2", "c3", "c4", "c5"])
        assert count == pytest.approx(1.0)

    def test_count_score_one_cert_partial(self):
        _, count = score_cert_relevance(["AWS Machine Learning Specialty"])
        assert count == pytest.approx(0.20)

    def test_f_cert_ml_relevance_in_result(self):
        result = extract_features(_IDEAL_FLAT)
        assert result["f_cert_ml_relevance"] == pytest.approx(1.0)

    def test_f_cert_ml_relevance_non_ml_cert(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["cert_names"] = ["Six Sigma Green Belt", "PMP"]
        result = extract_features(flat)
        assert result["f_cert_ml_relevance"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# §H  Language features
# ─────────────────────────────────────────────────────────────────────────────

class TestLanguageFeatures:

    @pytest.mark.parametrize("prof, expected", [
        ("native",        1.00),
        ("fluent",        0.95),
        ("professional",  0.90),
        ("advanced",      0.80),
        ("intermediate",  0.60),
        ("elementary",    0.35),
        ("beginner",      0.20),
        ("unknown",       0.70),
    ])
    def test_english_proficiency_scores(self, prof, expected):
        assert score_english_proficiency(prof) == pytest.approx(expected)

    def test_english_proficiency_unknown_string_defaults(self):
        s = score_english_proficiency("conversational")  # not in table
        assert 0.0 <= s <= 1.0

    def test_f_english_proficiency_professional(self):
        result = extract_features(_IDEAL_FLAT)
        assert result["f_english_proficiency_score"] == pytest.approx(0.90)

    def test_f_language_diversity_two_languages(self):
        result = extract_features(_IDEAL_FLAT)
        assert result["f_language_diversity"] == pytest.approx(0.5)   # 2/4

    def test_f_language_diversity_four_plus_full(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["language_count"] = 4
        result = extract_features(flat)
        assert result["f_language_diversity"] == pytest.approx(1.0)

    def test_f_language_diversity_zero(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["language_count"]    = 0
        flat["english_proficiency"] = "unknown"
        result = extract_features(flat)
        assert result["f_language_diversity"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# §I  Location & availability features
# ─────────────────────────────────────────────────────────────────────────────

class TestLocationAvailabilityFeatures:

    def test_f_location_score_india_noida_perfect(self):
        result = extract_features(_IDEAL_FLAT)
        assert result["f_location_score"] == pytest.approx(1.00)

    def test_f_location_score_outside_india_no_reloc(self):
        result = extract_features(_BAD_FLAT)
        assert result["f_location_score"] <= 0.15

    def test_f_notice_score_30_days_perfect(self):
        result = extract_features(_IDEAL_FLAT)
        assert result["f_notice_score"] == pytest.approx(1.00)

    def test_f_notice_score_120_days_low(self):
        result = extract_features(_BAD_FLAT)
        assert result["f_notice_score"] <= 0.45

    def test_f_relocation_ready_true(self):
        result = extract_features(_IDEAL_FLAT)
        assert result["f_relocation_ready"] == pytest.approx(1.0)

    def test_f_relocation_ready_false(self):
        result = extract_features(_BAD_FLAT)
        assert result["f_relocation_ready"] == pytest.approx(0.0)

    def test_score_activity_recent_high(self):
        s = score_activity(days_since_active=5, open_to_work=True)
        assert s > 0.90

    def test_score_activity_stale_low(self):
        s = score_activity(days_since_active=365, open_to_work=False)
        assert s < 0.10

    def test_score_activity_open_to_work_boosts(self):
        without = score_activity(days_since_active=30, open_to_work=False)
        with_otw = score_activity(days_since_active=30, open_to_work=True)
        assert with_otw > without

    def test_score_activity_sentinel_9999_very_low(self):
        s = score_activity(days_since_active=9999, open_to_work=False)
        assert s <= 0.10

    def test_f_activity_score_ideal_candidate(self):
        result = extract_features(_IDEAL_FLAT)
        assert result["f_activity_score"] > 0.90


# ─────────────────────────────────────────────────────────────────────────────
# §J  Disqualifier features
# ─────────────────────────────────────────────────────────────────────────────

class TestDisqualifierFeatures:

    def test_f_disqualifier_penalty_clean_candidate_is_one(self):
        result = extract_features(_IDEAL_FLAT)
        assert result["f_disqualifier_penalty"] == pytest.approx(1.00)

    def test_f_disqualifier_penalty_consulting_only_reduced(self):
        result = extract_features(_BAD_FLAT)
        assert result["f_disqualifier_penalty"] < 0.30

    def test_f_consulting_clean_product_company(self):
        result = extract_features(_IDEAL_FLAT)
        assert result["f_consulting_clean"] == pytest.approx(1.0)

    def test_f_consulting_clean_consulting_only(self):
        result = extract_features(_BAD_FLAT)
        assert result["f_consulting_clean"] == pytest.approx(0.0)

    def test_multiple_disqualifiers_compound(self):
        # BAD_FLAT has: consulting_only + nontechnical_title + langchain_only + outside_india
        bad_penalty = extract_features(_BAD_FLAT)["f_disqualifier_penalty"]
        # consulting alone = 0.15; stacking more should make it even lower
        assert bad_penalty < 0.10


# ─────────────────────────────────────────────────────────────────────────────
# §K  build_feature_matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeatureMatrix:

    @pytest.fixture(scope="class")
    def matrix(self):
        return build_feature_matrix([_IDEAL_FLAT, _BAD_FLAT])

    def test_returns_dataframe(self, matrix):
        assert isinstance(matrix, pd.DataFrame)

    def test_shape_rows(self, matrix):
        assert len(matrix) == 2

    def test_shape_cols(self, matrix):
        assert matrix.shape[1] == 27

    def test_columns_match_feature_names(self, matrix):
        assert list(matrix.columns) == FEATURE_NAMES

    def test_index_is_candidate_id(self, matrix):
        assert "IDEAL_001" in matrix.index
        assert "BAD_001" in matrix.index

    def test_index_name(self, matrix):
        assert matrix.index.name == "candidate_id"

    def test_dtype_float32(self, matrix):
        assert matrix.dtypes.unique().tolist() == [np.float32]

    def test_all_values_in_0_1(self, matrix):
        assert (matrix >= 0.0).all().all()
        assert (matrix <= 1.0).all().all()

    def test_accepts_dataframe_input(self):
        df_input = pd.DataFrame([_IDEAL_FLAT, _BAD_FLAT])
        matrix = build_feature_matrix(df_input)
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape == (2, 27)

    def test_empty_list_returns_empty_df(self):
        matrix = build_feature_matrix([])
        assert len(matrix) == 0
        assert list(matrix.columns) == FEATURE_NAMES

    def test_single_candidate(self):
        matrix = build_feature_matrix([_IDEAL_FLAT])
        assert matrix.shape == (1, 27)


# ─────────────────────────────────────────────────────────────────────────────
# §L  Integration — ideal vs disqualified
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    @pytest.fixture(scope="class")
    def ideal_features(self):
        return extract_features(_IDEAL_FLAT)

    @pytest.fixture(scope="class")
    def bad_features(self):
        return extract_features(_BAD_FLAT)

    def test_ideal_has_higher_experience_fit(self, ideal_features, bad_features):
        assert ideal_features["f_yoe_fit"] > bad_features["f_yoe_fit"]

    def test_ideal_has_higher_product_ratio(self, ideal_features, bad_features):
        assert ideal_features["f_product_company_ratio"] > bad_features["f_product_company_ratio"]

    def test_ideal_has_higher_tier1_density(self, ideal_features, bad_features):
        assert ideal_features["f_tier1_density"] > bad_features["f_tier1_density"]

    def test_ideal_has_higher_edu_tier(self, ideal_features, bad_features):
        assert ideal_features["f_edu_tier_score"] > bad_features["f_edu_tier_score"]

    def test_ideal_has_higher_location_score(self, ideal_features, bad_features):
        assert ideal_features["f_location_score"] > bad_features["f_location_score"]

    def test_ideal_has_higher_notice_score(self, ideal_features, bad_features):
        assert ideal_features["f_notice_score"] > bad_features["f_notice_score"]

    def test_ideal_has_better_disqualifier_penalty(self, ideal_features, bad_features):
        assert ideal_features["f_disqualifier_penalty"] > bad_features["f_disqualifier_penalty"]

    def test_ideal_overall_mean_higher(self, ideal_features, bad_features):
        ideal_mean = sum(ideal_features.values()) / 27
        bad_mean   = sum(bad_features.values()) / 27
        assert ideal_mean > bad_mean, (
            f"Ideal mean {ideal_mean:.3f} not above bad mean {bad_mean:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# §M  Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_completely_empty_flat_no_crash(self):
        result = extract_features({"candidate_id": "EMPTY"})
        assert isinstance(result, dict)
        assert len(result) == 27
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"{k}={v}"

    def test_none_string_fields_handled(self):
        flat = {
            "candidate_id":        "NULL_001",
            "current_title":       None,
            "current_industry":    None,
            "current_company_size": None,
            "country":             None,
            "location":            None,
            "english_proficiency": None,
            "highest_edu_tier":    None,
            "most_recent_title":   None,
        }
        result = extract_features(flat)
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"{k}={v}"

    def test_negative_skill_count_safe(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["skill_count"] = 0
        result = extract_features(flat)   # should not divide by zero
        assert 0.0 <= result["f_tier1_density"] <= 1.0

    def test_very_long_notice_period_capped(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["notice_period_days"] = 365
        result = extract_features(flat)
        assert result["f_notice_score"] <= 1.0

    def test_very_high_yoe_does_not_exceed_1(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["years_of_experience"] = 50.0
        result = extract_features(flat)
        assert result["f_yoe_normalized"] == pytest.approx(1.0)
        assert result["f_yoe_fit"] <= 1.0

    def test_days_inactive_sentinel_9999_handled(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["days_since_last_active"] = 9999
        result = extract_features(flat)
        assert result["f_activity_score"] <= 0.15

    def test_skill_assessment_scores_none_handled(self):
        flat = deepcopy(_IDEAL_FLAT)
        flat["skill_assessment_scores"] = None
        result = extract_features(flat)
        assert result["f_assessment_score"] == pytest.approx(0.50)

    def test_missing_language_fields_use_defaults(self):
        flat = {k: v for k, v in _IDEAL_FLAT.items()
                if k not in ("language_count", "english_proficiency", "language_names")}
        result = extract_features(flat)
        assert 0.0 <= result["f_english_proficiency_score"] <= 1.0
        assert result["f_language_diversity"] == pytest.approx(0.0)
