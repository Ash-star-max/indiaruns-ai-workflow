"""
tests/test_trap_detection.py — comprehensive tests for trap_detection.py

§ A  Module structure & constants       (12 tests)
§ B  detect_keyword_stuffing            (10 tests)
§ C  detect_fake_ai_profile             (11 tests)
§ D  detect_generic_chatgpt_user        ( 9 tests)
§ E  detect_research_only               (10 tests)
§ F  detect_low_quality_profile         ( 9 tests)
§ G  detect_inactive_candidate          ( 9 tests)
§ H  detect_inconsistent_career         (10 tests)
§ I  detect_suspicious_timeline         (10 tests)
§ J  detect_ai_keywords_no_production   (10 tests)
§ K  detect_behavioral_trust_issues     (10 tests)
§ L  detect_traps (main)                (12 tests)
§ M  build_trap_report                  ( 7 tests)
§ N  Edge cases & sentinels             ( 9 tests)

Total: 138 tests
"""

import pytest
import pandas as pd

from src.trap_detection import (
    # constants
    TRAP_NAMES,
    TRAP_PENALTY_FACTORS,
    MIN_COMPOUND_PENALTY,
    # dataclass
    TrapSignal,
    # individual detectors
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
    # main API
    run_all_detectors,
    detect_traps,
    build_trap_report,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

CLEAN = {
    # A solid, realistic Senior AI Engineer profile
    "candidate_id":              "clean-001",
    "summary":                   (
        "Senior ML Engineer with 7 years building production recommendation and "
        "search systems. Deployed real-time inference endpoints serving 50M daily "
        "active users with sub-50ms latency. Expert in FAISS, Pinecone, and "
        "PyTorch; strong background in A/B testing, model monitoring, and MLflow."
    ),
    "headline":                  "Senior ML Engineer | Production AI Systems",
    "career_descriptions_text":  (
        "Built and deployed vector search pipeline handling 10M QPS. Led migration "
        "from Elasticsearch to Qdrant for semantic ranking. Designed inference APIs "
        "with TorchServe on Kubernetes. Implemented drift detection with Grafana alerts."
    ),
    "candidate_text":            "",
    "skill_names":               ["Python", "PyTorch", "FAISS", "Kubernetes", "MLflow",
                                  "Docker", "Redis", "PostgreSQL", "TensorFlow", "NumPy"],
    "skill_count":               10,
    "tier1_skill_count":         5,
    "tier2_skill_count":         3,
    "expert_zero_duration_count":0,
    "tier1_avg_duration_months": 30.0,
    "tier2_avg_duration_months": 18.0,
    "career_titles":             ["ML Engineer", "Senior ML Engineer"],
    "n_career_roles":            2,
    "total_career_months":       84,
    "years_of_experience":       7,
    "most_recent_title":         "Senior ML Engineer",
    "is_consulting_only":        False,
    "product_company_ratio":     1.0,
    "highest_edu_tier":          "tier_1",
    "edu_fields":                ["computer science"],
    "edu_count":                 1,
    "cert_names":                ["AWS ML Specialty"],
    "cert_count":                1,
    "profile_completeness_score":82.0,
    "connection_count":          350,
    "endorsements_received":     45,
    "github_activity_score":     72.0,
    "salary_inverted":           False,
    "salary_min_lpa":            40.0,
    "salary_max_lpa":            70.0,
    "days_since_last_active":    3,
    "open_to_work_flag":         True,
    "applications_submitted_30d":4,
    "profile_views_received_30d":55,
    "search_appearance_30d":     130,
    "recruiter_response_rate":   0.80,
    "avg_response_time_hours":   4.0,
    "interview_completion_rate": 0.90,
    "offer_acceptance_rate":     0.75,
    "verified_email":            True,
    "verified_phone":            True,
    "linkedin_connected":        True,
    "saved_by_recruiters_30d":   8,
}

EMPTY: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# § A  Module structure & constants
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleStructure:
    def test_trap_names_count(self):
        assert len(TRAP_NAMES) == 10

    def test_all_expected_names_present(self):
        expected = {
            "keyword_stuffing", "fake_ai_profile", "generic_chatgpt_user",
            "research_only", "low_quality_profile", "inactive_candidate",
            "inconsistent_career", "suspicious_timeline",
            "ai_keywords_no_production", "behavioral_trust_issues",
        }
        assert set(TRAP_NAMES) == expected

    def test_penalty_factors_keys_match_names(self):
        assert set(TRAP_PENALTY_FACTORS.keys()) == set(TRAP_NAMES)

    def test_penalty_factors_in_valid_range(self):
        for name, factor in TRAP_PENALTY_FACTORS.items():
            assert 0.0 < factor <= 1.0, f"{name}: {factor}"

    def test_min_compound_penalty_small(self):
        assert 0.0 < MIN_COMPOUND_PENALTY < 0.20

    def test_trap_signal_is_dataclass(self):
        ts = TrapSignal("test", True, 0.5, 0.7, {}, "desc")
        assert ts.name == "test"
        assert ts.triggered is True

    def test_trap_signal_as_dict_keys(self):
        ts = TrapSignal("keyword_stuffing", False, 0.2, 1.0, {}, "ok")
        d  = ts.as_dict()
        assert set(d.keys()) == {"name", "triggered", "confidence",
                                  "penalty_factor", "evidence", "description"}

    def test_run_all_detectors_returns_10(self):
        signals = run_all_detectors(CLEAN)
        assert len(signals) == 10

    def test_run_all_detectors_types(self):
        for s in run_all_detectors(CLEAN):
            assert isinstance(s, TrapSignal)

    def test_run_all_detectors_confidence_in_range(self):
        for s in run_all_detectors(CLEAN):
            assert 0.0 <= s.confidence <= 1.0, f"{s.name}: {s.confidence}"

    def test_detect_traps_top_level_keys(self):
        result = detect_traps(CLEAN)
        assert set(result.keys()) == {"trap_risk_score", "trap_labels",
                                       "trap_penalty", "explanation"}

    def test_detect_traps_explanation_length(self):
        result = detect_traps(CLEAN)
        assert len(result["explanation"]) == 10


# ─────────────────────────────────────────────────────────────────────────────
# § B  detect_keyword_stuffing
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordStuffing:
    def test_clean_profile_not_triggered(self):
        s = detect_keyword_stuffing(CLEAN)
        assert not s.triggered

    def test_massive_skill_list_triggers(self):
        flat = {**CLEAN, "skill_count": 55, "expert_zero_duration_count": 0}
        s = detect_keyword_stuffing(flat)
        assert s.triggered

    def test_many_expert_zero_skills_trigger(self):
        flat = {**CLEAN, "expert_zero_duration_count": 10, "skill_count": 15}
        s = detect_keyword_stuffing(flat)
        assert s.triggered

    def test_high_buzzword_density_triggers(self):
        buzzword_summary = " ".join([
            "machine learning deep learning neural network pytorch tensorflow "
            "bert gpt llm transformer nlp computer vision reinforcement learning "
            "rag retrieval augmented generation faiss pinecone weaviate embedding "
            "vector database fine-tuning mlops data science generative ai"
        ] * 3)
        flat = {**CLEAN, "summary": buzzword_summary, "skill_count": 8}
        s = detect_keyword_stuffing(flat)
        assert s.triggered

    def test_returns_trap_signal(self):
        s = detect_keyword_stuffing(CLEAN)
        assert isinstance(s, TrapSignal)
        assert s.name == "keyword_stuffing"

    def test_evidence_has_skill_count(self):
        flat = {**CLEAN, "skill_count": 40}
        s = detect_keyword_stuffing(flat)
        assert "skill_count" in s.evidence

    def test_evidence_has_density(self):
        s = detect_keyword_stuffing(CLEAN)
        assert "buzzword_density" in s.evidence

    def test_empty_flat_no_crash(self):
        s = detect_keyword_stuffing(EMPTY)
        assert isinstance(s, TrapSignal)

    def test_confidence_monotone_with_skills(self):
        c10 = detect_keyword_stuffing({**CLEAN, "skill_count": 10}).confidence
        c30 = detect_keyword_stuffing({**CLEAN, "skill_count": 30}).confidence
        c50 = detect_keyword_stuffing({**CLEAN, "skill_count": 50}).confidence
        assert c10 <= c30 <= c50

    def test_penalty_only_applied_when_triggered(self):
        s = detect_keyword_stuffing(CLEAN)
        if not s.triggered:
            assert s.penalty_factor == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# § C  detect_fake_ai_profile
# ─────────────────────────────────────────────────────────────────────────────

class TestFakeAiProfile:
    def test_clean_profile_not_triggered(self):
        s = detect_fake_ai_profile(CLEAN)
        assert not s.triggered

    def test_salary_inverted_triggers(self):
        flat = {**CLEAN, "salary_inverted": True}
        s = detect_fake_ai_profile(flat)
        assert s.triggered

    def test_salary_inverted_high_confidence(self):
        flat = {**CLEAN, "salary_inverted": True}
        s = detect_fake_ai_profile(flat)
        assert s.confidence > 0.40

    def test_many_expert_zero_skills_triggers(self):
        flat = {**CLEAN, "expert_zero_duration_count": 12,
                "profile_completeness_score": 99.0, "connection_count": 500}
        s = detect_fake_ai_profile(flat)
        assert s.triggered

    def test_perfect_completeness_alone_insufficient(self):
        flat = {**CLEAN, "profile_completeness_score": 100.0}
        s = detect_fake_ai_profile(flat)
        # Perfect completeness alone is not enough to trigger
        # (real candidates can fill every field)
        assert s.confidence < 0.60 or not s.triggered

    def test_accumulation_of_flags_triggers(self):
        flat = {
            **CLEAN,
            "profile_completeness_score": 100.0,
            "expert_zero_duration_count": 9,
            "connection_count":           500,
            "github_activity_score":      -1.0,
            "years_of_experience":        10,
        }
        s = detect_fake_ai_profile(flat)
        assert s.triggered

    def test_evidence_dict_populated(self):
        flat = {**CLEAN, "salary_inverted": True}
        s = detect_fake_ai_profile(flat)
        assert len(s.evidence) > 0

    def test_returns_correct_name(self):
        s = detect_fake_ai_profile(CLEAN)
        assert s.name == "fake_ai_profile"

    def test_penalty_factor_when_triggered(self):
        flat = {**CLEAN, "salary_inverted": True}
        s = detect_fake_ai_profile(flat)
        assert s.triggered
        assert s.penalty_factor == TRAP_PENALTY_FACTORS["fake_ai_profile"]

    def test_penalty_factor_1_when_not_triggered(self):
        s = detect_fake_ai_profile(CLEAN)
        if not s.triggered:
            assert s.penalty_factor == 1.0

    def test_high_completeness_no_github_is_red_flag(self):
        flat = {**CLEAN, "profile_completeness_score": 95.0,
                "github_activity_score": -1.0, "connection_count": 500,
                "expert_zero_duration_count": 8, "salary_inverted": False}
        s = detect_fake_ai_profile(flat)
        assert s.confidence > 0.30


# ─────────────────────────────────────────────────────────────────────────────
# § D  detect_generic_chatgpt_user
# ─────────────────────────────────────────────────────────────────────────────

class TestGenericChatgptUser:
    def test_clean_profile_not_triggered(self):
        s = detect_generic_chatgpt_user(CLEAN)
        assert not s.triggered

    def test_many_phrases_trigger(self):
        template = (
            "I am a results-driven, passionate about machine learning professional "
            "with a proven track record of leveraging cutting-edge innovative solutions "
            "in cross-functional teams. As a team player, I am self-motivated "
            "and dedicated professional seeking new opportunities in AI."
        )
        flat = {**CLEAN, "summary": template}
        s = detect_generic_chatgpt_user(flat)
        assert s.triggered

    def test_no_summary_not_triggered(self):
        flat = {**CLEAN, "summary": ""}
        s = detect_generic_chatgpt_user(flat)
        assert not s.triggered

    def test_phrase_hits_in_evidence(self):
        template = "I am passionate about AI and self-motivated team player with best practices."
        flat = {**CLEAN, "summary": template}
        s = detect_generic_chatgpt_user(flat)
        assert "phrase_hits" in s.evidence

    def test_single_phrase_low_confidence(self):
        flat = {**CLEAN, "summary": "I am passionate about AI engineering."}
        s = detect_generic_chatgpt_user(flat)
        assert s.confidence < 0.40

    def test_returns_correct_name(self):
        s = detect_generic_chatgpt_user(CLEAN)
        assert s.name == "generic_chatgpt_user"

    def test_empty_flat_no_crash(self):
        s = detect_generic_chatgpt_user(EMPTY)
        assert isinstance(s, TrapSignal)

    def test_technical_summary_not_triggered(self):
        technical = (
            "Deployed FAISS-based ANN index serving 100M candidates with p99 "
            "latency under 8ms. Fine-tuned Sentence-BERT on domain corpus of "
            "2M job postings. Implemented hybrid BM25+dense reranker."
        )
        flat = {**CLEAN, "summary": technical}
        s = detect_generic_chatgpt_user(flat)
        assert not s.triggered

    def test_confidence_increases_with_phrase_count(self):
        one   = "I am passionate about machine learning."
        three = (
            "I am passionate about AI. I am results-driven professional "
            "with a proven track record."
        )
        s1 = detect_generic_chatgpt_user({**CLEAN, "summary": one})
        s3 = detect_generic_chatgpt_user({**CLEAN, "summary": three})
        assert s3.confidence >= s1.confidence


# ─────────────────────────────────────────────────────────────────────────────
# § E  detect_research_only
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchOnly:
    def test_clean_profile_not_triggered(self):
        s = detect_research_only(CLEAN)
        assert not s.triggered

    def test_all_research_titles_triggers(self):
        flat = {
            **CLEAN,
            "career_titles": ["PhD Student", "Research Intern", "Research Scientist",
                              "Post-doc Researcher"],
            "n_career_roles":         4,
            "product_company_ratio":  0.0,
            "tier1_skill_count":      0,
        }
        s = detect_research_only(flat)
        assert s.triggered

    def test_no_tier1_skills_boosts_confidence(self):
        flat = {**CLEAN, "tier1_skill_count": 0, "product_company_ratio": 0.0,
                "career_titles": ["Research Engineer"], "n_career_roles": 1}
        s_no_tier1 = detect_research_only(flat)
        s_with_tier1 = detect_research_only({**flat, "tier1_skill_count": 4})
        assert s_no_tier1.confidence >= s_with_tier1.confidence

    def test_academic_career_descriptions_boost(self):
        flat = {
            **CLEAN,
            "career_descriptions_text": (
                "Published 5 papers at NeurIPS and ICML. "
                "Completed PhD dissertation on neural information retrieval. "
                "Cited 200+ times. Post-doc fellowship at top university."
            ),
            "tier1_skill_count":      0,
            "product_company_ratio":  0.0,
        }
        s = detect_research_only(flat)
        assert s.confidence > 0.40

    def test_high_product_ratio_suppresses(self):
        flat = {
            **CLEAN,
            "product_company_ratio":  1.0,
            "tier1_skill_count":      4,
            "career_titles": ["Software Engineer", "ML Engineer"],
        }
        s = detect_research_only(flat)
        assert not s.triggered

    def test_evidence_has_research_title_ratio(self):
        flat = {**CLEAN, "career_titles": ["Research Scientist"], "n_career_roles": 1}
        s = detect_research_only(flat)
        assert "research_title_ratio" in s.evidence

    def test_returns_correct_name(self):
        s = detect_research_only(CLEAN)
        assert s.name == "research_only"

    def test_empty_flat_no_crash(self):
        s = detect_research_only(EMPTY)
        assert isinstance(s, TrapSignal)

    def test_product_engineer_not_triggered(self):
        flat = {
            **CLEAN,
            "career_titles": ["Software Engineer", "ML Engineer", "Senior ML Engineer"],
            "product_company_ratio": 1.0,
            "tier1_skill_count":     4,
        }
        s = detect_research_only(flat)
        assert not s.triggered

    def test_penalty_factor_correct_on_trigger(self):
        flat = {
            **CLEAN,
            "career_titles": ["PhD Researcher", "Research Intern", "Post-doc"],
            "n_career_roles": 3,
            "product_company_ratio": 0.0,
            "tier1_skill_count": 0,
        }
        s = detect_research_only(flat)
        if s.triggered:
            assert s.penalty_factor == TRAP_PENALTY_FACTORS["research_only"]


# ─────────────────────────────────────────────────────────────────────────────
# § F  detect_low_quality_profile
# ─────────────────────────────────────────────────────────────────────────────

class TestLowQualityProfile:
    def test_clean_profile_not_triggered(self):
        s = detect_low_quality_profile(CLEAN)
        assert not s.triggered

    def test_very_low_completeness_triggers(self):
        flat = {**CLEAN, "profile_completeness_score": 15.0,
                "skill_count": 1, "tier1_skill_count": 0, "tier2_skill_count": 0}
        s = detect_low_quality_profile(flat)
        assert s.triggered

    def test_no_skills_no_career_triggers(self):
        flat = {
            **CLEAN,
            "skill_count": 1, "tier1_skill_count": 0, "tier2_skill_count": 0,
            "cert_count": 0,  "n_career_roles": 0,    "edu_count": 0,
            "profile_completeness_score": 10.0,
        }
        s = detect_low_quality_profile(flat)
        assert s.triggered

    def test_short_summary_adds_to_score(self):
        flat = {**CLEAN, "summary": "I love AI.", "profile_completeness_score": 35.0}
        s = detect_low_quality_profile(flat)
        assert s.confidence > 0.20

    def test_returns_correct_name(self):
        s = detect_low_quality_profile(CLEAN)
        assert s.name == "low_quality_profile"

    def test_empty_flat_no_crash(self):
        s = detect_low_quality_profile(EMPTY)
        assert isinstance(s, TrapSignal)

    def test_evidence_keys_populated_on_trigger(self):
        flat = {**CLEAN, "profile_completeness_score": 10.0,
                "skill_count": 1, "tier1_skill_count": 0, "tier2_skill_count": 0,
                "n_career_roles": 0, "cert_count": 0}
        s = detect_low_quality_profile(flat)
        assert len(s.evidence) > 0

    def test_good_profile_has_low_confidence(self):
        s = detect_low_quality_profile(CLEAN)
        assert s.confidence < 0.40

    def test_penalty_correct_on_trigger(self):
        flat = {**CLEAN, "profile_completeness_score": 10.0,
                "skill_count": 0, "tier1_skill_count": 0, "tier2_skill_count": 0}
        s = detect_low_quality_profile(flat)
        if s.triggered:
            assert s.penalty_factor == TRAP_PENALTY_FACTORS["low_quality_profile"]


# ─────────────────────────────────────────────────────────────────────────────
# § G  detect_inactive_candidate
# ─────────────────────────────────────────────────────────────────────────────

class TestInactiveCandidate:
    def test_clean_profile_not_triggered(self):
        s = detect_inactive_candidate(CLEAN)
        assert not s.triggered

    def test_very_stale_not_seeking_triggers(self):
        flat = {
            **CLEAN,
            "days_since_last_active":      400,
            "open_to_work_flag":           False,
            "applications_submitted_30d":  0,
            "profile_views_received_30d":  1,
            "search_appearance_30d":       0,
        }
        s = detect_inactive_candidate(flat)
        assert s.triggered

    def test_active_but_not_seeking_not_triggered(self):
        flat = {
            **CLEAN,
            "days_since_last_active":      5,
            "open_to_work_flag":           False,
            "applications_submitted_30d":  0,
            "profile_views_received_30d":  30,
        }
        s = detect_inactive_candidate(flat)
        assert not s.triggered

    def test_staleness_increases_confidence(self):
        c30  = detect_inactive_candidate({**CLEAN, "days_since_last_active": 30,  "open_to_work_flag": False}).confidence
        c180 = detect_inactive_candidate({**CLEAN, "days_since_last_active": 180, "open_to_work_flag": False}).confidence
        c360 = detect_inactive_candidate({**CLEAN, "days_since_last_active": 360, "open_to_work_flag": False}).confidence
        assert c30 <= c180 <= c360

    def test_evidence_has_days(self):
        flat = {**CLEAN, "days_since_last_active": 200}
        s = detect_inactive_candidate(flat)
        assert "days_since_last_active" in s.evidence

    def test_returns_correct_name(self):
        s = detect_inactive_candidate(CLEAN)
        assert s.name == "inactive_candidate"

    def test_empty_flat_no_crash(self):
        s = detect_inactive_candidate(EMPTY)
        assert isinstance(s, TrapSignal)

    def test_threshold_90_days_sensitive(self):
        c60  = detect_inactive_candidate({**CLEAN, "days_since_last_active": 60,  "open_to_work_flag": False}).confidence
        c200 = detect_inactive_candidate({**CLEAN, "days_since_last_active": 200, "open_to_work_flag": False}).confidence
        assert c200 > c60

    def test_all_activity_zero_raises_score(self):
        flat = {
            **CLEAN,
            "days_since_last_active":      250,
            "open_to_work_flag":           False,
            "applications_submitted_30d":  0,
            "profile_views_received_30d":  0,
            "search_appearance_30d":       0,
        }
        s = detect_inactive_candidate(flat)
        assert s.confidence > 0.50


# ─────────────────────────────────────────────────────────────────────────────
# § H  detect_inconsistent_career
# ─────────────────────────────────────────────────────────────────────────────

class TestInconsistentCareer:
    def test_clean_profile_not_triggered(self):
        s = detect_inconsistent_career(CLEAN)
        assert not s.triggered

    def test_seniority_regression_raises_confidence(self):
        flat = {
            **CLEAN,
            "career_titles":  ["Director of AI", "Junior ML Engineer", "Senior Engineer"],
            "n_career_roles": 3,
            "years_of_experience": 7,
        }
        s = detect_inconsistent_career(flat)
        assert s.confidence > 0.10

    def test_extreme_job_hopping_triggers(self):
        flat = {
            **CLEAN,
            "career_titles":      ["Eng"] * 10,
            "n_career_roles":     10,
            "years_of_experience": 2,
        }
        s = detect_inconsistent_career(flat)
        assert s.triggered

    def test_consulting_with_ml_claims_raises(self):
        flat = {**CLEAN, "is_consulting_only": True, "tier1_skill_count": 4}
        s = detect_inconsistent_career(flat)
        assert s.confidence > 0.15

    def test_yoe_no_history_raises(self):
        flat = {**CLEAN, "n_career_roles": 0, "years_of_experience": 8}
        s = detect_inconsistent_career(flat)
        assert s.confidence > 0.10

    def test_evidence_jobs_per_year(self):
        flat = {**CLEAN, "n_career_roles": 12, "years_of_experience": 3}
        s = detect_inconsistent_career(flat)
        assert "jobs_per_year" in s.evidence

    def test_returns_correct_name(self):
        s = detect_inconsistent_career(CLEAN)
        assert s.name == "inconsistent_career"

    def test_empty_flat_no_crash(self):
        s = detect_inconsistent_career(EMPTY)
        assert isinstance(s, TrapSignal)

    def test_normal_progression_not_triggered(self):
        flat = {
            **CLEAN,
            "career_titles":  ["ML Engineer", "Senior ML Engineer", "Lead ML Engineer"],
            "n_career_roles": 3,
            "years_of_experience": 8,
        }
        s = detect_inconsistent_career(flat)
        assert not s.triggered

    def test_penalty_only_when_triggered(self):
        s = detect_inconsistent_career(CLEAN)
        if not s.triggered:
            assert s.penalty_factor == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# § I  detect_suspicious_timeline
# ─────────────────────────────────────────────────────────────────────────────

class TestSuspiciousTimeline:
    def test_clean_profile_not_triggered(self):
        s = detect_suspicious_timeline(CLEAN)
        assert not s.triggered

    def test_overflow_career_months_triggers(self):
        flat = {
            **CLEAN,
            "total_career_months":  200,   # 16.7 years
            "years_of_experience":  7,     # claimed 7 years → 84 months
        }
        s = detect_suspicious_timeline(flat)
        assert s.triggered

    def test_yoe_no_roles_triggers(self):
        flat = {**CLEAN, "years_of_experience": 8, "n_career_roles": 0,
                "total_career_months": 0}
        s = detect_suspicious_timeline(flat)
        assert s.triggered

    def test_salary_inverted_raises_confidence(self):
        flat = {**CLEAN, "salary_inverted": True}
        s = detect_suspicious_timeline(flat)
        assert s.confidence > 0.10

    def test_extreme_job_hopping_triggers(self):
        flat = {
            **CLEAN,
            "n_career_roles":      15,
            "years_of_experience":  2,
            "total_career_months": 24,
        }
        s = detect_suspicious_timeline(flat)
        assert s.triggered

    def test_evidence_overflow_populated(self):
        flat = {**CLEAN, "total_career_months": 300, "years_of_experience": 7}
        s = detect_suspicious_timeline(flat)
        if s.triggered:
            assert "overflow_ratio" in s.evidence or "total_career_months" in s.evidence

    def test_returns_correct_name(self):
        s = detect_suspicious_timeline(CLEAN)
        assert s.name == "suspicious_timeline"

    def test_empty_flat_no_crash(self):
        s = detect_suspicious_timeline(EMPTY)
        assert isinstance(s, TrapSignal)

    def test_clean_timeline_low_confidence(self):
        s = detect_suspicious_timeline(CLEAN)
        assert s.confidence < 0.40

    def test_confidence_monotone_with_overflow(self):
        c1 = detect_suspicious_timeline({**CLEAN, "total_career_months": 90,  "years_of_experience": 7}).confidence
        c2 = detect_suspicious_timeline({**CLEAN, "total_career_months": 150, "years_of_experience": 7}).confidence
        c3 = detect_suspicious_timeline({**CLEAN, "total_career_months": 250, "years_of_experience": 7}).confidence
        assert c1 <= c2 <= c3


# ─────────────────────────────────────────────────────────────────────────────
# § J  detect_ai_keywords_no_production
# ─────────────────────────────────────────────────────────────────────────────

class TestAiKeywordsNoProduction:
    def test_clean_profile_not_triggered(self):
        s = detect_ai_keywords_no_production(CLEAN)
        assert not s.triggered

    def test_buzzwords_only_triggers(self):
        flat = {
            **CLEAN,
            "summary": (
                "machine learning deep learning neural network pytorch tensorflow "
                "bert gpt llm transformer nlp computer vision rag retrieval "
                "augmented generation embedding vector database fine-tuning"
            ),
            "career_descriptions_text": (
                "machine learning deep learning transformer bert gpt pytorch "
                "tensorflow llm rag vector database embedding faiss pinecone "
                "semantic search natural language processing computer vision"
            ),
            "years_of_experience": 5,
        }
        s = detect_ai_keywords_no_production(flat)
        assert s.triggered

    def test_production_keywords_suppress(self):
        flat = {
            **CLEAN,
            "summary": (
                "Deployed production inference API serving 1M users/day. "
                "Kubernetes cluster with docker containers. MLflow model monitoring. "
                "A/B testing framework for ranking. Latency p99 < 50ms."
            ),
        }
        s = detect_ai_keywords_no_production(flat)
        assert not s.triggered

    def test_junior_leniency(self):
        junior = {
            **CLEAN,
            "summary": (
                "machine learning pytorch tensorflow bert gpt llm transformer "
                "rag retrieval vector database embedding fine-tuning"
            ),
            "career_descriptions_text": "studied machine learning at university",
            "years_of_experience": 1,
        }
        s = detect_ai_keywords_no_production(junior)
        # Junior should have reduced confidence due to leniency
        assert not s.triggered or s.confidence < 0.60

    def test_senior_no_leniency(self):
        senior = {
            **CLEAN,
            "summary": (
                "machine learning pytorch tensorflow bert gpt llm transformer rag "
                "retrieval vector database embedding fine-tuning neural network"
            ),
            "career_descriptions_text": (
                "machine learning pytorch tensorflow bert gpt llm transformer rag"
            ),
            "years_of_experience": 8,
        }
        s = detect_ai_keywords_no_production(senior)
        # Senior with no production keywords → should trigger
        assert s.triggered

    def test_evidence_has_hit_counts(self):
        s = detect_ai_keywords_no_production(CLEAN)
        assert "ai_keyword_hits" in s.evidence
        assert "production_keyword_hits" in s.evidence

    def test_returns_correct_name(self):
        s = detect_ai_keywords_no_production(CLEAN)
        assert s.name == "ai_keywords_no_production"

    def test_empty_flat_no_crash(self):
        s = detect_ai_keywords_no_production(EMPTY)
        assert isinstance(s, TrapSignal)

    def test_no_text_low_confidence(self):
        flat = {**CLEAN, "summary": "", "career_descriptions_text": ""}
        s = detect_ai_keywords_no_production(flat)
        assert not s.triggered

    def test_leniency_flag_in_evidence(self):
        junior = {**CLEAN, "years_of_experience": 1}
        s = detect_ai_keywords_no_production(junior)
        assert "leniency_applied" in s.evidence
        assert s.evidence["leniency_applied"] is True


# ─────────────────────────────────────────────────────────────────────────────
# § K  detect_behavioral_trust_issues
# ─────────────────────────────────────────────────────────────────────────────

class TestBehavioralTrustIssues:
    def test_clean_profile_not_triggered(self):
        s = detect_behavioral_trust_issues(CLEAN)
        assert not s.triggered

    def test_ghost_recruiter_triggers(self):
        flat = {
            **CLEAN,
            "recruiter_response_rate":   0.01,
            "interview_completion_rate": 0.10,
            "offer_acceptance_rate":     0.00,
            "github_activity_score":     -1.0,
            "linkedin_connected":        False,
        }
        s = detect_behavioral_trust_issues(flat)
        assert s.triggered

    def test_low_response_rate_raises_confidence(self):
        flat = {**CLEAN, "recruiter_response_rate": 0.01}
        s = detect_behavioral_trust_issues(flat)
        assert s.confidence > 0.15

    def test_no_online_presence_raises_confidence(self):
        flat = {**CLEAN, "github_activity_score": -1.0, "linkedin_connected": False}
        s = detect_behavioral_trust_issues(flat)
        assert s.confidence > 0.10

    def test_salary_inverted_is_trust_signal(self):
        flat = {**CLEAN, "salary_inverted": True}
        s = detect_behavioral_trust_issues(flat)
        assert s.confidence > 0.05

    def test_low_interview_completion_raises(self):
        flat = {**CLEAN, "interview_completion_rate": 0.10}
        s = detect_behavioral_trust_issues(flat)
        assert s.confidence > 0.15

    def test_good_behavioral_profile_low_confidence(self):
        s = detect_behavioral_trust_issues(CLEAN)
        assert s.confidence < 0.30

    def test_returns_correct_name(self):
        s = detect_behavioral_trust_issues(CLEAN)
        assert s.name == "behavioral_trust_issues"

    def test_empty_flat_no_crash(self):
        s = detect_behavioral_trust_issues(EMPTY)
        assert isinstance(s, TrapSignal)

    def test_penalty_correct_on_trigger(self):
        flat = {
            **CLEAN,
            "recruiter_response_rate":   0.01,
            "interview_completion_rate": 0.05,
            "github_activity_score":     -1.0,
            "linkedin_connected":        False,
            "offer_acceptance_rate":     0.0,
        }
        s = detect_behavioral_trust_issues(flat)
        if s.triggered:
            assert s.penalty_factor == TRAP_PENALTY_FACTORS["behavioral_trust_issues"]


# ─────────────────────────────────────────────────────────────────────────────
# § L  detect_traps (main)
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectTraps:
    def test_returns_dict_with_required_keys(self):
        result = detect_traps(CLEAN)
        assert "trap_risk_score" in result
        assert "trap_labels"     in result
        assert "trap_penalty"    in result
        assert "explanation"     in result

    def test_clean_profile_no_labels(self):
        result = detect_traps(CLEAN)
        assert result["trap_labels"] == []

    def test_clean_profile_no_penalty(self):
        result = detect_traps(CLEAN)
        assert result["trap_penalty"] == 1.0

    def test_clean_profile_zero_risk(self):
        result = detect_traps(CLEAN)
        assert result["trap_risk_score"] == 0.0

    def test_fabricated_profile_high_penalty(self):
        flat = {
            **CLEAN,
            "salary_inverted":             True,
            "expert_zero_duration_count":  12,
            "profile_completeness_score":  100.0,
            "connection_count":            500,
            "github_activity_score":       -1.0,
        }
        result = detect_traps(flat)
        assert result["trap_penalty"] < 0.80

    def test_multi_trap_penalty_compounds(self):
        flat = {
            **CLEAN,
            "salary_inverted":              True,
            "days_since_last_active":       400,
            "open_to_work_flag":            False,
            "applications_submitted_30d":   0,
            "profile_views_received_30d":   0,
            "recruiter_response_rate":      0.01,
        }
        result = detect_traps(flat)
        assert len(result["trap_labels"]) >= 2
        # Compound penalty must be less than minimum individual penalty
        min_penalty = min(
            TRAP_PENALTY_FACTORS[name] for name in result["trap_labels"]
        )
        assert result["trap_penalty"] <= min_penalty + 0.01

    def test_compound_penalty_never_below_floor(self):
        # Worst possible profile
        flat = {
            "salary_inverted":             True,
            "days_since_last_active":      9999,
            "open_to_work_flag":           False,
            "recruiter_response_rate":     0.0,
            "interview_completion_rate":   0.0,
            "profile_completeness_score":  5.0,
            "skill_count":                 0,
            "n_career_roles":              0,
            "years_of_experience":         10,
        }
        result = detect_traps(flat)
        assert result["trap_penalty"] >= MIN_COMPOUND_PENALTY

    def test_risk_score_in_range(self):
        result = detect_traps(CLEAN)
        assert 0.0 <= result["trap_risk_score"] <= 1.0

    def test_explanation_has_10_entries(self):
        result = detect_traps(CLEAN)
        assert len(result["explanation"]) == 10

    def test_explanation_triggered_matches_labels(self):
        flat = {**CLEAN, "salary_inverted": True}
        result = detect_traps(flat)
        triggered_in_explain = {
            e["name"] for e in result["explanation"] if e["triggered"]
        }
        assert set(result["trap_labels"]) == triggered_in_explain

    def test_empty_flat_no_crash(self):
        result = detect_traps(EMPTY)
        assert isinstance(result["trap_risk_score"], float)
        assert result["trap_penalty"] >= MIN_COMPOUND_PENALTY

    def test_trap_penalty_rounded(self):
        result = detect_traps(CLEAN)
        assert result["trap_penalty"] == round(result["trap_penalty"], 4)


# ─────────────────────────────────────────────────────────────────────────────
# § M  build_trap_report
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildTrapReport:
    def test_returns_dataframe(self):
        df = build_trap_report([CLEAN])
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_candidate(self):
        df = build_trap_report([CLEAN, EMPTY, CLEAN])
        assert len(df) == 3

    def test_expected_columns(self):
        df = build_trap_report([CLEAN])
        assert "trap_risk_score" in df.columns
        assert "trap_penalty"    in df.columns
        assert "trap_count"      in df.columns
        for name in TRAP_NAMES:
            assert f"{name}_triggered" in df.columns

    def test_triggered_columns_are_bool(self):
        df = build_trap_report([CLEAN, EMPTY])
        for name in TRAP_NAMES:
            col = f"{name}_triggered"
            assert df[col].dtype == bool, f"{col} is not bool"

    def test_trap_count_matches_labels(self):
        flat = {**CLEAN, "salary_inverted": True, "days_since_last_active": 400,
                "open_to_work_flag": False, "applications_submitted_30d": 0,
                "profile_views_received_30d": 0}
        df = build_trap_report([flat])
        expected_count = detect_traps(flat)
        assert df["trap_count"].iloc[0] == len(expected_count["trap_labels"])

    def test_indexed_by_candidate_id(self):
        df = build_trap_report([CLEAN])
        assert df.index.name == "candidate_id"

    def test_batch_multiple_candidates(self):
        flats = [CLEAN] * 20 + [EMPTY] * 5
        df = build_trap_report(flats)
        assert len(df) == 25


# ─────────────────────────────────────────────────────────────────────────────
# § N  Edge cases & sentinels
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_none_values_no_crash(self):
        flat = {k: None for k in CLEAN}
        result = detect_traps(flat)
        assert isinstance(result["trap_risk_score"], float)

    def test_negative_years_no_crash(self):
        flat = {**CLEAN, "years_of_experience": -5}
        result = detect_traps(flat)
        assert 0.0 <= result["trap_risk_score"] <= 1.0

    def test_zero_years_no_crash(self):
        flat = {**CLEAN, "years_of_experience": 0}
        result = detect_traps(flat)
        assert isinstance(result, dict)

    def test_empty_career_titles_list(self):
        flat = {**CLEAN, "career_titles": [], "n_career_roles": 0}
        result = detect_traps(flat)
        assert isinstance(result, dict)

    def test_very_large_skill_count(self):
        flat = {**CLEAN, "skill_count": 1000}
        s = detect_keyword_stuffing(flat)
        assert s.confidence <= 1.0

    def test_very_large_career_months(self):
        flat = {**CLEAN, "total_career_months": 10000, "years_of_experience": 7}
        s = detect_suspicious_timeline(flat)
        assert s.confidence <= 1.0

    def test_description_is_always_string(self):
        for fn in [detect_keyword_stuffing, detect_fake_ai_profile,
                   detect_generic_chatgpt_user, detect_research_only,
                   detect_low_quality_profile, detect_inactive_candidate,
                   detect_inconsistent_career, detect_suspicious_timeline,
                   detect_ai_keywords_no_production, detect_behavioral_trust_issues]:
            s = fn(EMPTY)
            assert isinstance(s.description, str)

    def test_confidence_always_clipped(self):
        for fn in [detect_keyword_stuffing, detect_fake_ai_profile,
                   detect_generic_chatgpt_user, detect_research_only]:
            s = fn({**CLEAN, "skill_count": 9999, "expert_zero_duration_count": 100,
                    "salary_inverted": True})
            assert 0.0 <= s.confidence <= 1.0

    def test_risk_score_clean_is_zero(self):
        result = detect_traps(CLEAN)
        assert result["trap_risk_score"] == 0.0
