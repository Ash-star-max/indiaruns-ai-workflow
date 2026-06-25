"""
Tests for src/jd_understanding.py

§A  Structure tests  — every pattern dict has required keys
§B  Content tests    — specific JD requirements appear in correct dicts
§C  Embedding text tests
§D  score_skills_match tests
§E  detect_disqualifiers tests
§F  score_location_match tests
§G  score_experience_fit tests
§H  score_notice_period tests
§I  compute_disqualifier_penalty tests
§J  JD_REQUIREMENTS master dict tests
"""

from __future__ import annotations

import pytest

from src.jd_understanding import (
    MUST_HAVE_SKILLS,
    NICE_TO_HAVE_SKILLS,
    VECTOR_DB_PATTERNS,
    RETRIEVAL_PATTERNS,
    RANKING_PATTERNS,
    EVALUATION_PATTERNS,
    PRODUCTION_ML_PATTERNS,
    SHIPPER_PATTERNS,
    STARTUP_PATTERNS,
    LOCATION_REQUIREMENTS,
    EXPERIENCE_REQUIREMENTS,
    SALARY_EXPECTATIONS,
    DISQUALIFIER_PATTERNS,
    JD_REQUIREMENTS,
    JD_METADATA,
    get_jd_embedding_text,
    score_skills_match,
    detect_disqualifiers,
    score_location_match,
    score_experience_fit,
    score_notice_period,
    compute_disqualifier_penalty,
)

# ─────────────────────────────────────────────────────────────────────────────
# §A  Structure tests
# ─────────────────────────────────────────────────────────────────────────────

MUST_HAVE_REQUIRED_KEYS = {"keywords", "phrases", "threshold", "weight", "required", "jd_quote"}
NICE_REQUIRED_KEYS      = {"keywords", "phrases", "threshold", "weight", "required", "jd_quote"}
DOMAIN_REQUIRED_KEYS    = {"keywords", "phrases", "weight", "required", "description"}
DISQ_REQUIRED_KEYS      = {"penalty_factor", "detectable", "detection_fields", "jd_quote"}


class TestStructure:

    def test_must_have_skills_has_four_groups(self):
        assert len(MUST_HAVE_SKILLS) == 4

    @pytest.mark.parametrize("group", list(MUST_HAVE_SKILLS))
    def test_must_have_group_required_keys(self, group):
        missing = MUST_HAVE_REQUIRED_KEYS - set(MUST_HAVE_SKILLS[group])
        assert not missing, f"Group '{group}' missing keys: {missing}"

    @pytest.mark.parametrize("group", list(MUST_HAVE_SKILLS))
    def test_must_have_all_required_true(self, group):
        assert MUST_HAVE_SKILLS[group]["required"] is True

    @pytest.mark.parametrize("group", list(MUST_HAVE_SKILLS))
    def test_must_have_keywords_nonempty(self, group):
        assert len(MUST_HAVE_SKILLS[group]["keywords"]) > 0

    def test_nice_to_have_skills_has_five_groups(self):
        assert len(NICE_TO_HAVE_SKILLS) == 5

    @pytest.mark.parametrize("group", list(NICE_TO_HAVE_SKILLS))
    def test_nice_group_required_keys(self, group):
        missing = NICE_REQUIRED_KEYS - set(NICE_TO_HAVE_SKILLS[group])
        assert not missing, f"Nice-to-have group '{group}' missing keys: {missing}"

    @pytest.mark.parametrize("group", list(NICE_TO_HAVE_SKILLS))
    def test_nice_all_required_false(self, group):
        assert NICE_TO_HAVE_SKILLS[group]["required"] is False

    @pytest.mark.parametrize("pattern", [
        VECTOR_DB_PATTERNS, RETRIEVAL_PATTERNS, RANKING_PATTERNS,
        EVALUATION_PATTERNS, PRODUCTION_ML_PATTERNS,
    ])
    def test_domain_pattern_required_keys(self, pattern):
        missing = DOMAIN_REQUIRED_KEYS - set(pattern)
        assert not missing, f"Domain pattern missing keys: {missing}"

    def test_disqualifier_patterns_count(self):
        assert len(DISQUALIFIER_PATTERNS) >= 8

    @pytest.mark.parametrize("name", list(DISQUALIFIER_PATTERNS))
    def test_disqualifier_required_keys(self, name):
        missing = DISQ_REQUIRED_KEYS - set(DISQUALIFIER_PATTERNS[name])
        assert not missing, f"Disqualifier '{name}' missing keys: {missing}"

    @pytest.mark.parametrize("name", list(DISQUALIFIER_PATTERNS))
    def test_disqualifier_penalty_in_range(self, name):
        pf = DISQUALIFIER_PATTERNS[name]["penalty_factor"]
        assert 0.0 < pf <= 1.0, f"Disqualifier '{name}' penalty {pf} out of (0,1]"

    @pytest.mark.parametrize("group", list(MUST_HAVE_SKILLS))
    def test_must_have_weight_positive(self, group):
        assert MUST_HAVE_SKILLS[group]["weight"] > 0

    @pytest.mark.parametrize("group", list(NICE_TO_HAVE_SKILLS))
    def test_nice_weight_positive(self, group):
        assert NICE_TO_HAVE_SKILLS[group]["weight"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# §B  Content tests  — specific requirements from the JD
# ─────────────────────────────────────────────────────────────────────────────

class TestContent:

    # Must-have: embeddings retrieval
    def test_sbert_in_embeddings_retrieval(self):
        kws = MUST_HAVE_SKILLS["embeddings_retrieval"]["keywords"]
        assert any("sbert" in k or "sentence-transformers" in k for k in kws)

    def test_bge_and_e5_in_embeddings_retrieval(self):
        kws = MUST_HAVE_SKILLS["embeddings_retrieval"]["keywords"]
        assert "bge" in kws
        assert "e5" in kws

    def test_embedding_drift_in_embeddings_retrieval(self):
        all_text = " ".join(MUST_HAVE_SKILLS["embeddings_retrieval"]["keywords"] +
                             MUST_HAVE_SKILLS["embeddings_retrieval"]["phrases"])
        assert "embedding drift" in all_text

    # Must-have: vector databases
    def test_pinecone_in_vector_databases(self):
        kws = MUST_HAVE_SKILLS["vector_databases"]["keywords"]
        assert "pinecone" in kws

    def test_faiss_in_vector_databases(self):
        kws = MUST_HAVE_SKILLS["vector_databases"]["keywords"]
        assert "faiss" in kws

    def test_qdrant_in_vector_databases(self):
        kws = MUST_HAVE_SKILLS["vector_databases"]["keywords"]
        assert "qdrant" in kws

    # Must-have: python
    def test_python_in_python_engineering(self):
        kws = MUST_HAVE_SKILLS["python_engineering"]["keywords"]
        assert "python" in kws

    # Must-have: ranking evaluation
    def test_ndcg_in_ranking_evaluation(self):
        kws = MUST_HAVE_SKILLS["ranking_evaluation"]["keywords"]
        assert "ndcg" in kws

    def test_mrr_in_ranking_evaluation(self):
        kws = MUST_HAVE_SKILLS["ranking_evaluation"]["keywords"]
        assert "mrr" in kws

    def test_ab_testing_in_ranking_evaluation(self):
        all_text = " ".join(MUST_HAVE_SKILLS["ranking_evaluation"]["keywords"] +
                             MUST_HAVE_SKILLS["ranking_evaluation"]["phrases"])
        assert "a/b testing" in all_text or "ab test" in all_text

    # Nice-to-have: LLM fine-tuning
    def test_lora_in_llm_finetuning(self):
        kws = NICE_TO_HAVE_SKILLS["llm_finetuning"]["keywords"]
        assert "lora" in kws and "qlora" in kws

    def test_peft_in_llm_finetuning(self):
        kws = NICE_TO_HAVE_SKILLS["llm_finetuning"]["keywords"]
        assert "peft" in kws

    # Nice-to-have: learning to rank
    def test_xgboost_in_ltr(self):
        kws = NICE_TO_HAVE_SKILLS["learning_to_rank"]["keywords"]
        assert "xgboost" in kws

    # Domain: vector DB
    def test_vector_db_keywords_nonempty(self):
        assert len(VECTOR_DB_PATTERNS["keywords"]) >= 5

    # Disqualifiers
    def test_consulting_firms_in_disqualifier(self):
        firms = DISQUALIFIER_PATTERNS["consulting_only_career"]["consulting_firms"]
        assert "TCS" in firms
        assert "Infosys" in firms
        assert "Wipro" in firms
        assert "Accenture" in firms

    def test_langchain_disqualifier_present(self):
        assert "langchain_only_no_tier1" in DISQUALIFIER_PATTERNS

    def test_title_chaser_disqualifier_present(self):
        assert "title_chaser" in DISQUALIFIER_PATTERNS

    # Location
    def test_preferred_cities_include_noida_pune(self):
        cities = LOCATION_REQUIREMENTS["preferred_cities"]
        assert "Noida" in cities
        assert "Pune" in cities

    def test_india_is_required_country(self):
        assert "India" in LOCATION_REQUIREMENTS["required_countries"]

    def test_visa_sponsorship_false(self):
        assert LOCATION_REQUIREMENTS["visa_sponsorship"] is False

    # Experience
    def test_experience_range(self):
        assert EXPERIENCE_REQUIREMENTS["stated_range_min"] == 5
        assert EXPERIENCE_REQUIREMENTS["stated_range_max"] == 9

    def test_experience_ideal_range(self):
        assert EXPERIENCE_REQUIREMENTS["ideal_range_min"] == 6
        assert EXPERIENCE_REQUIREMENTS["ideal_range_max"] == 8


# ─────────────────────────────────────────────────────────────────────────────
# §C  Embedding text tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingText:

    def test_embedding_text_nonempty(self):
        t = get_jd_embedding_text()
        assert len(t) > 100

    def test_embedding_text_contains_retrieval(self):
        t = get_jd_embedding_text().lower()
        assert "retrieval" in t

    def test_embedding_text_contains_vector(self):
        t = get_jd_embedding_text().lower()
        assert "vector" in t

    def test_embedding_text_contains_india(self):
        t = get_jd_embedding_text().lower()
        assert "india" in t

    def test_embedding_text_contains_production(self):
        t = get_jd_embedding_text().lower()
        assert "production" in t

    def test_embedding_text_contains_ndcg(self):
        t = get_jd_embedding_text().lower()
        assert "ndcg" in t

    def test_embedding_text_returns_string(self):
        assert isinstance(get_jd_embedding_text(), str)


# ─────────────────────────────────────────────────────────────────────────────
# §D  score_skills_match tests
# ─────────────────────────────────────────────────────────────────────────────

# Helpers
_IDEAL_SKILLS = [
    "sentence-transformers", "FAISS", "Elasticsearch", "NDCG", "MRR",
    "Python", "PyTorch", "Pinecone", "vector search", "hybrid search",
    "embeddings", "retrieval", "NLP",
]
_IDEAL_CAREER = (
    "Built a production embeddings-based retrieval system using sentence-transformers "
    "and FAISS deployed to real users. Designed NDCG/MRR evaluation frameworks. "
    "A/B tested ranking improvements. Deployed to production at scale."
)

_WEAK_SKILLS  = ["PowerPoint", "Excel", "VLOOKUP", "Pivot Tables"]
_WEAK_CAREER  = "Managed spreadsheets and prepared reports for management."

_LANGCHAIN_ONLY_SKILLS = ["langchain", "LangChain", "chatgpt"]
_LANGCHAIN_CAREER      = "Built a chatbot using LangChain calling OpenAI API."


class TestScoreSkillsMatch:

    def test_returns_dict(self):
        result = score_skills_match([])
        assert isinstance(result, dict)

    def test_all_expected_keys_present(self):
        result = score_skills_match(["python"])
        expected_keys = {
            "must_have_embeddings_retrieval",
            "must_have_vector_databases",
            "must_have_python_engineering",
            "must_have_ranking_evaluation",
            "overall_must_have",
            "nice_llm_finetuning",
            "nice_learning_to_rank",
            "nice_hrtech_marketplace",
            "nice_distributed_systems",
            "nice_open_source_ml",
            "overall_nice",
            "vector_db",
            "retrieval",
            "ranking",
            "evaluation",
            "production_ml",
            "composite_skill_score",
        }
        assert expected_keys.issubset(result.keys())

    def test_all_scores_in_0_1(self):
        result = score_skills_match(_IDEAL_SKILLS, _IDEAL_CAREER)
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"Score {k}={v} out of [0,1]"

    def test_ideal_candidate_scores_high_overall(self):
        result = score_skills_match(_IDEAL_SKILLS, _IDEAL_CAREER)
        assert result["composite_skill_score"] > 0.60

    def test_weak_candidate_scores_low_overall(self):
        result = score_skills_match(_WEAK_SKILLS, _WEAK_CAREER)
        assert result["composite_skill_score"] < 0.20

    def test_ideal_beats_weak_on_must_have(self):
        ideal = score_skills_match(_IDEAL_SKILLS, _IDEAL_CAREER)
        weak  = score_skills_match(_WEAK_SKILLS,  _WEAK_CAREER)
        assert ideal["overall_must_have"] > weak["overall_must_have"]

    def test_python_skill_detected(self):
        result = score_skills_match(["Python"])
        assert result["must_have_python_engineering"] > 0

    def test_case_insensitive_skill_matching(self):
        upper  = score_skills_match(["FAISS", "NDCG", "PYTHON"])
        lower  = score_skills_match(["faiss", "ndcg", "python"])
        assert upper == lower

    def test_empty_skills_no_crash(self):
        result = score_skills_match([], "", "")
        assert result["composite_skill_score"] == 0.0

    def test_production_ml_score_from_career_text(self):
        result = score_skills_match(
            [], career_text="deployed to production serving at scale"
        )
        assert result["production_ml"] > 0

    def test_lora_boosts_nice_llm_finetuning(self):
        with_lora    = score_skills_match(["lora", "qlora", "peft"])
        without_lora = score_skills_match(["python"])
        assert with_lora["nice_llm_finetuning"] > without_lora["nice_llm_finetuning"]

    def test_vector_db_score_from_pinecone(self):
        result = score_skills_match(["pinecone"])
        assert result["vector_db"] > 0

    def test_evaluation_score_from_ndcg(self):
        result = score_skills_match(["NDCG", "MRR", "MAP"])
        assert result["evaluation"] > 0

    def test_composite_is_weighted_combination(self):
        # Should not be 1.0 for a candidate with only one nice-to-have skill
        result = score_skills_match(["kaggle"])
        assert result["composite_skill_score"] < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# §E  detect_disqualifiers tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_flat(**overrides) -> dict:
    base = {
        "is_consulting_only":       False,
        "most_recent_title":        "Senior ML Engineer",
        "current_title":            "Senior ML Engineer",
        "skill_names":              ["python", "pytorch", "faiss"],
        "tier1_skill_count":        3,
        "career_companies":         ["Flipkart", "Swiggy"],
        "n_career_roles":           2,
        "total_career_months":      60,
        "country":                  "India",
        "willing_to_relocate":      True,
        "notice_period_days":       30,
        "days_since_last_active":   5,
        "open_to_work_flag":        True,
        "recruiter_response_rate":  0.80,
    }
    base.update(overrides)
    return base


class TestDetectDisqualifiers:

    def test_clean_candidate_no_disqualifiers(self):
        flat = _make_flat()
        result = detect_disqualifiers(flat)
        assert result == []

    def test_consulting_only_triggers(self):
        flat = _make_flat(is_consulting_only=True)
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "consulting_only_career" in names

    def test_nontechnical_title_triggers(self):
        flat = _make_flat(most_recent_title="HR Manager", current_title="HR Manager")
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "nontechnical_primary_title" in names

    def test_marketing_manager_triggers_nontechnical(self):
        flat = _make_flat(most_recent_title="Marketing Manager")
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "nontechnical_primary_title" in names

    def test_langchain_without_tier1_triggers(self):
        flat = _make_flat(skill_names=["langchain"], tier1_skill_count=0)
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "langchain_only_no_tier1" in names

    def test_langchain_with_tier1_no_trigger(self):
        flat = _make_flat(skill_names=["langchain", "faiss", "ndcg"], tier1_skill_count=2)
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "langchain_only_no_tier1" not in names

    def test_title_chaser_triggers_at_low_tenure(self):
        # 5 roles in 40 months → avg 8 months each
        flat = _make_flat(n_career_roles=5, total_career_months=40)
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "title_chaser" in names

    def test_title_chaser_not_triggered_at_normal_tenure(self):
        # 3 roles in 72 months → avg 24 months each
        flat = _make_flat(n_career_roles=3, total_career_months=72)
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "title_chaser" not in names

    def test_title_chaser_not_triggered_under_min_roles(self):
        # Only 2 roles → below min_roles_to_check=3, skip the check
        flat = _make_flat(n_career_roles=2, total_career_months=10)
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "title_chaser" not in names

    def test_outside_india_no_relocation_triggers(self):
        flat = _make_flat(country="United States", willing_to_relocate=False)
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "outside_india_no_relocation" in names

    def test_outside_india_willing_no_trigger(self):
        flat = _make_flat(country="United States", willing_to_relocate=True)
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "outside_india_no_relocation" not in names

    def test_long_notice_period_triggers(self):
        flat = _make_flat(notice_period_days=120)
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "long_notice_period" in names

    def test_short_notice_no_trigger(self):
        flat = _make_flat(notice_period_days=15)
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "long_notice_period" not in names

    def test_inactive_not_available_triggers(self):
        flat = _make_flat(
            days_since_last_active=200,
            open_to_work_flag=False,
            recruiter_response_rate=0.02,
        )
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "inactive_not_available" in names

    def test_disqualifier_result_has_required_keys(self):
        flat = _make_flat(is_consulting_only=True)
        for d in detect_disqualifiers(flat):
            assert "name" in d
            assert "description" in d
            assert "penalty_factor" in d
            assert "evidence" in d

    def test_multiple_disqualifiers_can_stack(self):
        flat = _make_flat(
            is_consulting_only=True,
            most_recent_title="HR Manager",
        )
        result = detect_disqualifiers(flat)
        assert len(result) >= 2

    def test_cv_robotics_without_nlp_triggers(self):
        flat = _make_flat(
            skill_names=["computer vision", "object detection", "opencv"],
            tier1_skill_count=0,
        )
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "cv_speech_robotics_no_nlp" in names

    def test_cv_with_nlp_no_trigger(self):
        flat = _make_flat(
            skill_names=["computer vision", "object detection", "nlp", "embeddings"],
            tier1_skill_count=2,
        )
        names = [d["name"] for d in detect_disqualifiers(flat)]
        assert "cv_speech_robotics_no_nlp" not in names


# ─────────────────────────────────────────────────────────────────────────────
# §F  score_location_match tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreLocationMatch:

    def test_india_noida_perfect_score(self):
        s = score_location_match("India", "Noida", True)
        assert s == 1.00

    def test_india_pune_perfect_score(self):
        s = score_location_match("India", "Pune", False)
        assert s == 1.00

    def test_india_hyderabad_acceptable(self):
        s = score_location_match("India", "Hyderabad", False)
        assert s == 0.80

    def test_india_mumbai_acceptable(self):
        s = score_location_match("India", "Mumbai", False)
        assert s == 0.80

    def test_india_bangalore_acceptable(self):
        s = score_location_match("India", "Bangalore", False)
        assert s == 0.80

    def test_india_other_city_partial(self):
        s = score_location_match("India", "Jaipur", False)
        assert s == 0.65

    def test_outside_india_willing_penalised(self):
        s = score_location_match("United States", "San Francisco", True)
        assert s == 0.45

    def test_outside_india_not_willing_heavily_penalised(self):
        s = score_location_match("Germany", "Berlin", False)
        assert s == 0.10

    def test_scores_monotone_india_gt_outside(self):
        india_score   = score_location_match("India", "Noida", False)
        outside_score = score_location_match("Canada", "Toronto", False)
        assert india_score > outside_score

    def test_willing_to_relocate_improves_outside_india(self):
        willing     = score_location_match("USA", "New York", True)
        not_willing = score_location_match("USA", "New York", False)
        assert willing > not_willing


# ─────────────────────────────────────────────────────────────────────────────
# §G  score_experience_fit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreExperienceFit:

    def test_seven_years_near_peak(self):
        s = score_experience_fit(7.0)
        assert s > 0.95   # Gaussian peak ≈ 1.0 at mu=7

    def test_six_years_still_high(self):
        s = score_experience_fit(6.0)
        assert s > 0.80

    def test_eight_years_still_high(self):
        s = score_experience_fit(8.0)
        assert s > 0.80

    def test_three_years_lower(self):
        s = score_experience_fit(3.0)
        assert s < 0.70

    def test_fifteen_years_lower(self):
        s = score_experience_fit(15.0)
        assert s < 0.50

    def test_one_year_floor(self):
        s = score_experience_fit(1.0)
        assert s == 0.05

    def test_zero_years_floor(self):
        s = score_experience_fit(0.0)
        assert s == 0.05

    def test_scores_in_0_1(self):
        for yoe in [0, 1, 3, 5, 7, 9, 12, 15, 20]:
            s = score_experience_fit(float(yoe))
            assert 0.0 <= s <= 1.0, f"yoe={yoe} gave score={s}"

    def test_gaussian_symmetric_around_peak(self):
        # Score at 7-2=5 should equal score at 7+2=9
        s_below = score_experience_fit(5.0)
        s_above = score_experience_fit(9.0)
        assert abs(s_below - s_above) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# §H  score_notice_period tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreNoticePeriod:

    def test_zero_days_best_score(self):
        assert score_notice_period(0) == 1.00

    def test_thirty_days_best_score(self):
        assert score_notice_period(30) == 1.00

    def test_sixty_days_good_score(self):
        s = score_notice_period(60)
        assert 0.75 <= s <= 0.85

    def test_ninety_days_acceptable(self):
        s = score_notice_period(90)
        assert 0.55 <= s <= 0.65

    def test_over_120_very_low(self):
        s = score_notice_period(150)
        assert s <= 0.25

    def test_monotone_decreasing(self):
        scores = [score_notice_period(d) for d in [0, 30, 60, 90, 120, 180]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"score_notice_period not monotone: {scores}"
            )

    def test_scores_in_0_1(self):
        for d in [0, 15, 30, 45, 60, 90, 120, 180]:
            s = score_notice_period(d)
            assert 0.0 <= s <= 1.0, f"notice_period_days={d} gave score={s}"


# ─────────────────────────────────────────────────────────────────────────────
# §I  compute_disqualifier_penalty tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeDisqualifierPenalty:

    def test_clean_candidate_penalty_is_one(self):
        flat = _make_flat()
        assert compute_disqualifier_penalty(flat) == 1.0

    def test_single_disqualifier_reduces_penalty(self):
        flat = _make_flat(is_consulting_only=True)
        p = compute_disqualifier_penalty(flat)
        assert 0.0 < p < 1.0

    def test_multiple_disqualifiers_compound(self):
        single = compute_disqualifier_penalty(_make_flat(is_consulting_only=True))
        double = compute_disqualifier_penalty(_make_flat(
            is_consulting_only=True,
            most_recent_title="HR Manager",
        ))
        assert double < single

    def test_consulting_only_penalty_value(self):
        flat = _make_flat(is_consulting_only=True)
        p = compute_disqualifier_penalty(flat)
        # consulting_only penalty_factor = 0.15; only one disqualifier
        assert abs(p - 0.15) < 0.001

    def test_penalty_is_float(self):
        assert isinstance(compute_disqualifier_penalty(_make_flat()), float)

    def test_penalty_never_negative(self):
        flat = _make_flat(
            is_consulting_only=True,
            most_recent_title="HR Manager",
            skill_names=["langchain"],
            tier1_skill_count=0,
        )
        p = compute_disqualifier_penalty(flat)
        assert p >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# §J  JD_REQUIREMENTS master dict tests
# ─────────────────────────────────────────────────────────────────────────────

class TestJDRequirements:

    _EXPECTED_KEYS = {
        "metadata", "must_have_skills", "nice_to_have_skills",
        "vector_db_patterns", "retrieval_patterns", "ranking_patterns",
        "evaluation_patterns", "production_ml_patterns",
        "shipper_patterns", "startup_patterns", "location_requirements",
        "experience_requirements", "salary_expectations",
        "disqualifier_patterns",
    }

    def test_all_top_level_keys_present(self):
        missing = self._EXPECTED_KEYS - set(JD_REQUIREMENTS)
        assert not missing, f"JD_REQUIREMENTS missing top-level keys: {missing}"

    def test_metadata_contains_title(self):
        assert "Senior AI Engineer" in JD_REQUIREMENTS["metadata"]["title"]

    def test_must_have_skills_same_object(self):
        assert JD_REQUIREMENTS["must_have_skills"] is MUST_HAVE_SKILLS

    def test_disqualifier_patterns_same_object(self):
        assert JD_REQUIREMENTS["disqualifier_patterns"] is DISQUALIFIER_PATTERNS

    def test_salary_currency_inr(self):
        assert JD_REQUIREMENTS["salary_expectations"]["currency"] == "INR"

    def test_jd_metadata_no_visa_sponsorship(self):
        assert JD_REQUIREMENTS["metadata"]["visa_sponsor"] is False

    def test_shipper_patterns_in_requirements(self):
        assert "shipped" in JD_REQUIREMENTS["shipper_patterns"]["keywords"]

    def test_startup_patterns_in_requirements(self):
        assert "startup" in JD_REQUIREMENTS["startup_patterns"]["keywords"]
