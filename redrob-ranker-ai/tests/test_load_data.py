"""
tests/test_load_data.py — Comprehensive tests for src/load_data.py

Coverage
--------
    § A  Pydantic model validation
    § B  File format detection
    § C  iter_candidates streaming
    § D  load_candidates + load_flat
    § E  build_candidate_text content and order
    § F  flatten_candidate structure
    § G  Derived features (consulting, skills, days_since, etc.)
    § H  Edge cases and error handling
    § I  Integration — sample_candidates.json
    § J  Integration — candidates.jsonl (real file, first 200 records)
    § K  Memory benchmark (smoke test)
"""

from __future__ import annotations

import gzip
import json
import math
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.load_data import (
    CandidateRecord,
    SkillEntry,
    CareerEntry,
    EducationEntry,
    RedrobSignals,
    Profile,
    SalaryRange,
    build_candidate_text,
    flatten_candidate,
    iter_candidates,
    load_candidates,
    load_flat,
    validate_schema,
    benchmark_memory,
)
from src.config import (
    SAMPLE_CANDIDATES_FILE,
    CANDIDATES_FILE,
    REFERENCE_DATE,
    CONSULTING_FIRMS,
    TIER1_SKILLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

MINIMAL_DICT: dict[str, Any] = {
    "candidate_id": "CAND_0000001",
    "profile": {
        "anonymized_name": "Arjun Mehta",
        "headline": "Senior ML Engineer | Embeddings & Retrieval",
        "summary": (
            "7 years building production retrieval systems. "
            "Shipped FAISS-based vector search at scale. "
            "Expert in NDCG evaluation and hybrid search."
        ),
        "location": "Bangalore",
        "country": "India",
        "years_of_experience": 7.0,
        "current_title": "ML Engineer",
        "current_company": "ProductCo",
        "current_company_size": "51-200",
        "current_industry": "Technology",
    },
    "career_history": [
        {
            "company": "ProductCo",
            "title": "ML Engineer",
            "start_date": "2022-01-01",
            "end_date": None,
            "duration_months": 30,
            "is_current": True,
            "industry": "Technology",
            "company_size": "51-200",
            "description": (
                "Built vector search using FAISS and sentence-transformers. "
                "Designed NDCG evaluation framework for ranking."
            ),
        },
        {
            "company": "TCS",
            "title": "Software Engineer",
            "start_date": "2018-01-01",
            "end_date": "2021-12-31",
            "duration_months": 48,
            "is_current": False,
            "industry": "IT Services",
            "company_size": "10001+",
            "description": "Java backend development for banking clients.",
        },
    ],
    "education": [
        {
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2014,
            "end_year": 2018,
            "grade": "8.5 CGPA",
            "tier": "tier_1",
        }
    ],
    "skills": [
        {"name": "FAISS",      "proficiency": "advanced",     "endorsements": 15, "duration_months": 24},
        {"name": "Python",     "proficiency": "expert",       "endorsements": 50, "duration_months": 60},
        {"name": "NLP",        "proficiency": "advanced",     "endorsements": 10, "duration_months": 30},
        {"name": "Pinecone",   "proficiency": "intermediate", "endorsements": 5,  "duration_months": 12},
        {"name": "LangChain",  "proficiency": "intermediate", "endorsements": 2,  "duration_months": 6},
    ],
    "certifications": [
        {"name": "AWS Machine Learning Specialty", "issuer": "Amazon", "year": 2023}
    ],
    "languages": [
        {"language": "English", "proficiency": "professional"}
    ],
    "redrob_signals": {
        "profile_completeness_score": 90.0,
        "signup_date": "2025-01-01",
        "last_active_date": "2026-06-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 50,
        "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.85,
        "avg_response_time_hours": 4.0,
        "skill_assessment_scores": {"Python": 92.0, "NLP": 78.0},
        "connection_count": 500,
        "endorsements_received": 75,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 30.0, "max": 50.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 75.0,
        "search_appearance_30d": 200,
        "saved_by_recruiters_30d": 10,
        "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.7,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    },
}

CONSULTING_ONLY_DICT: dict[str, Any] = {
    **deepcopy(MINIMAL_DICT),
    "candidate_id": "CAND_CONSULTING",
    "career_history": [
        {
            "company": "TCS",
            "title": "Software Engineer",
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": 36,
            "is_current": True,
            "industry": "IT Services",
            "company_size": "10001+",
            "description": "Java backend at banking clients.",
        },
        {
            "company": "Infosys",
            "title": "Analyst",
            "start_date": "2018-01-01",
            "end_date": "2019-12-31",
            "duration_months": 24,
            "is_current": False,
            "industry": "IT Services",
            "company_size": "10001+",
            "description": "Process automation at insurance client.",
        },
    ],
}

HONEYPOT_SKILLS_DICT: dict[str, Any] = {
    **deepcopy(MINIMAL_DICT),
    "candidate_id": "CAND_HONEYPOT",
    # 5 expert skills all with 0 duration — honeypot signal
    "skills": [
        {"name": s, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
        for s in ["Python", "FAISS", "NLP", "Embeddings", "RAG", "Milvus"]
    ],
}

INACTIVE_DICT: dict[str, Any] = {
    **deepcopy(MINIMAL_DICT),
    "candidate_id": "CAND_INACTIVE",
    "redrob_signals": {
        **deepcopy(MINIMAL_DICT["redrob_signals"]),
        "last_active_date": "2025-11-01",   # 226 days before REFERENCE_DATE
        "open_to_work_flag": False,
        "recruiter_response_rate": 0.05,
    },
}


@pytest.fixture
def minimal_record() -> CandidateRecord:
    return CandidateRecord.model_validate(MINIMAL_DICT)


@pytest.fixture
def minimal_flat(minimal_record) -> dict:
    return flatten_candidate(minimal_record)


@pytest.fixture
def tmp_jsonl(tmp_path) -> Path:
    """Write three MINIMAL_DICT records to a temp .jsonl file."""
    path = tmp_path / "test.jsonl"
    records = []
    for i in range(3):
        r = deepcopy(MINIMAL_DICT)
        r["candidate_id"] = f"CAND_{i:07d}"
        records.append(r)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


@pytest.fixture
def tmp_jsonl_gz(tmp_path) -> Path:
    """Write the same three records to a gzip-compressed .jsonl.gz file."""
    path = tmp_path / "test.jsonl.gz"
    records = []
    for i in range(3):
        r = deepcopy(MINIMAL_DICT)
        r["candidate_id"] = f"CAND_{i:07d}"
        records.append(r)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


@pytest.fixture
def tmp_json_array(tmp_path) -> Path:
    """Write three records to a temp .json array file (sample format)."""
    path = tmp_path / "sample.json"
    records = []
    for i in range(3):
        r = deepcopy(MINIMAL_DICT)
        r["candidate_id"] = f"CAND_{i:07d}"
        records.append(r)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh)
    return path


sample_json_exists  = pytest.mark.skipif(
    not SAMPLE_CANDIDATES_FILE.exists(),
    reason=f"sample_candidates.json not found at {SAMPLE_CANDIDATES_FILE}",
)
candidates_jsonl_exists = pytest.mark.skipif(
    not CANDIDATES_FILE.exists(),
    reason=f"candidates.jsonl not found at {CANDIDATES_FILE}",
)


# ─────────────────────────────────────────────────────────────────────────────
# § A  Pydantic model validation
# ─────────────────────────────────────────────────────────────────────────────

class TestPydanticModels:

    def test_minimal_dict_parses(self):
        record = CandidateRecord.model_validate(MINIMAL_DICT)
        assert record.candidate_id == "CAND_0000001"

    def test_profile_fields(self, minimal_record):
        p = minimal_record.profile
        assert p.headline == "Senior ML Engineer | Embeddings & Retrieval"
        assert p.country == "India"
        assert p.years_of_experience == 7.0

    def test_career_entry_null_end_date(self, minimal_record):
        current = minimal_record.career_history[0]
        assert current.is_current is True
        assert current.end_date is None

    def test_skill_proficiency_validated(self):
        with pytest.raises(Exception):
            SkillEntry(name="Test", proficiency="godlike", endorsements=0, duration_months=0)

    def test_negative_endorsements_clamped_to_zero(self):
        s = SkillEntry(name="X", proficiency="expert", endorsements=-5, duration_months=0)
        assert s.endorsements == 0

    def test_salary_range_parsed(self, minimal_record):
        sr = minimal_record.redrob_signals.expected_salary_range_inr_lpa
        assert sr.min == 30.0
        assert sr.max == 50.0

    def test_offer_acceptance_rate_negative_one_allowed(self):
        sig = RedrobSignals(offer_acceptance_rate=-1.0)
        assert sig.offer_acceptance_rate == -1.0

    def test_github_score_negative_one_allowed(self):
        sig = RedrobSignals(github_activity_score=-1.0)
        assert sig.github_activity_score == -1.0

    def test_profile_completeness_clamped(self):
        sig = RedrobSignals(profile_completeness_score=150.0)
        assert sig.profile_completeness_score == 100.0

    def test_empty_skill_assessment_scores(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["redrob_signals"]["skill_assessment_scores"] = {}
        record = CandidateRecord.model_validate(raw)
        assert record.redrob_signals.skill_assessment_scores == {}

    def test_missing_optional_fields_use_defaults(self):
        minimal = {
            "candidate_id": "CAND_TEST",
            "profile": deepcopy(MINIMAL_DICT["profile"]),
            "career_history": [],
            "education": [],
            "skills": [],
            "redrob_signals": MINIMAL_DICT["redrob_signals"],
        }
        record = CandidateRecord.model_validate(minimal)
        assert record.certifications == []
        assert record.languages == []

    def test_validate_schema_returns_true_for_valid(self):
        assert validate_schema(MINIMAL_DICT) is True

    def test_validate_schema_returns_false_for_invalid(self):
        bad = {"candidate_id": "X"}  # missing required 'profile'
        assert validate_schema(bad) is False


# ─────────────────────────────────────────────────────────────────────────────
# § B  File format detection
# ─────────────────────────────────────────────────────────────────────────────

class TestFileFormatDetection:

    def test_jsonl_format_streams_correctly(self, tmp_jsonl):
        records = list(iter_candidates(tmp_jsonl))
        assert len(records) == 3

    def test_json_array_format_loads_correctly(self, tmp_json_array):
        records = list(iter_candidates(tmp_json_array))
        assert len(records) == 3

    def test_jsonl_gz_format_loads_correctly(self, tmp_jsonl_gz):
        records = list(iter_candidates(tmp_jsonl_gz))
        assert len(records) == 3

    def test_jsonl_and_gz_produce_same_records(self, tmp_jsonl, tmp_jsonl_gz):
        jsonl_ids = [r.candidate_id for r in iter_candidates(tmp_jsonl)]
        gz_ids    = [r.candidate_id for r in iter_candidates(tmp_jsonl_gz)]
        assert jsonl_ids == gz_ids

    def test_nonexistent_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(iter_candidates(tmp_path / "nonexistent.jsonl"))

    def test_non_array_json_raises_value_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"not": "an_array"}')
        with pytest.raises(ValueError, match="Expected a JSON array"):
            list(iter_candidates(path))


# ─────────────────────────────────────────────────────────────────────────────
# § C  iter_candidates streaming
# ─────────────────────────────────────────────────────────────────────────────

class TestIterCandidates:

    def test_is_generator(self, tmp_jsonl):
        import types
        gen = iter_candidates(tmp_jsonl)
        assert isinstance(gen, types.GeneratorType)

    def test_count(self, tmp_jsonl):
        assert sum(1 for _ in iter_candidates(tmp_jsonl)) == 3

    def test_max_records_limits_output(self, tmp_jsonl):
        records = list(iter_candidates(tmp_jsonl, max_records=2))
        assert len(records) == 2

    def test_max_records_zero_returns_nothing(self, tmp_jsonl):
        records = list(iter_candidates(tmp_jsonl, max_records=0))
        assert len(records) == 0

    def test_invalid_record_skipped_when_skip_true(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        records = [MINIMAL_DICT, {"bad": "record"}, deepcopy(MINIMAL_DICT)]
        records[2]["candidate_id"] = "CAND_0000002"
        with open(path, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        result = list(iter_candidates(path, skip_invalid=True))
        assert len(result) == 2  # bad record skipped

    def test_invalid_record_raises_when_skip_false(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"bad": "record"}) + "\n")
        with pytest.raises(ValueError):
            list(iter_candidates(path, skip_invalid=False))

    def test_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        with open(path, "w") as fh:
            fh.write(json.dumps(MINIMAL_DICT) + "\n")
            fh.write("\n")
            fh.write("   \n")
            r2 = deepcopy(MINIMAL_DICT)
            r2["candidate_id"] = "CAND_0000002"
            fh.write(json.dumps(r2) + "\n")
        result = list(iter_candidates(path))
        assert len(result) == 2

    def test_candidate_ids_are_unique(self, tmp_jsonl):
        ids = [r.candidate_id for r in iter_candidates(tmp_jsonl)]
        assert len(ids) == len(set(ids))

    def test_records_are_candidate_record_instances(self, tmp_jsonl):
        for record in iter_candidates(tmp_jsonl):
            assert isinstance(record, CandidateRecord)


# ─────────────────────────────────────────────────────────────────────────────
# § D  load_candidates + load_flat
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadFunctions:

    def test_load_candidates_returns_list(self, tmp_jsonl):
        result = load_candidates(tmp_jsonl, show_progress=False)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_load_flat_returns_dataframe(self, tmp_jsonl):
        df = load_flat(tmp_jsonl, show_progress=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_load_flat_has_candidate_id_column(self, tmp_jsonl):
        df = load_flat(tmp_jsonl, show_progress=False)
        assert "candidate_id" in df.columns

    def test_load_flat_candidate_id_is_index(self, tmp_jsonl):
        df = load_flat(tmp_jsonl, show_progress=False)
        assert df.index.name == "candidate_id"

    def test_load_flat_max_records(self, tmp_jsonl):
        df = load_flat(tmp_jsonl, max_records=1, show_progress=False)
        assert len(df) == 1

    def test_load_flat_empty_file_returns_empty_df(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        df = load_flat(path, show_progress=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_load_flat_required_columns_present(self, tmp_jsonl):
        df = load_flat(tmp_jsonl, show_progress=False)
        required = [
            "candidate_id", "headline", "summary", "country",
            "years_of_experience", "current_title", "current_company",
            "career_companies", "career_titles", "career_descriptions_text",
            "total_career_months", "n_career_roles", "product_company_ratio",
            "is_consulting_only", "skill_names", "skill_count",
            "tier1_skill_count", "tier2_skill_count", "expert_zero_duration_count",
            "highest_edu_tier", "cert_names", "cert_count",
            "profile_completeness_score", "days_since_last_active",
            "open_to_work_flag", "recruiter_response_rate",
            "notice_period_days", "salary_min_lpa", "salary_max_lpa",
            "salary_inverted", "willing_to_relocate", "github_activity_score",
            "interview_completion_rate", "offer_acceptance_rate",
            "verified_email", "verified_phone", "linkedin_connected",
            "skill_assessment_scores", "candidate_text",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_flat_no_duplicate_ids(self, tmp_jsonl):
        df = load_flat(tmp_jsonl, show_progress=False)
        assert df["candidate_id"].nunique() == len(df)


# ─────────────────────────────────────────────────────────────────────────────
# § E  build_candidate_text
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCandidateText:

    def test_returns_string(self, minimal_record):
        text = build_candidate_text(minimal_record)
        assert isinstance(text, str)

    def test_not_empty(self, minimal_record):
        assert len(build_candidate_text(minimal_record)) > 0

    def test_contains_headline(self, minimal_record):
        text = build_candidate_text(minimal_record)
        assert "Senior ML Engineer" in text

    def test_contains_summary(self, minimal_record):
        text = build_candidate_text(minimal_record)
        assert "retrieval systems" in text

    def test_contains_current_title(self, minimal_record):
        text = build_candidate_text(minimal_record)
        assert "ML Engineer" in text

    def test_contains_skill_name(self, minimal_record):
        text = build_candidate_text(minimal_record)
        assert "FAISS" in text

    def test_contains_certification(self, minimal_record):
        text = build_candidate_text(minimal_record)
        assert "AWS Machine Learning Specialty" in text

    def test_contains_education(self, minimal_record):
        text = build_candidate_text(minimal_record)
        assert "IIT Bombay" in text

    def test_contains_career_description(self, minimal_record):
        text = build_candidate_text(minimal_record)
        assert "vector search" in text.lower()

    def test_headline_appears_before_career_descriptions(self, minimal_record):
        text = build_candidate_text(minimal_record)
        headline_pos  = text.find("Senior ML Engineer")
        career_pos    = text.find("ProductCo")
        assert headline_pos < career_pos, "Headline should appear before career content"

    def test_summary_appears_before_skills(self, minimal_record):
        text = build_candidate_text(minimal_record)
        summary_pos = text.find("retrieval systems")
        skills_pos  = text.find("Expert/Advanced")
        assert summary_pos < skills_pos, "Summary should appear before skills section"

    def test_truncated_to_max_chars(self, minimal_record):
        from src.config import MAX_CANDIDATE_TEXT_CHARS
        text = build_candidate_text(minimal_record)
        assert len(text) <= MAX_CANDIDATE_TEXT_CHARS

    def test_no_certifications_omits_section(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["certifications"] = []
        record = CandidateRecord.model_validate(raw)
        text = build_candidate_text(record)
        assert "Certifications:" not in text

    def test_no_education_omits_section(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["education"] = []
        record = CandidateRecord.model_validate(raw)
        text = build_candidate_text(record)
        assert "Education:" not in text

    def test_expert_skills_in_high_proficiency_group(self, minimal_record):
        text = build_candidate_text(minimal_record)
        assert "Expert/Advanced:" in text
        assert "Python" in text

    def test_intermediate_skills_in_low_proficiency_group(self, minimal_record):
        text = build_candidate_text(minimal_record)
        assert "Intermediate/Beginner:" in text
        assert "Pinecone" in text

    def test_empty_candidate_does_not_crash(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["profile"]["headline"] = ""
        raw["profile"]["summary"] = ""
        raw["skills"] = []
        raw["certifications"] = []
        raw["education"] = []
        raw["career_history"] = []
        record = CandidateRecord.model_validate(raw)
        text = build_candidate_text(record)
        assert isinstance(text, str)


# ─────────────────────────────────────────────────────────────────────────────
# § F  flatten_candidate structure
# ─────────────────────────────────────────────────────────────────────────────

class TestFlattenCandidate:

    def test_returns_dict(self, minimal_record):
        flat = flatten_candidate(minimal_record)
        assert isinstance(flat, dict)

    def test_candidate_id_preserved(self, minimal_flat):
        assert minimal_flat["candidate_id"] == "CAND_0000001"

    def test_career_companies_is_list(self, minimal_flat):
        assert isinstance(minimal_flat["career_companies"], list)
        assert "ProductCo" in minimal_flat["career_companies"]
        assert "TCS" in minimal_flat["career_companies"]

    def test_career_titles_is_list(self, minimal_flat):
        assert isinstance(minimal_flat["career_titles"], list)
        assert "ML Engineer" in minimal_flat["career_titles"]

    def test_career_descriptions_text_is_string(self, minimal_flat):
        assert isinstance(minimal_flat["career_descriptions_text"], str)
        assert len(minimal_flat["career_descriptions_text"]) > 0

    def test_skill_names_is_list(self, minimal_flat):
        assert isinstance(minimal_flat["skill_names"], list)
        assert "Python" in minimal_flat["skill_names"]

    def test_salary_min_max_flattened(self, minimal_flat):
        assert minimal_flat["salary_min_lpa"] == 30.0
        assert minimal_flat["salary_max_lpa"] == 50.0

    def test_salary_not_inverted_for_valid(self, minimal_flat):
        assert minimal_flat["salary_inverted"] is False

    def test_salary_inverted_flag_set_when_min_gt_max(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["redrob_signals"]["expected_salary_range_inr_lpa"] = {"min": 100.0, "max": 50.0}
        record = CandidateRecord.model_validate(raw)
        flat = flatten_candidate(record)
        assert flat["salary_inverted"] is True

    def test_skill_assessment_scores_preserved_as_dict(self, minimal_flat):
        scores = minimal_flat["skill_assessment_scores"]
        assert isinstance(scores, dict)
        assert scores.get("Python") == pytest.approx(92.0)

    def test_candidate_text_in_flat(self, minimal_flat):
        assert "candidate_text" in minimal_flat
        assert isinstance(minimal_flat["candidate_text"], str)
        assert len(minimal_flat["candidate_text"]) > 0

    def test_edu_fields_is_list(self, minimal_flat):
        assert isinstance(minimal_flat["edu_fields"], list)
        assert "Computer Science" in minimal_flat["edu_fields"]

    def test_cert_names_is_list(self, minimal_flat):
        assert isinstance(minimal_flat["cert_names"], list)
        assert "AWS Machine Learning Specialty" in minimal_flat["cert_names"]

    def test_n_career_roles_correct(self, minimal_flat):
        assert minimal_flat["n_career_roles"] == 2

    def test_most_recent_title_is_current_role(self, minimal_flat):
        assert minimal_flat["most_recent_title"] == "ML Engineer"

    def test_most_recent_company_is_current_company(self, minimal_flat):
        assert minimal_flat["most_recent_company"] == "ProductCo"


# ─────────────────────────────────────────────────────────────────────────────
# § G  Derived features
# ─────────────────────────────────────────────────────────────────────────────

class TestDerivedFeatures:

    # ── Consulting detection ─────────────────────────────────────────────────

    def test_is_consulting_only_false_for_product_candidate(self, minimal_flat):
        # ProductCo + TCS → NOT consulting-only (has one product company)
        assert minimal_flat["is_consulting_only"] is False

    def test_is_consulting_only_true_for_tcs_infosys_career(self):
        record = CandidateRecord.model_validate(CONSULTING_ONLY_DICT)
        flat = flatten_candidate(record)
        assert flat["is_consulting_only"] is True

    def test_product_company_ratio_partial(self, minimal_flat):
        # 1 product (ProductCo) out of 2 roles
        assert minimal_flat["product_company_ratio"] == pytest.approx(0.5)

    def test_product_company_ratio_zero_for_consulting_only(self):
        record = CandidateRecord.model_validate(CONSULTING_ONLY_DICT)
        flat = flatten_candidate(record)
        assert flat["product_company_ratio"] == pytest.approx(0.0)

    def test_product_company_ratio_one_for_no_consulting(self):
        raw = deepcopy(MINIMAL_DICT)
        for role in raw["career_history"]:
            role["company"] = "ProductStartup"
        record = CandidateRecord.model_validate(raw)
        flat = flatten_candidate(record)
        assert flat["product_company_ratio"] == pytest.approx(1.0)

    # ── Skill tier counts ────────────────────────────────────────────────────

    def test_tier1_skill_count_includes_faiss_python_nlp(self, minimal_flat):
        # FAISS, Python, NLP are all in TIER1_SKILLS
        assert minimal_flat["tier1_skill_count"] >= 3

    def test_tier1_count_zero_for_no_relevant_skills(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["skills"] = [
            {"name": "Excel",       "proficiency": "advanced", "endorsements": 5, "duration_months": 10},
            {"name": "PowerPoint",  "proficiency": "expert",   "endorsements": 3, "duration_months": 20},
        ]
        record = CandidateRecord.model_validate(raw)
        flat = flatten_candidate(record)
        assert flat["tier1_skill_count"] == 0

    def test_tier2_count_includes_finetuning_lora(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["skills"] = [
            {"name": "LoRA",         "proficiency": "advanced", "endorsements": 5, "duration_months": 10},
            {"name": "Fine-tuning",  "proficiency": "advanced", "endorsements": 5, "duration_months": 10},
        ]
        record = CandidateRecord.model_validate(raw)
        flat = flatten_candidate(record)
        assert flat["tier2_skill_count"] >= 1

    # ── Honeypot signals ─────────────────────────────────────────────────────

    def test_expert_zero_duration_count_honeypot(self):
        record = CandidateRecord.model_validate(HONEYPOT_SKILLS_DICT)
        flat = flatten_candidate(record)
        assert flat["expert_zero_duration_count"] == 6  # all 6 are expert + 0 duration

    def test_expert_zero_duration_zero_for_legit_candidate(self, minimal_flat):
        assert minimal_flat["expert_zero_duration_count"] == 0

    # ── Behavioral signals ───────────────────────────────────────────────────

    def test_days_since_last_active_computed_correctly(self, minimal_flat):
        # last_active_date = "2026-06-01", REFERENCE_DATE = 2026-06-15
        expected = (date(2026, 6, 15) - date(2026, 6, 1)).days
        assert minimal_flat["days_since_last_active"] == expected

    def test_days_since_last_active_large_for_inactive(self):
        record = CandidateRecord.model_validate(INACTIVE_DICT)
        flat = flatten_candidate(record)
        assert flat["days_since_last_active"] > 90

    def test_days_since_last_active_9999_when_date_missing(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["redrob_signals"]["last_active_date"] = None
        record = CandidateRecord.model_validate(raw)
        flat = flatten_candidate(record)
        assert flat["days_since_last_active"] == 9999

    def test_open_to_work_flag_preserved(self, minimal_flat):
        assert minimal_flat["open_to_work_flag"] is True

    def test_github_score_minus_one_preserved(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["redrob_signals"]["github_activity_score"] = -1.0
        record = CandidateRecord.model_validate(raw)
        flat = flatten_candidate(record)
        assert flat["github_activity_score"] == -1.0

    def test_offer_acceptance_rate_minus_one_preserved(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["redrob_signals"]["offer_acceptance_rate"] = -1.0
        record = CandidateRecord.model_validate(raw)
        flat = flatten_candidate(record)
        assert flat["offer_acceptance_rate"] == -1.0

    # ── Education tier ───────────────────────────────────────────────────────

    def test_highest_edu_tier_tier_1(self, minimal_flat):
        assert minimal_flat["highest_edu_tier"] == "tier_1"

    def test_highest_edu_tier_picks_best_when_multiple(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["education"] = [
            {"institution": "Local College", "degree": "B.Sc", "field_of_study": "Maths",
             "start_year": 2010, "end_year": 2014, "grade": None, "tier": "tier_3"},
            {"institution": "IIT Delhi", "degree": "M.Tech", "field_of_study": "CS",
             "start_year": 2014, "end_year": 2016, "grade": "9.0", "tier": "tier_1"},
        ]
        record = CandidateRecord.model_validate(raw)
        flat = flatten_candidate(record)
        assert flat["highest_edu_tier"] == "tier_1"

    def test_highest_edu_tier_unknown_when_no_education(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["education"] = []
        record = CandidateRecord.model_validate(raw)
        flat = flatten_candidate(record)
        assert flat["highest_edu_tier"] == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# § H  Edge cases and error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_career_history(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["career_history"] = []
        record = CandidateRecord.model_validate(raw)
        flat = flatten_candidate(record)
        assert flat["n_career_roles"] == 0
        assert flat["product_company_ratio"] == 0.0
        assert flat["is_consulting_only"] is False
        assert flat["total_career_months"] == 0

    def test_empty_skills(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["skills"] = []
        record = CandidateRecord.model_validate(raw)
        flat = flatten_candidate(record)
        assert flat["skill_count"] == 0
        assert flat["tier1_skill_count"] == 0

    def test_very_long_summary_truncated_in_text(self):
        from src.config import MAX_CANDIDATE_TEXT_CHARS
        raw = deepcopy(MINIMAL_DICT)
        raw["profile"]["summary"] = "x" * 5000
        record = CandidateRecord.model_validate(raw)
        text = build_candidate_text(record)
        assert len(text) <= MAX_CANDIDATE_TEXT_CHARS

    def test_career_swap_inverted_dates(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["career_history"][1]["start_date"] = "2022-01-01"
        raw["career_history"][1]["end_date"]   = "2018-01-01"  # end before start
        record = CandidateRecord.model_validate(raw)
        flat = flatten_candidate(record)
        assert flat["n_career_roles"] == 2  # did not crash

    def test_malformed_json_line_skipped(self, tmp_path):
        path = tmp_path / "malformed.jsonl"
        with open(path, "w") as fh:
            fh.write(json.dumps(MINIMAL_DICT) + "\n")
            fh.write("{ bad json !!!}\n")
            r2 = deepcopy(MINIMAL_DICT)
            r2["candidate_id"] = "CAND_0000002"
            fh.write(json.dumps(r2) + "\n")
        result = list(iter_candidates(path, skip_invalid=True))
        assert len(result) == 2

    def test_string_path_accepted(self, tmp_jsonl):
        # iter_candidates should accept str, not just Path
        result = list(iter_candidates(str(tmp_jsonl)))
        assert len(result) == 3

    def test_years_of_experience_clamped_at_50(self):
        raw = deepcopy(MINIMAL_DICT)
        raw["profile"]["years_of_experience"] = 999.0
        record = CandidateRecord.model_validate(raw)
        assert record.profile.years_of_experience == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# § I  Integration — sample_candidates.json
# ─────────────────────────────────────────────────────────────────────────────

@sample_json_exists
class TestSampleCandidatesJson:

    def test_loads_without_error(self):
        records = load_candidates(SAMPLE_CANDIDATES_FILE, show_progress=False)
        assert len(records) > 0

    def test_all_have_candidate_id(self):
        for rec in iter_candidates(SAMPLE_CANDIDATES_FILE):
            assert rec.candidate_id.startswith("CAND_")

    def test_load_flat_shape(self):
        df = load_flat(SAMPLE_CANDIDATES_FILE, show_progress=False)
        assert len(df) > 0
        assert "candidate_text" in df.columns

    def test_candidate_text_non_empty_for_all(self):
        df = load_flat(SAMPLE_CANDIDATES_FILE, show_progress=False)
        assert (df["candidate_text"].str.len() > 0).all()

    def test_first_candidate_is_cand_0000001(self):
        records = list(iter_candidates(SAMPLE_CANDIDATES_FILE))
        assert records[0].candidate_id == "CAND_0000001"

    def test_salary_inverted_column_is_boolean(self):
        df = load_flat(SAMPLE_CANDIDATES_FILE, show_progress=False)
        # salary_inverted is a honeypot detection signal — deliberately present
        # in ~20-30% of records. We verify the column exists and is bool-typed,
        # NOT that it is near-zero (the dataset uses inversions as a feature).
        assert "salary_inverted" in df.columns
        assert df["salary_inverted"].dtype == bool or df["salary_inverted"].isin([True, False]).all()


# ─────────────────────────────────────────────────────────────────────────────
# § J  Integration — candidates.jsonl (real file, first 200 records)
# ─────────────────────────────────────────────────────────────────────────────

@candidates_jsonl_exists
class TestCandidatesJsonl:

    def test_iter_200_records_without_error(self):
        count = sum(1 for _ in iter_candidates(CANDIDATES_FILE, max_records=200))
        assert count == 200

    def test_load_flat_200_correct_shape(self):
        df = load_flat(CANDIDATES_FILE, max_records=200, show_progress=False)
        assert len(df) == 200
        assert df.index.name == "candidate_id"

    def test_all_ids_match_cand_pattern(self):
        import re
        pattern = re.compile(r"^CAND_\d{7}$")
        for rec in iter_candidates(CANDIDATES_FILE, max_records=200):
            assert pattern.match(rec.candidate_id), f"Bad ID: {rec.candidate_id}"

    def test_years_of_experience_in_valid_range(self):
        df = load_flat(CANDIDATES_FILE, max_records=200, show_progress=False)
        assert df["years_of_experience"].between(0, 50).all()

    def test_recruiter_response_rate_in_0_1(self):
        df = load_flat(CANDIDATES_FILE, max_records=200, show_progress=False)
        assert df["recruiter_response_rate"].between(0, 1).all()

    def test_profile_completeness_in_0_100(self):
        df = load_flat(CANDIDATES_FILE, max_records=200, show_progress=False)
        assert df["profile_completeness_score"].between(0, 100).all()

    def test_tier1_count_is_integer(self):
        df = load_flat(CANDIDATES_FILE, max_records=200, show_progress=False)
        assert df["tier1_skill_count"].dtype in (int, "int64", "int32")

    def test_country_column_has_values(self):
        df = load_flat(CANDIDATES_FILE, max_records=200, show_progress=False)
        assert df["country"].notna().all()

    def test_candidate_text_under_max_chars(self):
        from src.config import MAX_CANDIDATE_TEXT_CHARS
        df = load_flat(CANDIDATES_FILE, max_records=200, show_progress=False)
        assert (df["candidate_text"].str.len() <= MAX_CANDIDATE_TEXT_CHARS).all()


# ─────────────────────────────────────────────────────────────────────────────
# § K  Memory benchmark (smoke test)
# ─────────────────────────────────────────────────────────────────────────────

@candidates_jsonl_exists
class TestMemoryBenchmark:

    def test_benchmark_returns_expected_keys(self):
        results = benchmark_memory(CANDIDATES_FILE, n_records=100, show_results=False)
        expected_keys = {
            "n_records", "streaming_peak_mb", "load_flat_peak_mb",
            "flat_df_size_mb", "bytes_per_record",
        }
        assert expected_keys == set(results.keys())

    def test_benchmark_n_records_matches(self):
        results = benchmark_memory(CANDIDATES_FILE, n_records=100, show_results=False)
        assert results["n_records"] == 100

    def test_streaming_peak_lower_than_load_flat(self):
        results = benchmark_memory(CANDIDATES_FILE, n_records=200, show_results=False)
        # Streaming should use significantly less peak memory
        assert results["streaming_peak_mb"] < results["load_flat_peak_mb"]

    def test_bytes_per_record_reasonable(self):
        results = benchmark_memory(CANDIDATES_FILE, n_records=200, show_results=False)
        # Each flattened record should be between 1KB and 500KB in a DataFrame
        assert 1_000 < results["bytes_per_record"] < 500_000
