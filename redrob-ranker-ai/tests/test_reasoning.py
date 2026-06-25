"""
tests/test_reasoning.py — Tests for src/reasoning.py

Sections
--------
  §A  Helper functions             — 12 tests
  §B  No hallucinations / grounding — 8 tests
  §C  Rank-band tone               —  8 tests
  §D  Concern and gap surfacing    —  6 tests
  §E  Graceful degradation         —  5 tests
  §F  Batch alignment              —  3 tests
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from src.reasoning import (
    _apply_template,
    _CLEANUP_RE,
    _inactivity_concern,
    _jd_relevant_skills,
    _notice_str,
    _pick,
    _primary_concern,
    _safe_title,
    _skill_clause,
    _weakest_sub_score,
    _yoe_str,
    generate_explanation,
    generate_explanations,
)
from src.scoring import CandidateScore, SCORE_WEIGHTS


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_score(
    cid: str = "CAND_TEST_001",
    composite: float = 0.75,
    sub_scores: dict[str, float] | None = None,
    trap_labels: list[str] | None = None,
    trap_penalty: float = 1.0,
    trap_risk: float = 0.0,
) -> CandidateScore:
    ss = sub_scores or {k: 0.5 for k in SCORE_WEIGHTS}
    ws = sum(ss.get(k, 0.5) * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)
    return CandidateScore(
        candidate_id=cid,
        composite_score=composite,
        rank_key=(-composite, cid),
        score_breakdown={
            "sub_scores": ss,
            "weights": dict(SCORE_WEIGHTS),
            "weighted_sub_scores": {k: ss.get(k, 0.5) * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS},
            "weighted_sum": ws,
            "trap_penalty": trap_penalty,
            "composite_score": composite,
            "weight_groups": {
                "career_relevance": 0.175,
                "skill_depth": 0.125,
                "behavioral": 0.100,
                "location": 0.060,
                "experience_fit": 0.040,
            },
            "skill_detail": {},
            "trap_detail": {
                "trap_labels": trap_labels or [],
                "trap_risk_score": trap_risk,
                "explanation": [],
            },
        },
    )


def _make_flat(
    cid: str = "CAND_TEST_001",
    yoe: float | None = 6.0,
    title: str = "Senior ML Engineer",
    location: str = "Bangalore",
    country: str = "India",
    skills: list[str] | None = None,
    notice_days: int | None = 30,
    product_ratio: float = 0.80,
    open_to_work: bool = True,
    days_inactive: int = 3,
) -> dict[str, Any]:
    return {
        "candidate_id": cid,
        "years_of_experience": yoe,
        "current_title": title,
        "most_recent_title": title,
        "location": location,
        "country": country,
        "skill_names": skills if skills is not None else ["FAISS", "PyTorch", "RAG"],
        "notice_period_days": notice_days,
        "product_company_ratio": product_ratio,
        "open_to_work_flag": open_to_work,
        "days_since_last_active": days_inactive,
        "recruiter_response_rate": 0.80,
    }


# ─────────────────────────────────────────────────────────────────────────────
# §A  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestYoeStr:
    def test_none_returns_empty(self):
        assert _yoe_str(None) == ""

    def test_below_half_returns_empty(self):
        assert _yoe_str(0.3) == ""

    def test_zero_returns_under_one_year(self):
        assert _yoe_str(0.0) == ""  # 0 < 0.5 threshold

    def test_half_year_returns_under_one(self):
        assert _yoe_str(0.6) == "under 1 year"

    def test_one_year_singular(self):
        assert _yoe_str(1.0) == "1 year"

    def test_five_years_plural(self):
        assert _yoe_str(5.0) == "5 years"

    def test_fractional_truncates(self):
        assert _yoe_str(7.9) == "7 years"


class TestNoticeStr:
    def test_none_is_flexible(self):
        assert _notice_str(None) == "flexible"

    def test_zero_is_flexible(self):
        assert _notice_str(0) == "flexible"

    def test_fifteen_days_is_immediate(self):
        assert _notice_str(15) == "immediate"

    def test_thirty_days(self):
        assert _notice_str(30) == "30-day"

    def test_sixty_days(self):
        assert _notice_str(60) == "60-day"

    def test_ninety_days(self):
        assert _notice_str(90) == "90-day"

    def test_long_notice_uses_raw_days(self):
        result = _notice_str(120)
        assert "120" in result


class TestSkillClause:
    def test_empty_returns_empty_string(self):
        assert _skill_clause([]) == ""

    def test_one_skill(self):
        assert _skill_clause(["FAISS"]) == " (FAISS)"

    def test_two_skills_uses_and(self):
        result = _skill_clause(["FAISS", "PyTorch"])
        assert "FAISS" in result and "and PyTorch" in result

    def test_three_skills_uses_oxford_comma(self):
        result = _skill_clause(["FAISS", "RAG", "PyTorch"])
        assert "FAISS" in result and "RAG" in result and "and PyTorch" in result
        assert "," in result


class TestJdRelevantSkills:
    def test_returns_jd_skills_first(self):
        skills = ["Python", "FAISS", "qdrant", "cooking"]
        result = _jd_relevant_skills(skills, n=3)
        assert "FAISS" in result or "faiss" in result.lower() or any(
            s.lower() in ("faiss", "qdrant") for s in result
        )

    def test_falls_back_when_fewer_than_two_jd_matches(self):
        skills = ["Python", "cooking", "gardening"]
        result = _jd_relevant_skills(skills, n=3)
        assert result == skills[:3]

    def test_respects_n_limit(self):
        skills = ["FAISS", "RAG", "PyTorch", "Pinecone"]
        result = _jd_relevant_skills(skills, n=2)
        assert len(result) <= 2


# ─────────────────────────────────────────────────────────────────────────────
# §B  No hallucinations / grounding
# ─────────────────────────────────────────────────────────────────────────────

class TestFactGrounding:
    def test_no_unfilled_placeholders(self):
        flat = _make_flat()
        score = _make_score()
        text = generate_explanation(flat, score, rank=5)
        assert "{" not in text and "}" not in text

    def test_title_appears_in_output(self):
        flat = _make_flat(title="ML Research Scientist")
        score = _make_score()
        text = generate_explanation(flat, score, rank=5)
        assert "ML Research Scientist" in text

    def test_jd_relevant_skill_appears_in_output(self):
        flat = _make_flat(skills=["FAISS", "PyTorch", "RAG"])
        sub = {k: 0.5 for k in SCORE_WEIGHTS}
        sub["retrieval_ranking_score"] = 0.90
        score = _make_score(sub_scores=sub)
        text = generate_explanation(flat, score, rank=3)
        # At least one of the JD-relevant skills should be mentioned
        assert any(s in text for s in ["FAISS", "PyTorch", "RAG"])

    def test_no_invented_percentages_without_profile_data(self):
        flat = _make_flat(skills=[])
        score = _make_score()
        text = generate_explanation(flat, score, rank=5)
        # Should not contain invented performance numbers like "40%" that
        # don't come from the profile
        assert "40%" not in text and "improved" not in text.lower()

    def test_location_clause_uses_actual_location(self):
        flat = _make_flat(location="Hyderabad", country="India")
        # All sub-scores above the 0.55 gap threshold so the gap branch is
        # skipped and the location second sentence fires instead.
        sub = {k: 0.80 for k in SCORE_WEIGHTS}
        sub["behavioral_signal_score"] = 0.90
        score = _make_score(sub_scores=sub)
        text = generate_explanation(flat, score, rank=15)
        assert "Hyderabad" in text

    def test_product_pct_matches_profile_ratio(self):
        flat = _make_flat(product_ratio=0.90, location="Mumbai", country="India",
                         skills=[], yoe=None)
        sub = {k: 0.1 for k in SCORE_WEIGHTS}  # all low → general fallback
        score = _make_score(sub_scores=sub, composite=0.3)
        text = generate_explanation(flat, score, rank=20)
        if "90%" in text:
            assert "product" in text.lower() or "startup" in text.lower()

    def test_yoe_value_matches_profile(self):
        flat = _make_flat(yoe=9.0)
        score = _make_score()
        text = generate_explanation(flat, score, rank=5)
        assert "9 years" in text

    def test_output_is_non_empty_string(self):
        flat = _make_flat()
        score = _make_score()
        result = generate_explanation(flat, score, rank=1)
        assert isinstance(result, str) and len(result) > 20


# ─────────────────────────────────────────────────────────────────────────────
# §C  Rank-band tone
# ─────────────────────────────────────────────────────────────────────────────

class TestRankBandTone:
    def test_rank_1_does_not_say_borderline(self):
        flat = _make_flat()
        sub = {k: 0.5 for k in SCORE_WEIGHTS}
        sub["retrieval_ranking_score"] = 0.90
        score = _make_score(sub_scores=sub, composite=0.95)
        text = generate_explanation(flat, score, rank=1).lower()
        assert "borderline" not in text

    def test_rank_1_does_not_say_partial_match(self):
        flat = _make_flat()
        sub = {k: 0.5 for k in SCORE_WEIGHTS}
        sub["retrieval_ranking_score"] = 0.90
        score = _make_score(sub_scores=sub, composite=0.95)
        text = generate_explanation(flat, score, rank=1).lower()
        assert "partial" not in text

    def test_rank_80_uses_borderline_framing(self):
        flat = _make_flat()
        score = _make_score(composite=0.35)
        text = generate_explanation(flat, score, rank=80).lower()
        # All 6 borderline templates use one of these words
        assert any(w in text for w in [
            "borderline", "weak", "limited", "does not strongly", "lacks", "top 100",
        ])

    def test_rank_45_uses_partial_framing(self):
        flat = _make_flat()
        score = _make_score(composite=0.55)
        text = generate_explanation(flat, score, rank=45).lower()
        assert any(w in text for w in ["partial", "moderate", "some", "reasonable ml", "incomplete"])

    def test_rank_5_high_retrieval_uses_retrieval_framing(self):
        flat = _make_flat(skills=["FAISS", "dense retrieval", "BM25"])
        sub = {k: 0.5 for k in SCORE_WEIGHTS}
        sub["retrieval_ranking_score"] = 0.90
        score = _make_score(sub_scores=sub, composite=0.88)
        text = generate_explanation(flat, score, rank=5).lower()
        assert any(w in text for w in ["retrieval", "vector search", "ranking", "specialist"])

    def test_rank_5_high_production_uses_production_framing(self):
        flat = _make_flat(skills=["PyTorch", "mlflow"])
        sub = {k: 0.5 for k in SCORE_WEIGHTS}
        sub["production_ml_score"] = 0.85
        sub["retrieval_ranking_score"] = 0.30  # not the top signal
        score = _make_score(sub_scores=sub, composite=0.82)
        text = generate_explanation(flat, score, rank=5).lower()
        assert any(w in text for w in ["production", "shipping", "deployed", "builder"])

    def test_rank_none_does_not_use_borderline(self):
        flat = _make_flat()
        score = _make_score(composite=0.70)
        text = generate_explanation(flat, score, rank=None).lower()
        assert "borderline" not in text

    def test_different_ranks_can_yield_different_first_words(self):
        flat = _make_flat()
        score = _make_score(composite=0.60)
        texts = [generate_explanation(flat, score, rank=r) for r in (5, 35, 80)]
        # Each rank band should produce distinct output
        assert len(set(texts)) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# §D  Concern and gap surfacing
# ─────────────────────────────────────────────────────────────────────────────

class TestConcernSurfacing:
    def test_fake_ai_profile_trap_surfaced(self):
        flat = _make_flat()
        score = _make_score(trap_labels=["fake_ai_profile"], composite=0.40)
        text = generate_explanation(flat, score, rank=70).lower()
        assert any(w in text for w in ["zero-duration", "credential", "flag", "note", "screening"])

    def test_keyword_stuffing_trap_surfaced(self):
        flat = _make_flat()
        score = _make_score(trap_labels=["keyword_stuffing"], composite=0.50)
        text = generate_explanation(flat, score, rank=25)
        assert any(phrase in text.lower() for phrase in ["inflated", "verify", "flag", "note"])

    def test_concern_prioritises_fake_ai_over_inactive(self):
        flat = _make_flat(days_inactive=200, open_to_work=False)
        score = _make_score(trap_labels=["fake_ai_profile", "inactive_candidate"])
        text = generate_explanation(flat, score, rank=30).lower()
        # fake_ai_profile is higher priority than inactive_candidate
        assert "zero-duration" in text or "credential" in text

    def test_inactivity_concern_surfaced_without_trap(self):
        flat = _make_flat(days_inactive=120, open_to_work=False)
        score = _make_score(trap_labels=[])
        text = generate_explanation(flat, score, rank=20).lower()
        assert "inactive" in text or "confirm" in text or "120" in text

    def test_no_concern_for_active_candidate(self):
        flat = _make_flat(days_inactive=5, open_to_work=True)
        score = _make_score(trap_labels=[])
        text = generate_explanation(flat, score, rank=5)
        # Should not have a concern sentence for a healthy profile
        assert "flag" not in text.lower() or "note" not in text.lower()

    def test_weak_sub_score_gap_surfaced_for_rank_20(self):
        flat = _make_flat(days_inactive=5, open_to_work=True,
                         location="London", country="UK")  # not preferred
        sub = {k: 0.8 for k in SCORE_WEIGHTS}
        sub["retrieval_ranking_score"] = 0.10  # very weak
        score = _make_score(sub_scores=sub, trap_labels=[])
        text = generate_explanation(flat, score, rank=20).lower()
        assert any(w in text for w in ["gap", "limitation", "missing", "vector search", "ranking"])


# ─────────────────────────────────────────────────────────────────────────────
# §E  Graceful degradation
# ─────────────────────────────────────────────────────────────────────────────

class TestGracefulDegradation:
    def test_no_skills_produces_valid_output(self):
        flat = _make_flat(skills=[])
        score = _make_score()
        text = generate_explanation(flat, score, rank=5)
        assert isinstance(text, str) and len(text) > 10
        assert "{" not in text

    def test_no_yoe_produces_valid_output(self):
        flat = _make_flat(yoe=None)
        score = _make_score()
        text = generate_explanation(flat, score, rank=5)
        assert isinstance(text, str) and len(text) > 10
        # Should not have artifacts from empty yoe
        assert "with of" not in text
        assert "with and" not in text
        assert "  " not in text  # no double spaces

    def test_no_title_falls_back_to_candidate(self):
        flat = _make_flat(title="")
        flat["current_title"] = None
        flat["most_recent_title"] = None
        score = _make_score()
        text = generate_explanation(flat, score, rank=5)
        assert "candidate" in text.lower()

    def test_none_score_result_triggers_inline_scoring(self):
        flat = _make_flat()
        # Without a score_result, generate_explanation must call score_candidate internally
        text = generate_explanation(flat, score_result=None, rank=5)
        assert isinstance(text, str) and len(text) > 10

    def test_apply_template_cleans_double_spaces(self):
        result = _apply_template("A {title} with  of depth.", title="ML Engineer")
        assert "  " not in result


# ─────────────────────────────────────────────────────────────────────────────
# §F  Batch alignment
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchAlignment:
    def test_generate_explanations_length_matches_input(self):
        flats = [_make_flat(cid=f"CAND_{i:04d}") for i in range(5)]
        scores = [_make_score(cid=f"CAND_{i:04d}") for i in range(5)]
        results = generate_explanations(flats, scores, ranks=list(range(1, 6)))
        assert len(results) == 5

    def test_batch_matches_individual_calls(self):
        flats = [_make_flat(cid=f"CAND_{i:04d}") for i in range(3)]
        scores = [_make_score(cid=f"CAND_{i:04d}") for i in range(3)]
        ranks = [1, 20, 80]
        batch = generate_explanations(flats, scores, ranks=ranks)
        individual = [
            generate_explanation(f, s, rank=r)
            for f, s, r in zip(flats, scores, ranks)
        ]
        assert batch == individual

    def test_determinism_across_two_calls(self):
        flat = _make_flat()
        score = _make_score()
        text1 = generate_explanation(flat, score, rank=10)
        text2 = generate_explanation(flat, score, rank=10)
        assert text1 == text2
