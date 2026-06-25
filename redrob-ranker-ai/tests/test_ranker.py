"""
tests/test_ranker.py — Tests for src/ranker.py

Sections
--------
  §A  _reconstruct_candidate_score   — 6 tests
  §B  _meta_to_flat                  — 5 tests
  §C  validate_submission            — 8 tests
  §D  select_top_100 integration     — 8 tests (uses in-memory mock artifacts)
"""

from __future__ import annotations

import json
import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.ranker import (
    _meta_to_flat,
    _reconstruct_candidate_score,
    select_top_100,
    validate_submission,
)
from src.scoring import SCORE_WEIGHTS, CandidateScore


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build minimal sub_scores and meta DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def _make_sub_scores_row(
    cid: str,
    composite: float = 0.5,
    trap_penalty: float = 1.0,
    trap_labels: list[str] | None = None,
) -> dict[str, Any]:
    sub = {k: 0.5 for k in SCORE_WEIGHTS}
    weighted_sum = sum(sub[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)
    return {
        "candidate_id":           cid,
        "composite_score":        composite,
        "weighted_sum":           weighted_sum,
        "trap_penalty":           trap_penalty,
        "trap_risk_score":        0.0,
        "trap_labels":            json.dumps(trap_labels or []),
        "group_career_relevance": 0.175,
        "group_skill_depth":      0.125,
        "group_behavioral":       0.100,
        "group_location":         0.060,
        "group_experience_fit":   0.040,
        **sub,
    }


def _make_sub_scores_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.set_index("candidate_id")
    return df


def _make_meta_df(cids: list[str]) -> pd.DataFrame:
    rows = [
        {
            "candidate_id":          cid,
            "years_of_experience":   5.0,
            "current_title":         "ML Engineer",
            "most_recent_title":     "ML Engineer",
            "country":               "India",
            "location":              "Bangalore",
            "notice_period_days":    30,
            "product_company_ratio": 0.8,
            "open_to_work_flag":     True,
            "days_since_last_active": 5,
            "recruiter_response_rate": 0.80,
            "skill_names":           json.dumps(["Python", "PyTorch", "FAISS"]),
        }
        for cid in cids
    ]
    df = pd.DataFrame(rows).set_index("candidate_id")
    return df


def _make_artifacts(n: int = 5, scores: list[float] | None = None) -> dict[str, Any]:
    cids = [f"cand-{i:03d}" for i in range(n)]
    score_vals = scores if scores else [1.0 - i * 0.1 for i in range(n)]

    sub_rows = [
        _make_sub_scores_row(cid, composite=score_vals[i])
        for i, cid in enumerate(cids)
    ]
    return {
        "sub_scores":    _make_sub_scores_df(sub_rows),
        "meta":          _make_meta_df(cids),
        "features":      pd.DataFrame(index=pd.Index(cids, name="candidate_id")),
        "candidate_ids": np.array(cids),
        "honeypot_flags": np.zeros(n, dtype=bool),
        "embeddings":    np.zeros((n, 384), dtype=np.float32),
        "jd_embedding":  np.zeros(384, dtype=np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# §A  _reconstruct_candidate_score
# ─────────────────────────────────────────────────────────────────────────────

class TestReconstructCandidateScore:
    def test_returns_candidate_score_instance(self):
        row_data = _make_sub_scores_row("x-001", composite=0.72)
        row = pd.Series(row_data)
        cs = _reconstruct_candidate_score("x-001", row)
        assert isinstance(cs, CandidateScore)

    def test_candidate_id_set_correctly(self):
        row = pd.Series(_make_sub_scores_row("abc-123", composite=0.5))
        cs = _reconstruct_candidate_score("abc-123", row)
        assert cs.candidate_id == "abc-123"

    def test_composite_score_matches(self):
        row = pd.Series(_make_sub_scores_row("x", composite=0.83))
        cs = _reconstruct_candidate_score("x", row)
        assert cs.composite_score == pytest.approx(0.83)

    def test_rank_key_has_negative_score(self):
        row = pd.Series(_make_sub_scores_row("x", composite=0.75))
        cs = _reconstruct_candidate_score("x", row)
        assert cs.rank_key[0] == pytest.approx(-0.75)

    def test_trap_labels_decoded_from_json(self):
        row_data = _make_sub_scores_row("x", trap_labels=["fake_ai_profile"])
        row = pd.Series(row_data)
        cs = _reconstruct_candidate_score("x", row)
        assert cs.score_breakdown["trap_detail"]["trap_labels"] == ["fake_ai_profile"]

    def test_sub_scores_have_all_10_keys(self):
        row = pd.Series(_make_sub_scores_row("x"))
        cs = _reconstruct_candidate_score("x", row)
        assert set(cs.score_breakdown["sub_scores"].keys()) == set(SCORE_WEIGHTS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# §B  _meta_to_flat
# ─────────────────────────────────────────────────────────────────────────────

class TestMetaToFlat:
    def _make_row(self, cid: str = "c-001") -> pd.Series:
        return pd.Series({
            "years_of_experience": 6.0,
            "current_title": "ML Engineer",
            "most_recent_title": "ML Engineer",
            "country": "India",
            "location": "Bangalore",
            "notice_period_days": 30,
            "product_company_ratio": 0.9,
            "open_to_work_flag": True,
            "days_since_last_active": 3,
            "recruiter_response_rate": 0.85,
            "skill_names": json.dumps(["Python", "FAISS"]),
        })

    def test_candidate_id_in_flat(self):
        flat = _meta_to_flat("c-001", self._make_row())
        assert flat["candidate_id"] == "c-001"

    def test_skill_names_decoded_as_list(self):
        flat = _meta_to_flat("c-001", self._make_row())
        assert isinstance(flat["skill_names"], list)
        assert "Python" in flat["skill_names"]

    def test_scalar_fields_preserved(self):
        flat = _meta_to_flat("c-001", self._make_row())
        assert flat["years_of_experience"] == pytest.approx(6.0)
        assert flat["country"] == "India"

    def test_empty_skill_names_gives_empty_list(self):
        row = self._make_row()
        row["skill_names"] = "[]"
        flat = _meta_to_flat("c-001", row)
        assert flat["skill_names"] == []

    def test_invalid_json_skill_names_gives_empty_list(self):
        row = self._make_row()
        row["skill_names"] = "not-json"
        flat = _meta_to_flat("c-001", row)
        assert flat["skill_names"] == []


# ─────────────────────────────────────────────────────────────────────────────
# §C  validate_submission
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateSubmission:
    def _good_df(self, n: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            "candidate_id": [f"c-{i}" for i in range(1, n + 1)],
            "rank":         list(range(1, n + 1)),
            "score":        [round(1.0 - i * 0.05, 4) for i in range(n)],
            "reasoning":    ["Some explanation." for _ in range(n)],
        })

    def test_good_df_passes(self):
        validate_submission(self._good_df())   # no exception

    def test_missing_column_raises(self):
        df = self._good_df()
        df = df.drop(columns=["reasoning"])
        with pytest.raises(ValueError, match="missing columns"):
            validate_submission(df)

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_submission(pd.DataFrame(columns=["candidate_id","rank","score","reasoning"]))

    def test_more_than_100_rows_raises(self):
        df = self._good_df(n=101)
        with pytest.raises(ValueError, match="max is 100"):
            validate_submission(df)

    def test_non_consecutive_ranks_raises(self):
        df = self._good_df(n=5)
        df.loc[2, "rank"] = 99
        with pytest.raises(ValueError, match="Ranks must be consecutive"):
            validate_submission(df)

    def test_duplicate_ids_raises(self):
        df = self._good_df(n=5)
        df.loc[0, "candidate_id"] = df.loc[1, "candidate_id"]
        with pytest.raises(ValueError, match="Duplicate"):
            validate_submission(df)

    def test_score_above_one_raises(self):
        df = self._good_df(n=5)
        df.loc[0, "score"] = 1.5
        with pytest.raises(ValueError, match="out of"):
            validate_submission(df)

    def test_empty_reasoning_raises(self):
        df = self._good_df(n=3)
        df.loc[0, "reasoning"] = ""
        with pytest.raises(ValueError, match="reasoning strings are empty"):
            validate_submission(df)


# ─────────────────────────────────────────────────────────────────────────────
# §D  select_top_100 integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectTop100:
    def test_returns_dataframe(self):
        arts = _make_artifacts(5)
        df = select_top_100(arts)
        assert isinstance(df, pd.DataFrame)

    def test_columns_are_correct(self):
        df = select_top_100(_make_artifacts(5))
        assert set(df.columns) == {"candidate_id", "rank", "score", "reasoning"}

    def test_sorted_descending_by_score(self):
        arts = _make_artifacts(5, scores=[0.3, 0.8, 0.5, 0.9, 0.1])
        df = select_top_100(arts)
        assert df["score"].tolist() == sorted(df["score"].tolist(), reverse=True)

    def test_rank_column_starts_at_one(self):
        df = select_top_100(_make_artifacts(3))
        assert df["rank"].tolist() == [1, 2, 3]

    def test_no_more_than_100_rows(self):
        arts = _make_artifacts(150)
        df = select_top_100(arts)
        assert len(df) <= 100

    def test_reasoning_strings_non_empty(self):
        df = select_top_100(_make_artifacts(5))
        assert all(len(str(r)) > 0 for r in df["reasoning"])

    def test_passes_validate_submission(self):
        df = select_top_100(_make_artifacts(10))
        validate_submission(df)   # should not raise

    def test_tie_break_by_candidate_id(self):
        # Two candidates with identical scores → lower ID should rank first
        arts = _make_artifacts(2, scores=[0.5, 0.5])
        arts["sub_scores"].index = pd.Index(["zzz-002", "aaa-001"], name="candidate_id")
        arts["meta"].index       = pd.Index(["zzz-002", "aaa-001"], name="candidate_id")
        arts["candidate_ids"]    = np.array(["zzz-002", "aaa-001"])
        df = select_top_100(arts)
        assert df.iloc[0]["candidate_id"] == "aaa-001"
