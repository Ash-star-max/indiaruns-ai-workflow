"""
tests/test_validate_submission.py — Tests for src/validate_submission.py

Sections
--------
  §A  Individual check functions   — 26 tests (schema, row_count, ranks,
                                     unique_ids, score_range, monotonic, reasoning)
  §B  validate_dataframe           — 10 tests
  §C  validate_file                —  8 tests (including UTF-8)
  §D  ValidationReport properties  —  5 tests
  §E  print_report                 —  3 tests
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validate_submission import (
    EXPECTED_ROW_COUNT,
    CheckResult,
    ValidationReport,
    check_monotonic_scores,
    check_ranks,
    check_reasoning,
    check_row_count,
    check_schema,
    check_score_range,
    check_unique_ids,
    check_utf8,
    print_report,
    validate_dataframe,
    validate_file,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _good_df(n: int = EXPECTED_ROW_COUNT) -> pd.DataFrame:
    """Build a fully valid submission DataFrame with n rows."""
    scores = [round(1.0 - i * (0.9 / max(n - 1, 1)), 6) for i in range(n)]
    return pd.DataFrame({
        "candidate_id": [f"CAND_{i:07d}" for i in range(1, n + 1)],
        "rank":         list(range(1, n + 1)),
        "score":        scores,
        "reasoning":    [f"Explanation for candidate {i}." for i in range(1, n + 1)],
    })


def _write_csv(tmp_path: Path, df: pd.DataFrame, name: str = "sub.csv") -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False, encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# §A  Individual check functions
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSchema:
    def test_good_df_passes(self):
        assert check_schema(_good_df()).passed

    def test_missing_one_column_fails(self):
        df = _good_df().drop(columns=["reasoning"])
        r  = check_schema(df)
        assert not r.passed
        assert "reasoning" in r.message

    def test_missing_all_columns_fails(self):
        r = check_schema(pd.DataFrame())
        assert not r.passed

    def test_non_integer_rank_fails(self):
        df = _good_df()
        df["rank"] = df["rank"].astype(str).str.replace("1", "one")
        r = check_schema(df)
        assert not r.passed

    def test_non_numeric_score_fails(self):
        df = _good_df()
        df["score"] = "not-a-number"
        r = check_schema(df)
        assert not r.passed

    def test_extra_columns_are_tolerated(self):
        df = _good_df()
        df["extra_col"] = "ignored"
        r = check_schema(df)
        assert r.passed
        assert any("Extra columns" in d for d in r.details)


class TestCheckRowCount:
    def test_exactly_100_passes(self):
        assert check_row_count(_good_df(100)).passed

    def test_99_rows_fails(self):
        r = check_row_count(_good_df(99))
        assert not r.passed
        assert "99" in r.message

    def test_101_rows_fails(self):
        r = check_row_count(_good_df(101))
        assert not r.passed
        assert "101" in r.message

    def test_empty_df_fails(self):
        assert not check_row_count(pd.DataFrame()).passed


class TestCheckRanks:
    def test_valid_ranks_pass(self):
        assert check_ranks(_good_df()).passed

    def test_gap_in_ranks_fails(self):
        df = _good_df()
        df.loc[50, "rank"] = 200
        r = check_ranks(df)
        assert not r.passed

    def test_duplicate_rank_fails(self):
        df = _good_df()
        df.loc[1, "rank"] = df.loc[0, "rank"]
        assert not check_ranks(df).passed

    def test_zero_based_ranks_fail(self):
        df = _good_df()
        df["rank"] = df["rank"] - 1   # 0–99 instead of 1–100
        assert not check_ranks(df).passed

    def test_missing_rank_column_fails(self):
        assert not check_ranks(pd.DataFrame({"score": [0.5]})).passed


class TestCheckUniqueIds:
    def test_unique_ids_pass(self):
        assert check_unique_ids(_good_df()).passed

    def test_duplicate_id_fails(self):
        df = _good_df()
        df.loc[1, "candidate_id"] = df.loc[0, "candidate_id"]
        r = check_unique_ids(df)
        assert not r.passed
        assert "duplicate" in r.message.lower()

    def test_missing_id_column_fails(self):
        assert not check_unique_ids(pd.DataFrame({"rank": [1]})).passed


class TestCheckScoreRange:
    def test_valid_scores_pass(self):
        assert check_score_range(_good_df()).passed

    def test_score_above_one_fails(self):
        df = _good_df()
        df.loc[0, "score"] = 1.5
        r = check_score_range(df)
        assert not r.passed

    def test_negative_score_fails(self):
        df = _good_df()
        df.loc[0, "score"] = -0.1
        assert not check_score_range(df).passed

    def test_boundary_zero_passes(self):
        df = _good_df()
        df["score"] = 0.0
        assert check_score_range(df).passed

    def test_boundary_one_passes(self):
        df = _good_df()
        df["score"] = 1.0
        assert check_score_range(df).passed

    def test_non_numeric_fails(self):
        df = _good_df()
        df["score"] = "bad"
        assert not check_score_range(df).passed


class TestCheckMonotonicScores:
    def test_strictly_decreasing_passes(self):
        assert check_monotonic_scores(_good_df()).passed

    def test_equal_adjacent_scores_pass(self):
        df = _good_df()
        df.loc[1, "score"] = df.loc[0, "score"]   # tie — allowed
        assert check_monotonic_scores(df).passed

    def test_increasing_score_fails(self):
        df = _good_df()
        # Lower rank-1 score so rank-2 (default ~0.991) becomes higher → inversion
        df.loc[0, "score"] = 0.5
        r = check_monotonic_scores(df)
        assert not r.passed
        assert len(r.details) > 0

    def test_tolerance_allows_tiny_float_noise(self):
        df = _good_df()
        # Add sub-epsilon noise: should still pass
        df.loc[1, "score"] = df.loc[0, "score"] + 1e-12
        assert check_monotonic_scores(df).passed

    def test_missing_rank_column_fails(self):
        df = _good_df().drop(columns=["rank"])
        assert not check_monotonic_scores(df).passed


class TestCheckReasoning:
    def test_all_non_empty_passes(self):
        assert check_reasoning(_good_df()).passed

    def test_empty_string_fails(self):
        df = _good_df()
        df.loc[0, "reasoning"] = ""
        r = check_reasoning(df)
        assert not r.passed

    def test_whitespace_only_fails(self):
        df = _good_df()
        df.loc[0, "reasoning"] = "   "
        r = check_reasoning(df)
        assert not r.passed

    def test_null_fails(self):
        df = _good_df()
        df.loc[0, "reasoning"] = None
        r = check_reasoning(df)
        assert not r.passed

    def test_missing_column_fails(self):
        assert not check_reasoning(pd.DataFrame({"rank": [1]})).passed


# ─────────────────────────────────────────────────────────────────────────────
# §B  validate_dataframe
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateDataframe:
    def test_good_df_all_pass(self):
        report = validate_dataframe(_good_df())
        assert report.passed
        assert report.n_failed == 0

    def test_failed_schema_stops_most_checks(self):
        df     = _good_df().drop(columns=["score"])
        report = validate_dataframe(df)
        assert not report.passed
        # Only schema + row_count run when schema fails
        assert len(report.checks) <= 3

    def test_wrong_row_count_fails(self):
        report = validate_dataframe(_good_df(50))
        assert not report.passed
        assert any(c.name == "row_count" and not c.passed for c in report.checks)

    def test_duplicate_id_fails(self):
        df = _good_df()
        df.loc[5, "candidate_id"] = df.loc[0, "candidate_id"]
        report = validate_dataframe(df)
        assert not report.passed
        assert any(c.name == "unique_ids" and not c.passed for c in report.checks)

    def test_non_monotonic_score_fails(self):
        df = _good_df()
        # Lower rank-1 score so rank-2 (~0.991 default) becomes higher → inversion
        df.loc[0, "score"] = 0.5
        report = validate_dataframe(df)
        assert not report.passed
        assert any(c.name == "monotonic_scores" and not c.passed for c in report.checks)

    def test_empty_reasoning_fails(self):
        df = _good_df()
        df.loc[0, "reasoning"] = ""
        report = validate_dataframe(df)
        assert not report.passed

    def test_all_check_names_are_unique(self):
        report = validate_dataframe(_good_df())
        names  = [c.name for c in report.checks]
        assert len(names) == len(set(names))

    def test_each_check_has_non_empty_message(self):
        report = validate_dataframe(_good_df())
        for chk in report.checks:
            assert chk.message, f"check '{chk.name}' has empty message"

    def test_score_out_of_range_fails(self):
        df = _good_df()
        df.loc[0, "score"] = 2.0
        report = validate_dataframe(df)
        assert not report.passed
        assert any(c.name == "score_range" and not c.passed for c in report.checks)

    def test_partial_failure_reports_n_failed(self):
        df = _good_df()
        df.loc[0, "reasoning"] = ""
        df.loc[1, "score"] = 2.0
        report = validate_dataframe(df)
        assert report.n_failed >= 2


# ─────────────────────────────────────────────────────────────────────────────
# §C  validate_file
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateFile:
    def test_good_csv_passes(self, tmp_path):
        p = _write_csv(tmp_path, _good_df())
        assert validate_file(p).passed

    def test_report_has_path(self, tmp_path):
        p      = _write_csv(tmp_path, _good_df())
        report = validate_file(p)
        assert report.path == p

    def test_missing_file_fails_with_utf8_check(self, tmp_path):
        p      = tmp_path / "nonexistent.csv"
        report = validate_file(p)
        assert not report.passed
        assert any(c.name == "utf8_encoding" and not c.passed for c in report.checks)

    def test_non_utf8_file_fails(self, tmp_path):
        p = tmp_path / "bad.csv"
        # Write Latin-1 content that is invalid UTF-8
        p.write_bytes("candidate_id,rank,score,reasoning\ncafé,1,0.9,test\n".encode("latin-1"))
        report = validate_file(p)
        assert not report.passed
        assert any(c.name == "utf8_encoding" and not c.passed for c in report.checks)

    def test_valid_utf8_with_unicode_content_passes_encoding_check(self, tmp_path):
        df = _good_df()
        df.loc[0, "reasoning"] = "Candidate excels at ML — सशक्त इंजीनियर।"
        p  = _write_csv(tmp_path, df)
        report = validate_file(p)
        utf8_check = next(c for c in report.checks if c.name == "utf8_encoding")
        assert utf8_check.passed

    def test_real_submission_csv_passes(self):
        sub_path = Path("outputs/submission.csv")
        if not sub_path.exists():
            pytest.skip("outputs/submission.csv not found — run run.py first")
        report = validate_file(sub_path)
        assert report.passed, [c for c in report.failed_checks()]

    def test_wrong_row_count_in_file_fails(self, tmp_path):
        p = _write_csv(tmp_path, _good_df(50))
        assert not validate_file(p).passed

    def test_accepts_path_as_string(self, tmp_path):
        p = _write_csv(tmp_path, _good_df())
        assert validate_file(str(p)).passed

    def test_duplicate_ids_in_file_fails(self, tmp_path):
        df = _good_df()
        df.loc[5, "candidate_id"] = df.loc[0, "candidate_id"]
        p  = _write_csv(tmp_path, df)
        assert not validate_file(p).passed


# ─────────────────────────────────────────────────────────────────────────────
# §D  ValidationReport properties
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationReport:
    def test_passed_when_all_checks_pass(self):
        report = ValidationReport(checks=[
            CheckResult("a", True,  "ok"),
            CheckResult("b", True,  "ok"),
        ])
        assert report.passed

    def test_not_passed_when_any_check_fails(self):
        report = ValidationReport(checks=[
            CheckResult("a", True,  "ok"),
            CheckResult("b", False, "fail"),
        ])
        assert not report.passed

    def test_n_passed_counts_correctly(self):
        report = ValidationReport(checks=[
            CheckResult("a", True,  "ok"),
            CheckResult("b", False, "fail"),
            CheckResult("c", True,  "ok"),
        ])
        assert report.n_passed == 2
        assert report.n_failed == 1

    def test_empty_report_is_not_passed(self):
        assert not ValidationReport().passed

    def test_failed_checks_filters_correctly(self):
        report = ValidationReport(checks=[
            CheckResult("a", True,  "ok"),
            CheckResult("b", False, "fail"),
        ])
        failed = report.failed_checks()
        assert len(failed) == 1
        assert failed[0].name == "b"


# ─────────────────────────────────────────────────────────────────────────────
# §E  print_report
# ─────────────────────────────────────────────────────────────────────────────

class TestPrintReport:
    def _capture(self, report: ValidationReport) -> str:
        buf = io.StringIO()
        print_report(report, file=buf)
        return buf.getvalue()

    def test_output_contains_pass_for_valid(self):
        report = validate_dataframe(_good_df())
        out    = self._capture(report)
        assert "VALID" in out or "PASS" in out

    def test_output_contains_fail_for_invalid(self):
        df     = _good_df()
        df.loc[0, "reasoning"] = ""
        report = validate_dataframe(df)
        out    = self._capture(report)
        assert "FAIL" in out or "INVALID" in out

    def test_each_check_name_appears_in_output(self):
        report = validate_dataframe(_good_df())
        out    = self._capture(report)
        for chk in report.checks:
            # name appears in title-case form
            assert chk.name.replace("_", " ").split()[0].capitalize() in out or chk.name in out
