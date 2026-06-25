"""
validate_submission.py — Standalone submission validator for the Redrob hackathon.

Checks
------
  1. schema           — correct columns and compatible dtypes
  2. row_count        — exactly 100 rows
  3. ranks            — 1, 2, …, 100 with no gaps or duplicates
  4. unique_ids       — no duplicate candidate_ids
  5. score_range      — all scores in [0.0, 1.0]
  6. monotonic_scores — scores non-increasing with rank
  7. reasoning        — no null or empty reasoning strings
  8. utf8_encoding    — file is valid UTF-8 (validate_file only)

Usage (CLI)
-----------
  python -m src.validate_submission outputs/submission.csv
  python -m src.validate_submission outputs/submission.csv --quiet

Public API
----------
  validate_file(path)       -> ValidationReport   (reads CSV from disk)
  validate_dataframe(df)    -> ValidationReport   (pure DataFrame checks)
  print_report(report)      -> None               (human-readable output)
  CheckResult               — per-check result dataclass
  ValidationReport          — aggregated report with .passed / .n_passed
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# § 1  Constants and data types
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: tuple[str, ...] = ("candidate_id", "rank", "score", "reasoning")
EXPECTED_ROW_COUNT: int = 100
_SCORE_TOL: float = 1e-9   # tolerance for float equality in monotonicity check


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name:    str
    passed:  bool
    message: str
    details: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Aggregated results of all validation checks."""
    checks: list[CheckResult] = field(default_factory=list)
    path:   Path | None       = None

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


# ─────────────────────────────────────────────────────────────────────────────
# § 2  Individual check functions  (all pure — only take df or path)
# ─────────────────────────────────────────────────────────────────────────────

def check_schema(df: pd.DataFrame) -> CheckResult:
    """Column presence and basic dtype compatibility."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra   = [c for c in df.columns      if c not in REQUIRED_COLUMNS]

    if missing:
        return CheckResult(
            name    = "schema",
            passed  = False,
            message = f"Missing columns: {missing}",
            details = [
                f"Expected : {list(REQUIRED_COLUMNS)}",
                f"Got      : {list(df.columns)}",
            ],
        )

    errors: list[str] = []   # actual schema errors (cause failure)
    notes:  list[str] = []   # informational only (do not cause failure)

    try:
        df["rank"].astype(int)
    except (ValueError, TypeError):
        errors.append("'rank' column cannot be cast to int")

    try:
        pd.to_numeric(df["score"], errors="raise")
    except (ValueError, TypeError):
        errors.append("'score' column is not numeric")

    if extra:
        notes.append(f"Extra columns (ignored): {extra}")

    passed = len(errors) == 0
    return CheckResult(
        name    = "schema",
        passed  = passed,
        message = "Schema valid" if passed else "Schema invalid",
        details = errors + notes,
    )


def check_row_count(df: pd.DataFrame) -> CheckResult:
    """Exactly EXPECTED_ROW_COUNT rows required."""
    n      = len(df)
    passed = n == EXPECTED_ROW_COUNT
    return CheckResult(
        name    = "row_count",
        passed  = passed,
        message = f"{n} rows (expected {EXPECTED_ROW_COUNT})",
        details = [] if passed else [
            f"Got {n} rows; submission must have exactly {EXPECTED_ROW_COUNT}"
        ],
    )


def check_ranks(df: pd.DataFrame) -> CheckResult:
    """Ranks must be exactly 1, 2, …, 100 with no gaps or duplicates."""
    if "rank" not in df.columns:
        return CheckResult(
            name="ranks", passed=False, message="'rank' column missing"
        )

    try:
        ranks    = sorted(df["rank"].astype(int).tolist())
    except (ValueError, TypeError):
        return CheckResult(
            name="ranks", passed=False, message="'rank' column contains non-integer values"
        )

    expected = list(range(1, EXPECTED_ROW_COUNT + 1))
    passed   = ranks == expected

    details: list[str] = []
    if not passed:
        details = [
            f"Expected : 1 … {EXPECTED_ROW_COUNT}",
            f"Got      : {ranks[:8]}{'…' if len(ranks) > 8 else ''}",
            f"N={len(ranks)}  min={min(ranks)}  max={max(ranks)}",
        ]
        if len(set(ranks)) < len(ranks):
            dupes = [r for r in set(ranks) if ranks.count(r) > 1]
            details.append(f"Duplicate rank values: {dupes[:5]}")

    return CheckResult(
        name    = "ranks",
        passed  = passed,
        message = "Ranks 1-100 valid" if passed else "Rank sequence invalid",
        details = details,
    )


def check_unique_ids(df: pd.DataFrame) -> CheckResult:
    """candidate_id values must all be unique."""
    if "candidate_id" not in df.columns:
        return CheckResult(
            name="unique_ids", passed=False, message="'candidate_id' column missing"
        )

    dupes  = df[df["candidate_id"].duplicated(keep=False)]["candidate_id"].unique().tolist()
    passed = len(dupes) == 0
    return CheckResult(
        name    = "unique_ids",
        passed  = passed,
        message = "All IDs unique" if passed else f"{len(dupes)} duplicate ID(s)",
        details = [] if passed else [f"Duplicated: {dupes[:10]}"],
    )


def check_score_range(df: pd.DataFrame) -> CheckResult:
    """All scores must lie in [0.0, 1.0]."""
    if "score" not in df.columns:
        return CheckResult(
            name="score_range", passed=False, message="'score' column missing"
        )

    try:
        scores = pd.to_numeric(df["score"], errors="raise")
    except (ValueError, TypeError):
        return CheckResult(
            name="score_range", passed=False, message="'score' column is not numeric"
        )

    bad_mask = ~scores.between(0.0, 1.0)
    passed   = not bad_mask.any()

    details: list[str] = []
    if not passed:
        bad_ids = (
            df.loc[bad_mask, "candidate_id"].tolist()
            if "candidate_id" in df.columns else []
        )
        details = [
            f"Range in file: [{scores.min():.6f}, {scores.max():.6f}]",
            f"Offending IDs : {bad_ids[:5]}",
        ]

    return CheckResult(
        name    = "score_range",
        passed  = passed,
        message = "All scores in [0, 1]" if passed else f"{bad_mask.sum()} score(s) out of [0, 1]",
        details = details,
    )


def check_monotonic_scores(df: pd.DataFrame) -> CheckResult:
    """
    Scores must be non-increasing with rank.
    score[rank=1] ≥ score[rank=2] ≥ … ≥ score[rank=100].
    A small tolerance (_SCORE_TOL) allows for float representation noise.
    """
    if not {"rank", "score"}.issubset(df.columns):
        return CheckResult(
            name    = "monotonic_scores",
            passed  = False,
            message = "'rank' or 'score' column missing",
        )

    try:
        sorted_df = df.sort_values("rank").reset_index(drop=True)
        scores    = pd.to_numeric(sorted_df["score"], errors="raise").reset_index(drop=True)
    except (ValueError, TypeError):
        return CheckResult(
            name="monotonic_scores", passed=False, message="'score' column is not numeric"
        )

    violations: list[str] = []
    for i in range(1, len(scores)):
        if scores.iloc[i] > scores.iloc[i - 1] + _SCORE_TOL:
            r_prev = int(sorted_df.iloc[i - 1]["rank"])
            r_cur  = int(sorted_df.iloc[i]["rank"])
            violations.append(
                f"rank {r_cur} ({scores.iloc[i]:.6f}) > rank {r_prev} ({scores.iloc[i-1]:.6f})"
            )
            if len(violations) >= 5:
                break

    passed = len(violations) == 0
    return CheckResult(
        name    = "monotonic_scores",
        passed  = passed,
        message = "Scores non-increasing" if passed else f"{len(violations)} inversion(s)",
        details = violations,
    )


def check_reasoning(df: pd.DataFrame) -> CheckResult:
    """All reasoning strings must be non-null and non-empty."""
    if "reasoning" not in df.columns:
        return CheckResult(
            name="reasoning", passed=False, message="'reasoning' column missing"
        )

    null_mask  = df["reasoning"].isna()
    empty_mask = df["reasoning"].astype(str).str.strip() == ""
    bad_mask   = null_mask | empty_mask

    passed     = not bad_mask.any()
    bad_ranks  = (
        df.loc[bad_mask, "rank"].tolist()
        if ("rank" in df.columns and not passed) else []
    )
    return CheckResult(
        name    = "reasoning",
        passed  = passed,
        message = (
            "All reasoning strings non-empty"
            if passed
            else f"{bad_mask.sum()} empty/null reasoning string(s)"
        ),
        details = [] if passed else [f"Affected ranks: {bad_ranks[:10]}"],
    )


def check_utf8(path: Path) -> CheckResult:
    """File bytes must decode as valid UTF-8."""
    try:
        path.read_bytes().decode("utf-8")
        return CheckResult(name="utf8_encoding", passed=True, message="Valid UTF-8")
    except FileNotFoundError:
        return CheckResult(
            name    = "utf8_encoding",
            passed  = False,
            message = f"File not found: {path}",
        )
    except UnicodeDecodeError as exc:
        return CheckResult(
            name    = "utf8_encoding",
            passed  = False,
            message = "File is not valid UTF-8",
            details = [str(exc)],
        )


# ─────────────────────────────────────────────────────────────────────────────
# § 3  Aggregate validators
# ─────────────────────────────────────────────────────────────────────────────

_DF_CHECKS = [
    check_row_count,
    check_ranks,
    check_unique_ids,
    check_score_range,
    check_monotonic_scores,
    check_reasoning,
]


def validate_dataframe(df: pd.DataFrame) -> ValidationReport:
    """
    Run all checks on an in-memory DataFrame (UTF-8 check is skipped).

    Schema is validated first; remaining checks only run if schema passes.
    """
    report = ValidationReport()

    schema = check_schema(df)
    report.checks.append(schema)

    if schema.passed:
        for fn in _DF_CHECKS:
            report.checks.append(fn(df))
    else:
        # Row count doesn't need valid columns — always include it
        report.checks.append(check_row_count(df))

    return report


def validate_file(path: str | Path) -> ValidationReport:
    """
    Read a submission CSV from disk and run all validation checks.
    Includes the UTF-8 encoding check that validate_dataframe skips.
    """
    p      = Path(path)
    report = ValidationReport(path=p)

    utf8 = check_utf8(p)
    report.checks.append(utf8)
    if not utf8.passed:
        return report

    try:
        df = pd.read_csv(p, encoding="utf-8")
    except Exception as exc:
        report.checks.append(CheckResult(
            name    = "csv_parse",
            passed  = False,
            message = f"CSV parse error: {exc}",
        ))
        return report

    sub_report = validate_dataframe(df)
    report.checks.extend(sub_report.checks)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# § 4  Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: ValidationReport, *, file: Any = None) -> None:
    """Print a human-readable validation report (ASCII-safe for all terminals)."""
    out   = file or sys.stdout
    width = 64
    sep   = "-" * width

    header_lines = ["Submission Validator -- Redrob AI Challenge"]
    if report.path:
        header_lines.append(f"  File   : {report.path.resolve()}")
    header_lines.append(
        f"  Checks : {report.n_passed} passed, {report.n_failed} failed"
    )

    print(sep, file=out)
    for line in header_lines:
        print(line, file=out)
    print(sep, file=out)

    for chk in report.checks:
        icon   = "[OK]  " if chk.passed else "[FAIL]"
        label  = chk.name.replace("_", " ").title().ljust(22)
        print(f"  {icon}  {label}  {chk.message}", file=out)
        for detail in chk.details:
            print(f"                 |- {detail}", file=out)

    print(sep, file=out)
    verdict = "  SUBMISSION VALID" if report.passed else "  SUBMISSION INVALID"
    print(verdict, file=out)
    print(sep, file=out)


# ─────────────────────────────────────────────────────────────────────────────
# § 5  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """
    CLI: python -m src.validate_submission <path> [--quiet]
    Exit 0 if valid, 1 if invalid.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog        = "validate_submission",
        description = "Validate a Redrob hackathon submission CSV.",
    )
    parser.add_argument(
        "path",
        nargs   = "?",
        default = "outputs/submission.csv",
        help    = "Path to submission CSV (default: outputs/submission.csv)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action = "store_true",
        help   = "Print only the final PASS/FAIL line",
    )
    args = parser.parse_args(argv)

    report = validate_file(args.path)

    if args.quiet:
        status = "PASSED" if report.passed else "FAILED"
        print(f"{status}  ({report.n_passed}/{len(report.checks)} checks)  {args.path}")
    else:
        print_report(report)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
