"""
run.py — Ranking entrypoint (Stage 3 reproducible command)

Usage:
    python run.py --candidates data/raw/candidates.jsonl --out outputs/submission.csv

Constraints (enforced by the hackathon judge):
    ≤5 min wall-clock · ≤16 GB RAM · CPU only · No network · ≤5 GB disk

This script loads precomputed artifacts from data/processed/ and data/artifacts/,
runs top-100 selection with reasoning, and writes the ranked submission CSV.

Run scripts/precompute.py first if artifacts do not exist.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

from src.ranker import load_artifacts, select_top_100, validate_submission

console = Console()
app = typer.Typer(help="Redrob Ranker AI — produce ranked top-100 submission CSV.")


@app.command()
def main(
    candidates: Path = typer.Option(
        Path("data/raw/candidates.jsonl"),
        "--candidates", "-c",
        help="Path to candidates.jsonl (used only for validation; scoring uses precomputed artifacts)",
        exists=False,   # don't require existence — artifacts may be all that's needed
    ),
    out: Path = typer.Option(
        Path("outputs/submission.csv"),
        "--out", "-o",
        help="Output CSV path",
    ),
    artifacts_dir: Path = typer.Option(
        Path("data/artifacts"),
        "--artifacts",
        help="Directory containing precomputed .npy and .joblib artifacts",
    ),
    processed_dir: Path = typer.Option(
        Path("data/processed"),
        "--processed",
        help="Directory containing features.parquet and sub_scores.parquet",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Load and validate artifacts only; do not write output",
    ),
) -> None:
    start = time.perf_counter()

    console.rule("[bold blue]Redrob Ranker AI[/bold blue]")
    logger.info(f"Artifacts dir  : {artifacts_dir}")
    logger.info(f"Processed dir  : {processed_dir}")
    logger.info(f"Output path    : {out}")

    # ── Artifact existence check ────────────────────────────────────────────
    required = [
        artifacts_dir / "jd_embedding.npy",
        artifacts_dir / "candidate_ids.npy",
        artifacts_dir / "honeypot_flags.npy",
        artifacts_dir / "embeddings_combined.npy",
        processed_dir / "features.parquet",
        processed_dir / "sub_scores.parquet",
        processed_dir / "candidates_meta.parquet",
    ]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        logger.error("Missing precomputed artifacts. Run: python scripts/precompute.py")
        for m in missing:
            logger.error(f"  Missing: {m}")
        raise typer.Exit(code=1)

    if dry_run:
        logger.info("Dry run — artifacts exist, skipping ranking and output.")
        raise typer.Exit(code=0)

    # ── Load precomputed artifacts ───────────────────────────────────────────
    artifacts = load_artifacts(
        artifacts_dir=Path(artifacts_dir),
        processed_dir=Path(processed_dir),
    )

    # ── Select top 100 + generate reasoning ─────────────────────────────────
    logger.info("Selecting top 100 …")
    submission_df = select_top_100(artifacts)

    # ── Validate ─────────────────────────────────────────────────────────────
    validate_submission(submission_df)
    logger.info(f"Submission validated: {len(submission_df)} rows")

    # ── Write CSV ────────────────────────────────────────────────────────────
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(out, index=False)

    elapsed = time.perf_counter() - start
    console.rule(f"[bold green]Done in {elapsed:.1f}s[/bold green]")
    logger.info(f"Submission written to: {out}")

    top_candidate = submission_df.iloc[0]
    logger.info(
        f"  Rank 1: {top_candidate['candidate_id']}  "
        f"score={top_candidate['score']:.4f}"
    )
    logger.info(f"  Reasoning sample: {top_candidate['reasoning'][:120]}")

    if elapsed > 300:
        logger.warning(f"Runtime {elapsed:.0f}s exceeds 5-minute budget!")


if __name__ == "__main__":
    app()
