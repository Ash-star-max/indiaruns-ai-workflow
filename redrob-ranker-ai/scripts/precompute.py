"""
precompute.py — Offline pre-computation (no time limit)

Run this ONCE before the ranking step. Produces all artifacts
that run.py loads during the ≤5-min online ranking phase.

Steps
-----
    1. Stream candidates.jsonl → list of flat dicts (~200 MB RAM for 100K)
    2. Build TF-IDF pipeline → jd_semantic_scores.npy + tfidf_pipeline.joblib
    3. Extract 27-feature matrix → features.parquet
    4. Run full composite scoring → sub_scores.parquet
       (includes all 10 sub-scores, trap_penalty, composite_score, trap_labels)
    5. Save reasoning metadata → candidates_meta.parquet
    6. Embed candidate profiles (sentence-transformer) → embeddings_combined.npy
    7. Embed JD → jd_embedding.npy
    8. Save candidate_ids.npy, honeypot_flags.npy

Usage
-----
    python scripts/precompute.py
    python scripts/precompute.py --candidates data/raw/candidates.jsonl
    python scripts/precompute.py --no-embed   # skip sentence-transformer step
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from loguru import logger
from rich.console import Console
from tqdm import tqdm

# Ensure the project root is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.behavioral_signals import compute_behavioral_signal_score
from src.config import (
    ARTIFACTS_DIR,
    CANDIDATES_FILE,
    PROCESSED_DIR,
)
from src.feature_engineering import build_feature_matrix
from src.load_data import flatten_candidate, iter_candidates
from src.scoring import SCORE_WEIGHTS, score_candidate
from src.text_features import JD_SEMANTIC_SCORES_FILE, compute_jd_semantic_scores
from src.trap_detection import detect_traps

console = Console()
app = typer.Typer(help="Redrob Ranker AI — offline precomputation.")

# ─────────────────────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_FILE          = PROCESSED_DIR / "features.parquet"
CANDIDATES_META_FILE   = PROCESSED_DIR / "candidates_meta.parquet"
SUB_SCORES_FILE        = PROCESSED_DIR / "sub_scores.parquet"

EMBEDDINGS_FILE        = ARTIFACTS_DIR / "embeddings_combined.npy"
JD_EMBEDDING_FILE      = ARTIFACTS_DIR / "jd_embedding.npy"
CANDIDATE_IDS_FILE     = ARTIFACTS_DIR / "candidate_ids.npy"
HONEYPOT_FLAGS_FILE    = ARTIFACTS_DIR / "honeypot_flags.npy"

# ─────────────────────────────────────────────────────────────────────────────
# Metadata field lists
# ─────────────────────────────────────────────────────────────────────────────

# Fields needed for reasoning and tie-breaking
_META_SCALAR_FIELDS: list[str] = [
    "candidate_id",
    "years_of_experience",
    "current_title",
    "most_recent_title",
    "country",
    "location",
    "notice_period_days",
    "product_company_ratio",
    "open_to_work_flag",
    "days_since_last_active",
    "recruiter_response_rate",
]

# List fields to JSON-encode for parquet storage
_META_LIST_FIELDS: list[str] = [
    "skill_names",
]


def _encode_list(val: Any) -> str:
    """Encode a Python list (or None) to a JSON string for parquet storage."""
    if val is None:
        return "[]"
    return json.dumps(list(val))


def _make_meta_row(flat: dict[str, Any]) -> dict[str, Any]:
    """Extract reasoning metadata from a flat candidate dict."""
    row: dict[str, Any] = {}
    for field in _META_SCALAR_FIELDS:
        row[field] = flat.get(field)
    for field in _META_LIST_FIELDS:
        row[field] = _encode_list(flat.get(field))
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Scoring step (precompute all 10 sub-scores for every candidate)
# ─────────────────────────────────────────────────────────────────────────────

def _build_sub_scores_df(
    flat_rows: list[dict[str, Any]],
    jd_semantic_scores: np.ndarray,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score every candidate and return a wide DataFrame with:
      - all 10 sub-scores
      - trap_penalty, weighted_sum, composite_score
      - trap_labels (JSON-encoded list)
      - trap_risk_score
      - group contributions (5 columns)

    Uses precomputed features and jd_semantic_scores to avoid redundant work.
    """
    rows: list[dict[str, Any]] = []

    for i, flat in enumerate(tqdm(flat_rows, desc="Scoring candidates", unit="cands")):
        cid = str(flat.get("candidate_id") or "")

        # Resolve precomputed inputs
        jd_sem = float(jd_semantic_scores[i]) if i < len(jd_semantic_scores) else None

        # Precomputed feature dict for this candidate
        precomp: dict[str, float] | None = None
        if cid in features_df.index:
            precomp = features_df.loc[cid].to_dict()

        cs = score_candidate(
            flat,
            jd_semantic_score=jd_sem,
            precomputed_features=precomp,
        )
        bd = cs.score_breakdown

        row: dict[str, Any] = {
            "candidate_id":           cid,
            "composite_score":        cs.composite_score,
            "weighted_sum":           bd["weighted_sum"],
            "trap_penalty":           bd["trap_penalty"],
            "trap_risk_score":        bd["trap_detail"]["trap_risk_score"],
            "trap_labels":            json.dumps(bd["trap_detail"]["trap_labels"]),
            "group_career_relevance": bd["weight_groups"]["career_relevance"],
            "group_skill_depth":      bd["weight_groups"]["skill_depth"],
            "group_behavioral":       bd["weight_groups"]["behavioral"],
            "group_location":         bd["weight_groups"]["location"],
            "group_experience_fit":   bd["weight_groups"]["experience_fit"],
        }
        # Add all 10 sub-scores
        row.update(bd["sub_scores"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("candidate_id")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main precompute command
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def main(
    candidates: Path = typer.Option(
        CANDIDATES_FILE,
        "--candidates", "-c",
        help="Path to candidates.jsonl",
    ),
    max_records: int = typer.Option(
        0,
        "--max-records", "-n",
        help="Limit records for testing (0 = all)",
    ),
    embed: bool = typer.Option(
        True,
        "--embed/--no-embed",
        help="Whether to run sentence-transformer embedding (slow on CPU first run)",
    ),
    batch_size: int = typer.Option(
        256,
        "--batch-size",
        help="Sentence-transformer batch size for encoding",
    ),
) -> None:
    """
    Offline precomputation pipeline.  Run once; outputs go to
    data/processed/ and data/artifacts/.
    """
    t0 = time.perf_counter()
    console.rule("[bold blue]Redrob Ranker AI — Precompute[/bold blue]")

    candidates_path = Path(candidates)
    if not candidates_path.exists():
        logger.error(f"Candidates file not found: {candidates_path}")
        raise typer.Exit(code=1)

    # ── Create output directories ────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load all candidates ──────────────────────────────────────────
    logger.info("Step 1/7 — Loading candidates …")
    limit = max_records if max_records > 0 else None
    flat_rows: list[dict[str, Any]] = []
    for record in tqdm(
        iter_candidates(candidates_path, skip_invalid=True, max_records=limit),
        desc="Loading",
        unit=" cands",
    ):
        flat_rows.append(flatten_candidate(record))

    n = len(flat_rows)
    logger.info(f"  Loaded {n:,} candidates in {time.perf_counter()-t0:.1f}s")

    # Candidate IDs in load order
    candidate_ids = np.array([str(f.get("candidate_id", "")) for f in flat_rows])
    np.save(CANDIDATE_IDS_FILE, candidate_ids)
    logger.info(f"  Saved {CANDIDATE_IDS_FILE}")

    # ── Step 2: TF-IDF semantic scores ───────────────────────────────────────
    logger.info("Step 2/7 — Computing TF-IDF semantic scores …")
    t2 = time.perf_counter()
    candidate_texts: list[str] = [str(f.get("candidate_text") or "") for f in flat_rows]
    jd_semantic_scores, tfidf_pipeline = compute_jd_semantic_scores(
        candidate_texts,
        keyword_boost=True,
    )
    np.save(JD_SEMANTIC_SCORES_FILE, jd_semantic_scores)
    pipeline_path = tfidf_pipeline.save()
    logger.info(
        f"  TF-IDF done in {time.perf_counter()-t2:.1f}s — "
        f"vocab={tfidf_pipeline.vocab_size:,}, "
        f"saved to {JD_SEMANTIC_SCORES_FILE} + {pipeline_path}"
    )

    # ── Step 3: Feature matrix ───────────────────────────────────────────────
    logger.info("Step 3/7 — Extracting feature matrix …")
    t3 = time.perf_counter()
    features_df = build_feature_matrix(flat_rows, show_progress=True)
    features_df.to_parquet(FEATURES_FILE, index=True)
    logger.info(
        f"  Features done in {time.perf_counter()-t3:.1f}s — "
        f"shape={features_df.shape}, saved to {FEATURES_FILE}"
    )

    # ── Step 4: Composite scoring ────────────────────────────────────────────
    logger.info("Step 4/7 — Computing composite scores …")
    t4 = time.perf_counter()
    sub_scores_df = _build_sub_scores_df(flat_rows, jd_semantic_scores, features_df)
    sub_scores_df.to_parquet(SUB_SCORES_FILE, index=True)

    # Honeypot flags: candidates with trap_penalty below 0.10 are flagged
    honeypot_flags = np.array(
        [sub_scores_df.loc[cid, "trap_penalty"] < 0.10
         if cid in sub_scores_df.index else False
         for cid in candidate_ids],
        dtype=bool,
    )
    np.save(HONEYPOT_FLAGS_FILE, honeypot_flags)
    logger.info(
        f"  Scoring done in {time.perf_counter()-t4:.1f}s — "
        f"flagged {int(honeypot_flags.sum())} honeypots, "
        f"saved to {SUB_SCORES_FILE}"
    )

    # ── Step 5: Reasoning metadata ───────────────────────────────────────────
    logger.info("Step 5/7 — Saving reasoning metadata …")
    meta_rows = [_make_meta_row(f) for f in flat_rows]
    meta_df = pd.DataFrame(meta_rows)
    meta_df = meta_df.set_index("candidate_id")
    meta_df.to_parquet(CANDIDATES_META_FILE, index=True)
    logger.info(f"  Meta saved to {CANDIDATES_META_FILE}")

    # ── Step 6: Sentence-transformer embeddings ──────────────────────────────
    if embed:
        logger.info("Step 6/7 — Computing sentence-transformer embeddings …")
        t6 = time.perf_counter()
        try:
            from src.embedder import embed_candidates, embed_jd, load_model
            model = load_model()
            logger.info("  Model loaded")
            candidate_embeddings = embed_candidates(
                flat_rows, model, batch_size=batch_size, show_progress=True
            )
            np.save(EMBEDDINGS_FILE, candidate_embeddings)

            jd_emb = embed_jd(model)
            np.save(JD_EMBEDDING_FILE, jd_emb)

            logger.info(
                f"  Embeddings done in {time.perf_counter()-t6:.1f}s — "
                f"shape={candidate_embeddings.shape}, "
                f"saved to {EMBEDDINGS_FILE}"
            )
        except Exception as exc:
            logger.warning(f"  Embedding step failed: {exc}. Saving dummy arrays.")
            # Save placeholder arrays so run.py artifact check passes
            dummy_emb = np.zeros((n, 384), dtype=np.float32)
            np.save(EMBEDDINGS_FILE, dummy_emb)
            np.save(JD_EMBEDDING_FILE, np.zeros(384, dtype=np.float32))
    else:
        logger.info("Step 6/7 — Skipping embeddings (--no-embed)")
        # Save placeholder arrays so run.py artifact check passes
        if not EMBEDDINGS_FILE.exists():
            np.save(EMBEDDINGS_FILE, np.zeros((n, 384), dtype=np.float32))
        if not JD_EMBEDDING_FILE.exists():
            np.save(JD_EMBEDDING_FILE, np.zeros(384, dtype=np.float32))

    # ── Step 7: Summary ─────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    logger.info("Step 7/7 — Precompute complete")
    console.rule(f"[bold green]Done in {elapsed:.1f}s[/bold green]")

    top_score = float(sub_scores_df["composite_score"].max())
    logger.info(f"  Total candidates : {n:,}")
    logger.info(f"  Honeypots flagged: {int(honeypot_flags.sum())}")
    logger.info(f"  Top composite score: {top_score:.4f}")
    logger.info(f"  Artifacts in     : {ARTIFACTS_DIR}")
    logger.info(f"  Processed in     : {PROCESSED_DIR}")


if __name__ == "__main__":
    app()
