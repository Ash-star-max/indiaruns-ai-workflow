"""
ranker.py — Top-100 selection and submission builder

Loads precomputed artifacts from data/processed/ and data/artifacts/,
selects the top-100 candidates, generates fact-grounded reasoning strings,
and returns the submission DataFrame.

Tie-breaking rules
------------------
    1. Primary   : composite_score DESC
    2. Secondary : recruiter_response_rate DESC  (more responsive = better)
    3. Tertiary  : candidate_id ASC (lexicographic, deterministic)

Submission schema
-----------------
    candidate_id | rank | score | reasoning

Public API
----------
    load_artifacts(artifacts_dir, processed_dir) -> dict
    select_top_100(artifacts) -> pd.DataFrame
    validate_submission(df) -> None            (raises ValueError on violations)
    run_pipeline(artifacts_dir, processed_dir, out_dir) -> pd.DataFrame
    _build_score_breakdown(submission, artifacts) -> pd.DataFrame
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import ARTIFACTS_DIR, OUTPUTS_DIR, PROCESSED_DIR
from src.reasoning import generate_explanation
from src.scoring import SCORE_WEIGHTS, CandidateScore

# ─────────────────────────────────────────────────────────────────────────────
# § 1  Artifact paths (mirrors precompute.py output)
# ─────────────────────────────────────────────────────────────────────────────

_SUB_SCORES_FILE     = PROCESSED_DIR / "sub_scores.parquet"
_META_FILE           = PROCESSED_DIR / "candidates_meta.parquet"
_FEATURES_FILE       = PROCESSED_DIR / "features.parquet"
_CANDIDATE_IDS_FILE  = ARTIFACTS_DIR / "candidate_ids.npy"
_HONEYPOT_FLAGS_FILE = ARTIFACTS_DIR / "honeypot_flags.npy"
_EMBEDDINGS_FILE     = ARTIFACTS_DIR / "embeddings_combined.npy"
_JD_EMBEDDING_FILE   = ARTIFACTS_DIR / "jd_embedding.npy"

# ─────────────────────────────────────────────────────────────────────────────
# § 2  Artifact loader
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts(
    artifacts_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Load all precomputed artifacts from disk.

    Parameters
    ----------
    artifacts_dir : Path to data/artifacts/ (defaults to config.ARTIFACTS_DIR)
    processed_dir : Path to data/processed/ (defaults to config.PROCESSED_DIR)

    Returns
    -------
    dict with keys:
        sub_scores   : pd.DataFrame  — composite + all sub-scores (index=candidate_id)
        meta         : pd.DataFrame  — reasoning metadata (index=candidate_id)
        features     : pd.DataFrame  — 27 numeric features  (index=candidate_id)
        candidate_ids: np.ndarray    — string array of IDs aligned with embeddings
        honeypot_flags: np.ndarray   — bool array aligned with candidate_ids
        embeddings   : np.ndarray    — (N, 384) float32 candidate embeddings
        jd_embedding : np.ndarray    — (384,)   float32 JD embedding
    """
    art_dir  = Path(artifacts_dir)  if artifacts_dir  else ARTIFACTS_DIR
    proc_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR

    sub_scores_path = proc_dir / "sub_scores.parquet"
    meta_path       = proc_dir / "candidates_meta.parquet"
    features_path   = proc_dir / "features.parquet"
    ids_path        = art_dir  / "candidate_ids.npy"
    flags_path      = art_dir  / "honeypot_flags.npy"
    emb_path        = art_dir  / "embeddings_combined.npy"
    jd_emb_path     = art_dir  / "jd_embedding.npy"

    logger.info("Loading artifacts …")
    artifacts: dict[str, Any] = {}

    artifacts["sub_scores"]    = pd.read_parquet(sub_scores_path)
    logger.info(f"  sub_scores      : {artifacts['sub_scores'].shape}")

    artifacts["meta"]          = pd.read_parquet(meta_path)
    logger.info(f"  candidates_meta : {artifacts['meta'].shape}")

    artifacts["features"]      = pd.read_parquet(features_path)
    logger.info(f"  features        : {artifacts['features'].shape}")

    artifacts["candidate_ids"] = np.load(ids_path, allow_pickle=True)
    logger.info(f"  candidate_ids   : {len(artifacts['candidate_ids']):,}")

    artifacts["honeypot_flags"] = np.load(flags_path, allow_pickle=True)
    logger.info(f"  honeypot_flags  : {int(artifacts['honeypot_flags'].sum())} flagged")

    artifacts["embeddings"]    = np.load(emb_path)
    logger.info(f"  embeddings      : {artifacts['embeddings'].shape}")

    artifacts["jd_embedding"]  = np.load(jd_emb_path)
    logger.info(f"  jd_embedding    : {artifacts['jd_embedding'].shape}")

    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# § 3  CandidateScore reconstruction from precomputed data
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_candidate_score(cid: str, row: pd.Series) -> CandidateScore:
    """
    Reconstruct a CandidateScore from a sub_scores row.
    Used to provide generate_explanation() with a score_result object.
    """
    sub_scores: dict[str, float] = {k: float(row[k]) for k in SCORE_WEIGHTS if k in row}
    trap_labels: list[str] = json.loads(str(row.get("trap_labels", "[]")))
    composite   = float(row.get("composite_score", 0.0))
    trap_penalty = float(row.get("trap_penalty", 1.0))
    weighted_sum = float(row.get("weighted_sum", 0.0))

    weight_groups = {
        "career_relevance": float(row.get("group_career_relevance", 0.0)),
        "skill_depth":      float(row.get("group_skill_depth",      0.0)),
        "behavioral":       float(row.get("group_behavioral",       0.0)),
        "location":         float(row.get("group_location",         0.0)),
        "experience_fit":   float(row.get("group_experience_fit",   0.0)),
    }

    score_breakdown = {
        "sub_scores":          sub_scores,
        "weights":             dict(SCORE_WEIGHTS),
        "weighted_sub_scores": {k: round(sub_scores.get(k, 0.0) * SCORE_WEIGHTS[k], 6)
                                for k in SCORE_WEIGHTS},
        "weighted_sum":        weighted_sum,
        "trap_penalty":        trap_penalty,
        "composite_score":     composite,
        "weight_groups":       weight_groups,
        "skill_detail":        {},
        "trap_detail": {
            "trap_labels":     trap_labels,
            "trap_risk_score": float(row.get("trap_risk_score", 0.0)),
            "explanation":     [""] * 10,
        },
    }

    return CandidateScore(
        candidate_id    = cid,
        composite_score = composite,
        rank_key        = (-composite, cid),
        score_breakdown = score_breakdown,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § 4  Meta → flat dict reconstruction (for reasoning)
# ─────────────────────────────────────────────────────────────────────────────

def _meta_to_flat(cid: str, row: pd.Series) -> dict[str, Any]:
    """
    Reconstruct a minimal flat dict from the candidates_meta row.
    Only fields used by generate_explanation() are included.
    """
    flat: dict[str, Any] = {"candidate_id": cid}

    for col in row.index:
        val = row[col]
        if col == "skill_names":
            # JSON-encoded list field
            try:
                flat[col] = json.loads(str(val)) if val and val != "[]" else []
            except (json.JSONDecodeError, TypeError):
                flat[col] = []
        else:
            flat[col] = None if pd.isna(val) else val

    return flat


# ─────────────────────────────────────────────────────────────────────────────
# § 5  Top-100 selection
# ─────────────────────────────────────────────────────────────────────────────

def select_top_100(artifacts: dict[str, Any]) -> pd.DataFrame:
    """
    Select top 100 candidates from precomputed scores and generate reasoning.

    Honeypot-flagged candidates (trap_penalty < 0.10) are excluded before
    ranking to ensure fabricated profiles never appear in the submission.

    Tie-breaking:
        1. composite_score DESC
        2. recruiter_response_rate DESC  (from candidates_meta)
        3. candidate_id ASC

    Returns
    -------
    pd.DataFrame with columns:
        candidate_id | rank | score | reasoning
    Sorted by rank ascending (rank=1 is first place).
    """
    sub_df  = artifacts["sub_scores"]    # index=candidate_id
    meta_df = artifacts["meta"]          # index=candidate_id

    # Build honeypot exclusion set from flagged IDs
    candidate_ids  = artifacts.get("candidate_ids",  np.array([]))
    honeypot_flags = artifacts.get("honeypot_flags", np.zeros(len(candidate_ids), dtype=bool))
    honeypot_ids: set[str] = set(
        str(cid) for cid, flag in zip(candidate_ids, honeypot_flags) if flag
    )
    if honeypot_ids:
        logger.info(f"  Excluding {len(honeypot_ids)} honeypot candidates from ranking")
        sub_df = sub_df.loc[~sub_df.index.isin(honeypot_ids)]

    # Merge recruiter_response_rate for tie-breaking
    if "recruiter_response_rate" in meta_df.columns:
        rrr = meta_df[["recruiter_response_rate"]].copy()
        combined = sub_df.join(rrr, how="left")
        combined["recruiter_response_rate"] = combined[
            "recruiter_response_rate"
        ].fillna(0.0)
    else:
        combined = sub_df.copy()
        combined["recruiter_response_rate"] = 0.0

    # Deterministic three-key sort
    combined = combined.sort_values(
        by=["composite_score", "recruiter_response_rate"],
        ascending=[False, False],
    )
    # Within ties on both numeric keys, sort candidate_id ascending
    combined = combined.reset_index()
    combined = combined.sort_values(
        by=["composite_score", "recruiter_response_rate", "candidate_id"],
        ascending=[False, False, True],
    )

    top100 = combined.head(100).reset_index(drop=True)

    # ── Generate reasoning strings ────────────────────────────────────────────
    reasoning_strings: list[str] = []
    for rank_pos, row in top100.iterrows():
        cid = str(row["candidate_id"])
        rank = int(rank_pos) + 1   # 1-based

        # Reconstruct score_result and flat dict
        if cid in sub_df.index:
            score_result = _reconstruct_candidate_score(cid, sub_df.loc[cid])
        else:
            score_result = None

        if cid in meta_df.index:
            flat = _meta_to_flat(cid, meta_df.loc[cid])
        else:
            flat = {"candidate_id": cid}

        try:
            explanation = generate_explanation(flat, score_result, rank=rank)
        except Exception as exc:
            logger.warning(f"Reasoning failed for {cid}: {exc}")
            explanation = f"Ranked #{rank} by composite score."

        reasoning_strings.append(explanation)

    submission = pd.DataFrame({
        "candidate_id": top100["candidate_id"].tolist(),
        "rank":         list(range(1, len(top100) + 1)),
        "score":        top100["composite_score"].round(6).tolist(),
        "reasoning":    reasoning_strings,
    })

    return submission


# ─────────────────────────────────────────────────────────────────────────────
# § 6  Submission validator
# ─────────────────────────────────────────────────────────────────────────────

def validate_submission(df: pd.DataFrame) -> None:
    """
    Validate the submission DataFrame against the hackathon spec.

    Raises ValueError on the first violation found.
    """
    required_cols = {"candidate_id", "rank", "score", "reasoning"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Submission missing columns: {missing}")

    if len(df) == 0:
        raise ValueError("Submission is empty")

    if len(df) > 100:
        raise ValueError(f"Submission has {len(df)} rows; max is 100")

    ranks = sorted(df["rank"].tolist())
    expected = list(range(1, len(df) + 1))
    if ranks != expected:
        raise ValueError(f"Ranks must be consecutive 1-{len(df)}; got {ranks[:5]}…")

    if df["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate_ids in submission")

    if not df["score"].between(0.0, 1.0).all():
        bad = df.loc[~df["score"].between(0.0, 1.0), "candidate_id"].tolist()
        raise ValueError(f"Scores out of [0,1] for: {bad}")

    if df["reasoning"].isna().any() or (df["reasoning"] == "").any():
        raise ValueError("Some reasoning strings are empty or null")
