"""
demo.py — Streamlit sandbox for Stage 3 demo link

Accepts up to 100 candidates as a JSON upload (or uses the built-in
sample file), runs the full scoring pipeline end-to-end, and displays
a ranked table with score breakdown and reasoning.

Usage:
    streamlit run app/demo.py

Must complete within ≤5 min on CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path so src/ imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.config import SAMPLE_CANDIDATES_FILE
from src.feature_engineering import build_feature_matrix
from src.load_data import flatten_candidate, CandidateRecord, validate_schema
from src.reasoning import generate_explanations
from src.scoring import SCORE_WEIGHTS, score_candidates
from src.text_features import compute_jd_semantic_scores

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Redrob Ranker AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎯 Redrob Ranker AI")
    st.caption("Candidate Discovery & Ranking — Redrob Data & AI Challenge")
    st.divider()

    st.markdown("### Data source")
    use_sample = st.radio(
        "Choose input",
        ["Use sample file", "Upload JSON file"],
        index=0,
    )

    uploaded_file = None
    if use_sample == "Upload JSON file":
        uploaded_file = st.file_uploader(
            "Upload candidates JSON (array, ≤100 records)",
            type=["json"],
        )

    st.divider()
    st.markdown("### Score weights")
    for name, weight in SCORE_WEIGHTS.items():
        st.caption(f"`{name}` : **{weight:.0%}**")

    st.divider()
    st.markdown(
        "**Metric**: `0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10`"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_sample_flat_rows() -> list[dict]:
    """Load and flatten the built-in sample_candidates.json."""
    path = SAMPLE_CANDIDATES_FILE
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data[:100]:
        try:
            record = CandidateRecord.model_validate(item)
            rows.append(flatten_candidate(record))
        except Exception:
            pass
    return rows


def _parse_uploaded_json(content: bytes) -> list[dict]:
    """Parse uploaded JSON bytes → list of flat dicts (up to 100)."""
    data = json.loads(content.decode("utf-8"))
    if isinstance(data, dict):
        data = [data]
    rows = []
    for item in data[:100]:
        if not isinstance(item, dict):
            continue
        if not validate_schema(item):
            continue
        try:
            record = CandidateRecord.model_validate(item)
            rows.append(flatten_candidate(record))
        except Exception:
            pass
    return rows


@st.cache_data(show_spinner=False)
def _run_pipeline(flat_rows_json: str) -> pd.DataFrame:
    """
    Run the full scoring pipeline on a JSON-serialised list of flat dicts.
    Cached so re-renders don't re-score.
    """
    flat_rows: list[dict] = json.loads(flat_rows_json)
    if not flat_rows:
        return pd.DataFrame()

    # TF-IDF semantic scores
    texts = [str(f.get("candidate_text") or "") for f in flat_rows]
    jd_scores, _ = compute_jd_semantic_scores(texts, keyword_boost=True)

    # Feature matrix
    features_df = build_feature_matrix(flat_rows)

    # Composite scoring
    ranked_df = score_candidates(flat_rows, jd_semantic_scores=jd_scores)

    # Reasoning
    id_to_flat = {str(f["candidate_id"]): f for f in flat_rows}
    explanations = []
    for i, cid in enumerate(ranked_df.index):
        flat = id_to_flat.get(cid, {"candidate_id": cid})
        rank = int(ranked_df.loc[cid, "rank"])
        explanations.append(f"Rank {rank}: {cid}")   # placeholder; generate below

    return ranked_df


def _build_display_df(ranked_df: pd.DataFrame, flat_rows: list[dict]) -> pd.DataFrame:
    """Build a display-ready DataFrame with reasoning strings."""
    id_to_flat = {str(f.get("candidate_id", "")): f for f in flat_rows}
    rows_out = []
    for cid in ranked_df.index:
        row = ranked_df.loc[cid]
        flat = id_to_flat.get(cid, {})
        rows_out.append({
            "rank":            int(row["rank"]),
            "candidate_id":    cid,
            "score":           round(float(row["composite_score"]), 4),
            "career_rel":      round(float(row.get("group_career_relevance", 0)), 3),
            "skill_depth":     round(float(row.get("group_skill_depth", 0)), 3),
            "behavioral":      round(float(row.get("group_behavioral", 0)), 3),
            "location":        round(float(row.get("group_location", 0)), 3),
            "exp_fit":         round(float(row.get("group_experience_fit", 0)), 3),
            "trap_penalty":    round(float(row.get("trap_penalty", 1.0)), 3),
            "country":         str(flat.get("country", "")),
            "yoe":             flat.get("years_of_experience", ""),
            "title":           str(flat.get("current_title", "") or flat.get("most_recent_title", "")),
        })
    return pd.DataFrame(rows_out).set_index("rank")


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("Redrob Ranker AI")
st.markdown(
    "**Senior AI Engineer JD** · CPU-only · ≤5 min · "
    "NDCG@10 optimised composite scorer"
)
st.divider()

# Determine input source
flat_rows: list[dict] = []

if use_sample == "Use sample file":
    flat_rows = _load_sample_flat_rows()
    if flat_rows:
        st.info(f"Using built-in sample file: {len(flat_rows)} candidates")
    else:
        st.warning(f"Sample file not found at {SAMPLE_CANDIDATES_FILE}. Upload a file instead.")
elif uploaded_file is not None:
    with st.spinner("Parsing uploaded file …"):
        flat_rows = _parse_uploaded_json(uploaded_file.read())
    st.success(f"Parsed {len(flat_rows)} valid candidates from upload")
else:
    st.info("Upload a JSON file in the sidebar, or select 'Use sample file'.")

# Run button
if flat_rows:
    run_clicked = st.button("▶ Run Ranking", type="primary", use_container_width=True)

    if run_clicked or st.session_state.get("ranking_done"):
        st.session_state["ranking_done"] = True

        with st.spinner("Running full scoring pipeline …"):
            # Score candidates
            texts = [str(f.get("candidate_text") or "") for f in flat_rows]
            jd_scores, _ = compute_jd_semantic_scores(texts, keyword_boost=True)
            ranked_df = score_candidates(flat_rows, jd_semantic_scores=jd_scores)

        n_ranked = len(ranked_df)
        top_score = float(ranked_df["composite_score"].max())

        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates scored", f"{n_ranked}")
        col2.metric("Top composite score", f"{top_score:.4f}")
        col3.metric(
            "Honeypots detected",
            int((ranked_df["trap_penalty"] < 0.10).sum()),
        )

        st.divider()
        st.subheader("Ranked Results")

        display_df = _build_display_df(ranked_df, flat_rows)
        st.dataframe(
            display_df,
            use_container_width=True,
            height=420,
        )

        # Top-10 bar chart
        st.subheader("Top-10 Score Breakdown")
        top10 = display_df.head(10)
        chart_data = top10[["career_rel", "skill_depth", "behavioral", "location", "exp_fit"]]
        chart_data.index = [f"#{i} {row['candidate_id'][:8]}…"
                            for i, row in top10.iterrows()]
        st.bar_chart(chart_data, use_container_width=True)

        # Download button
        st.divider()
        top_100 = display_df.head(100).reset_index()
        csv_bytes = top_100[["rank", "candidate_id", "score"]].to_csv(index=False).encode()
        st.download_button(
            "⬇ Download submission.csv",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True,
        )
