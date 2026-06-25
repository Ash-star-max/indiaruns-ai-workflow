"""
streamlit_app.py — Redrob Ranker AI Professional Demo

Two modes
---------
  Precomputed  : loads outputs/submission.csv + data/processed/ → instant
  Live Demo    : upload candidates JSON/JSONL (≤500) → runs pipeline → shows results

Run
---
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── project root on path ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Redrob Ranker AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Redrob Ranker AI — Candidate Discovery Pipeline · INDIARUNS"},
)

# ── optional plotly ───────────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── lazy project imports (wrapped so the app loads even before pip install) ──
def _import_project():
    from src.config import SAMPLE_CANDIDATES_FILE, PROCESSED_DIR, ARTIFACTS_DIR, OUTPUTS_DIR
    from src.load_data import flatten_candidate, CandidateRecord, validate_schema
    from src.scoring import SCORE_WEIGHTS, score_candidates
    from src.text_features import compute_jd_semantic_scores
    from src.reasoning import generate_explanations
    from src.trap_detection import detect_traps
    from src.validate_submission import validate_file, print_report
    return (
        SAMPLE_CANDIDATES_FILE, PROCESSED_DIR, ARTIFACTS_DIR, OUTPUTS_DIR,
        flatten_candidate, CandidateRecord, validate_schema,
        SCORE_WEIGHTS, score_candidates,
        compute_jd_semantic_scores,
        generate_explanations,
        detect_traps,
        validate_file, print_report,
    )

try:
    (
        SAMPLE_CANDIDATES_FILE, PROCESSED_DIR, ARTIFACTS_DIR, OUTPUTS_DIR,
        flatten_candidate, CandidateRecord, validate_schema,
        SCORE_WEIGHTS, score_candidates,
        compute_jd_semantic_scores,
        generate_explanations,
        detect_traps,
        validate_file, _print_report,
    ) = _import_project()
    PROJECT_AVAILABLE = True
except Exception as _err:
    PROJECT_AVAILABLE = False
    SCORE_WEIGHTS = {}
    st.error(f"Could not import project modules: {_err}")


# ─────────────────────────────────────────────────────────────────────────────
# § 1  Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Metric cards */
div[data-testid="metric-container"] {
    background: #0e1117;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 12px 18px;
}
div[data-testid="metric-container"] label {
    color: #a0aec0 !important;
    font-size: 0.8rem !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* Score badge */
.badge-green  { background:#22543d; color:#9ae6b4; padding:2px 8px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.badge-blue   { background:#1a365d; color:#90cdf4; padding:2px 8px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.badge-yellow { background:#744210; color:#faf089; padding:2px 8px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.badge-red    { background:#742a2a; color:#fed7d7; padding:2px 8px; border-radius:12px; font-size:0.78rem; font-weight:600; }

/* Reasoning card */
.reasoning-card {
    background: #141924;
    border-left: 3px solid #4299e1;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #e2e8f0;
}

/* Section label */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

/* Candidate header */
.cand-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# § 2  Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_badge(score: float) -> str:
    s = round(score, 3)
    if s >= 0.85:
        css = "badge-green"
    elif s >= 0.70:
        css = "badge-blue"
    elif s >= 0.55:
        css = "badge-yellow"
    else:
        css = "badge-red"
    return f'<span class="{css}">{s:.3f}</span>'


def _trap_badge(penalty: float) -> str:
    p = round(penalty, 3)
    if p < 0.10:
        return f'<span class="badge-red">Honeypot {p:.2f}</span>'
    if p < 0.50:
        return f'<span class="badge-red">High risk {p:.2f}</span>'
    if p < 0.80:
        return f'<span class="badge-yellow">Caution {p:.2f}</span>'
    return f'<span class="badge-green">Clean {p:.2f}</span>'


def _pct_bar(value: float, width: int = 120) -> str:
    """Mini HTML progress bar for sub-score cells."""
    pct = int(value * 100)
    fill = int(value * width)
    color = "#4299e1" if value >= 0.6 else "#e53e3e" if value < 0.35 else "#d69e2e"
    return (
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="width:{width}px;height:8px;background:#2d3748;border-radius:4px">'
        f'<div style="width:{fill}px;height:8px;background:{color};border-radius:4px"></div>'
        f'</div><span style="font-size:0.78rem;color:#a0aec0">{pct}%</span>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# § 3  Data loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_precomputed() -> dict | None:
    """Load all precomputed artifacts. Returns None if not found."""
    proc  = Path("data/processed")
    out   = Path("outputs")
    sub_p = out / "submission.csv"
    ss_p  = proc / "sub_scores.parquet"
    mt_p  = proc / "candidates_meta.parquet"

    if not (ss_p.exists() and mt_p.exists()):
        return None

    sub_df  = pd.read_csv(sub_p) if sub_p.exists() else pd.DataFrame()
    ss_df   = pd.read_parquet(ss_p)
    meta_df = pd.read_parquet(mt_p)

    return {"submission": sub_df, "sub_scores": ss_df, "meta": meta_df}


@st.cache_data(show_spinner=False)
def _load_sample_flat_rows() -> list[dict]:
    if not PROJECT_AVAILABLE:
        return []
    path = SAMPLE_CANDIDATES_FILE
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data[:100]:
        try:
            rows.append(flatten_candidate(CandidateRecord.model_validate(item)))
        except Exception:
            pass
    return rows


def _parse_candidates_upload(raw: bytes) -> tuple[list[dict], list[str]]:
    """Parse JSON or JSONL bytes → (flat_rows, errors)."""
    errors: list[str] = []
    text = raw.decode("utf-8", errors="replace")

    # Try JSON array first, then JSONL
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        data = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                errors.append(f"Line {i}: JSON parse error")

    rows: list[dict] = []
    for item in data[:500]:
        if not isinstance(item, dict):
            continue
        if not validate_schema(item):
            errors.append(f"Skipped {item.get('candidate_id', '?')}: schema mismatch")
            continue
        try:
            rows.append(flatten_candidate(CandidateRecord.model_validate(item)))
        except Exception as e:
            errors.append(f"Skipped {item.get('candidate_id', '?')}: {e}")

    return rows, errors


def _run_live_pipeline(flat_rows: list[dict], jd_text: str | None = None) -> dict:
    """Run scoring pipeline; return dict with ranked_df + flat_rows."""
    t0 = time.perf_counter()

    texts     = [str(f.get("candidate_text") or "") for f in flat_rows]
    jd_scores, _ = compute_jd_semantic_scores(texts, keyword_boost=True)
    ranked_df = score_candidates(flat_rows, jd_semantic_scores=jd_scores, show_progress=False)

    id_to_flat = {str(f["candidate_id"]): f for f in flat_rows}
    cids  = ranked_df.index.tolist()
    ranks = ranked_df["rank"].tolist()
    flats = [id_to_flat.get(str(c), {"candidate_id": c}) for c in cids]
    score_results = None   # let generate_explanations call score_candidate inline

    reasoning = generate_explanations(flats, ranks=ranks)

    elapsed = time.perf_counter() - t0
    return {
        "ranked_df": ranked_df,
        "flat_rows": flat_rows,
        "id_to_flat": id_to_flat,
        "reasoning": dict(zip([str(c) for c in cids], reasoning)),
        "elapsed": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# § 4  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎯 Redrob Ranker AI")
    st.caption("Senior AI Engineer · Candidate Discovery Pipeline")
    st.divider()

    mode = st.radio(
        "**Mode**",
        ["📦 Precomputed (100K)", "⚡ Live Demo"],
        index=0,
        help="Precomputed: instant load from cached results. Live Demo: upload candidates and run pipeline.",
    )
    st.divider()

    live_flat_rows:  list[dict] = []
    live_jd_text:    str | None = None
    live_parse_errs: list[str]  = []

    if mode == "⚡ Live Demo":
        st.markdown("### Candidate file")
        cand_file = st.file_uploader(
            "Upload JSON / JSONL (≤ 500 candidates)",
            type=["json", "jsonl"],
            key="cand_upload",
        )
        if cand_file:
            live_flat_rows, live_parse_errs = _parse_candidates_upload(cand_file.read())
            if live_flat_rows:
                st.success(f"Parsed **{len(live_flat_rows)}** candidates")
            if live_parse_errs:
                with st.expander(f"⚠️ {len(live_parse_errs)} parse warnings"):
                    for e in live_parse_errs[:20]:
                        st.caption(e)
        else:
            if st.checkbox("Use sample file (10 candidates)", value=True):
                live_flat_rows = _load_sample_flat_rows()

        st.markdown("### Job Description")
        try:
            default_jd = Path("data/raw/job_description.txt").read_text(encoding="utf-8")[:3000]
        except FileNotFoundError:
            default_jd = "Senior AI Engineer role requiring vector search, ranking, and production ML experience."
        jd_file = st.file_uploader("Upload JD (.txt, optional)", type=["txt"], key="jd_upload")
        if jd_file:
            live_jd_text = jd_file.read().decode("utf-8", errors="replace")[:5000]
        else:
            live_jd_text = st.text_area(
                "JD text (editable)",
                value=default_jd,
                height=180,
                help="Paste or edit the job description. Used for TF-IDF semantic scoring.",
            )
        st.divider()

    # Score weights
    with st.expander("Score weights", expanded=False):
        group_map = {
            "Career Relevance": ["jd_semantic_score", "must_have_skill_score", "retrieval_ranking_score"],
            "Skill Depth":      ["production_ml_score", "product_shipper_score"],
            "Behavioral":       ["behavioral_signal_score"],
            "Location":         ["location_score", "salary_score"],
            "Experience Fit":   ["experience_score", "education_score"],
        }
        for grp, keys in group_map.items():
            st.markdown(f"**{grp}**")
            for k in keys:
                w = SCORE_WEIGHTS.get(k, 0)
                st.caption(f"• `{k.replace('_score','').replace('_',' ')}` → **{w:.0%}**")

    st.divider()
    st.caption("Optimised for `0.50 × NDCG@10 + 0.30 × NDCG@50`")
    st.caption("Team: **INDIARUNS** · Redrob AI Challenge 2026")


# ─────────────────────────────────────────────────────────────────────────────
# § 5  Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🎯 Redrob Ranker AI")
st.markdown(
    "**Senior AI Engineer JD** · NDCG@10-optimised composite scorer · "
    "CPU-only · Trap detection · Fact-grounded reasoning"
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# § 6  Data resolution — precomputed vs live
# ─────────────────────────────────────────────────────────────────────────────

data: dict | None = None   # keys: submission, sub_scores, meta  OR  ranked_df etc.
pipeline_result: dict | None = None
mode_label = ""

if mode == "📦 Precomputed (100K)":
    with st.spinner("Loading precomputed artifacts …"):
        precomputed = _load_precomputed()

    if precomputed is None:
        st.warning(
            "Precomputed artifacts not found. "
            "Run `python scripts/precompute.py` then `python run.py` first, "
            "or switch to **Live Demo** mode."
        )
        st.stop()

    data        = precomputed
    mode_label  = "Precomputed — 100K candidates"

else:  # Live Demo
    if not live_flat_rows:
        st.info("Upload a candidate file in the sidebar (or enable sample file) and click **Run Ranking**.")
        st.stop()

    run_btn = st.button("▶ Run Ranking", type="primary", use_container_width=True)

    # Persist result in session state so rerenders don't re-run
    if run_btn:
        st.session_state.pop("live_result", None)

    if run_btn or "live_result" in st.session_state:
        if "live_result" not in st.session_state:
            t_bar = st.progress(0, "Running TF-IDF scoring …")
            pipeline_result = _run_live_pipeline(live_flat_rows, live_jd_text)
            t_bar.progress(100, "Done!")
            st.session_state["live_result"] = pipeline_result
        else:
            pipeline_result = st.session_state["live_result"]

        mode_label = f"Live Demo — {len(live_flat_rows)} candidates"
    else:
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# § 7  Derive unified display frames
# ─────────────────────────────────────────────────────────────────────────────

if mode == "📦 Precomputed (100K)":
    sub_df  = data["submission"]
    ss_df   = data["sub_scores"]
    meta_df = data["meta"]

    n_total     = len(ss_df)
    n_honeypots = int((ss_df["trap_penalty"] < 0.10).sum())
    n_penalised = int((ss_df["trap_penalty"] < 1.0).sum())
    top_score   = float(ss_df["composite_score"].max()) if len(ss_df) else 0.0
    avg_score   = float(ss_df["composite_score"].mean()) if len(ss_df) else 0.0
    elapsed     = None

    # Build top-100 enriched table (sub_scores + meta joined to submission)
    top100_ids = sub_df["candidate_id"].tolist() if len(sub_df) else []
    ss_top     = ss_df.loc[ss_df.index.isin(top100_ids)].copy() if top100_ids else ss_df.head(100).copy()
    meta_top   = meta_df.loc[meta_df.index.isin(top100_ids)].copy() if top100_ids else meta_df.head(100).copy()
    combined   = ss_top.join(meta_top, how="left", rsuffix="_m")
    if len(sub_df):
        sub_idx = sub_df.set_index("candidate_id")
        combined = combined.join(sub_idx[["rank", "score", "reasoning"]], how="left")
    combined = combined.sort_values("rank") if "rank" in combined.columns else combined

else:
    ranked_df   = pipeline_result["ranked_df"]
    id_to_flat  = pipeline_result["id_to_flat"]
    reasoning   = pipeline_result["reasoning"]
    elapsed     = pipeline_result["elapsed"]

    n_total     = len(ranked_df)
    n_honeypots = int((ranked_df["trap_penalty"] < 0.10).sum())
    n_penalised = int((ranked_df["trap_penalty"] < 1.0).sum())
    top_score   = float(ranked_df["composite_score"].max()) if len(ranked_df) else 0.0
    avg_score   = float(ranked_df["composite_score"].mean()) if len(ranked_df) else 0.0

    # Rename to unified names
    ss_df  = ranked_df.copy()
    _reset = ranked_df.reset_index()
    sub_df = _reset[["candidate_id", "rank", "composite_score"]].rename(
        columns={"composite_score": "score"}
    ).copy()
    sub_df["reasoning"] = sub_df["candidate_id"].astype(str).map(reasoning)
    meta_top = pd.DataFrame(
        [id_to_flat.get(str(c), {"candidate_id": c}) for c in ranked_df.index],
        index=ranked_df.index,
    )
    combined = ranked_df.copy()
    combined = combined.join(meta_top[[
        "years_of_experience", "current_title", "location", "country",
        "notice_period_days", "skill_names", "open_to_work_flag",
    ]], how="left")
    combined["reasoning"] = combined.index.map(reasoning)
    combined["score"]     = combined["composite_score"]
    # rank already in combined from score_candidates


# ─────────────────────────────────────────────────────────────────────────────
# § 8  Top-level metric row
# ─────────────────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Candidates Scored", f"{n_total:,}")
c2.metric("Top Score", f"{top_score:.4f}")
c3.metric("Avg Score", f"{avg_score:.4f}")
c4.metric("Honeypots Excluded", f"{n_honeypots:,}")
c5.metric(
    "Pipeline Time",
    f"{elapsed:.1f}s" if elapsed else "< 1s (cached)",
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# § 9  Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_rank, tab_break, tab_trap, tab_reason, tab_dash = st.tabs([
    "🏆 Rankings",
    "📊 Score Breakdown",
    "⚠️ Trap Risk",
    "💡 Reasoning",
    "📈 Dashboard",
])


# ════════════════════════════════════════════════════════
# TAB 1 — Rankings
# ════════════════════════════════════════════════════════

with tab_rank:
    st.markdown(f"### Top Candidates  <span style='color:#718096;font-size:0.85rem'>{mode_label}</span>",
                unsafe_allow_html=True)

    # Search / filter
    search = st.text_input("🔍 Search by candidate ID or title", key="rank_search")

    # Build display frame
    disp_cols = {
        "rank": "Rank",
        "candidate_id": "Candidate ID",
        "score": "Score",
        "current_title": "Title",
        "years_of_experience": "YoE",
        "location": "Location",
        "trap_penalty": "Trap Penalty",
    }
    have_cols = [c for c in disp_cols if c in combined.columns]
    disp_df   = combined.reset_index() if "candidate_id" not in combined.columns else combined.copy()
    disp_df   = disp_df[[c for c in disp_cols if c in disp_df.columns]].copy()

    if search:
        mask = pd.Series(False, index=disp_df.index)
        for col in ["candidate_id", "current_title"]:
            if col in disp_df.columns:
                mask |= disp_df[col].astype(str).str.contains(search, case=False, na=False)
        disp_df = disp_df[mask]

    # Rename for display
    disp_df = disp_df.rename(columns=disp_cols)

    # Score column config
    col_cfg: dict = {}
    if "Score" in disp_df.columns:
        col_cfg["Score"] = st.column_config.ProgressColumn(
            "Score",
            min_value=0.0,
            max_value=1.0,
            format="%.4f",
        )
    if "Trap Penalty" in disp_df.columns:
        col_cfg["Trap Penalty"] = st.column_config.ProgressColumn(
            "Trap Penalty",
            min_value=0.0,
            max_value=1.0,
            format="%.3f",
        )

    st.dataframe(
        disp_df,
        use_container_width=True,
        height=480,
        column_config=col_cfg,
    )
    st.caption(f"Showing {len(disp_df)} candidates")

    st.divider()
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    # Download submission.csv
    sub_csv = sub_df[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False) if "reasoning" in sub_df.columns else sub_df.to_csv(index=False)
    dl_col1.download_button(
        "⬇ Download submission.csv",
        data=sub_csv.encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )

    # Download score breakdown
    breakdown_cols = ["candidate_id", "rank", "composite_score"] + list(SCORE_WEIGHTS.keys()) + [
        "trap_penalty", "group_career_relevance", "group_skill_depth",
        "group_behavioral", "group_location", "group_experience_fit",
    ]
    bd_src = combined.reset_index() if "candidate_id" not in combined.columns else combined.copy()
    bd_df  = bd_src[[c for c in breakdown_cols if c in bd_src.columns]]
    dl_col2.download_button(
        "⬇ Download score_breakdown.csv",
        data=bd_df.to_csv(index=False).encode("utf-8"),
        file_name="score_breakdown.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Validate button
    sub_path = Path("outputs/submission.csv")
    if dl_col3.button("✅ Validate Submission", use_container_width=True):
        if sub_path.exists():
            report = validate_file(sub_path)
            if report.passed:
                st.success(f"Submission VALID — {report.n_passed}/{len(report.checks)} checks passed")
            else:
                st.error(f"Submission INVALID — {report.n_failed} check(s) failed")
                for chk in report.failed_checks():
                    st.error(f"  ✗ {chk.name}: {chk.message}")
                    for d in chk.details:
                        st.caption(f"    {d}")
        else:
            st.warning("outputs/submission.csv not found — run `python run.py` first")


# ════════════════════════════════════════════════════════
# TAB 2 — Score Breakdown
# ════════════════════════════════════════════════════════

with tab_break:
    st.markdown("### Score Breakdown")

    sub_score_cols = [k for k in SCORE_WEIGHTS if k in combined.columns]
    group_cols = [c for c in [
        "group_career_relevance", "group_skill_depth", "group_behavioral",
        "group_location", "group_experience_fit",
    ] if c in combined.columns]
    group_labels = {
        "group_career_relevance": "Career Rel.",
        "group_skill_depth":      "Skill Depth",
        "group_behavioral":       "Behavioral",
        "group_location":         "Location",
        "group_experience_fit":   "Exp. Fit",
    }

    n_top = st.slider("Top N candidates to compare", 5, min(30, len(combined)), 10, key="break_n")
    top_n = combined.head(n_top).copy()
    top_n_idx = top_n.reset_index()["candidate_id"] if "candidate_id" not in top_n.columns else top_n["candidate_id"]
    labels = [f"#{int(r)} {str(c)[:12]}" for r, c in zip(
        top_n["rank"] if "rank" in top_n.columns else range(1, len(top_n)+1),
        top_n_idx,
    )]

    b_col1, b_col2 = st.columns([2, 1])

    with b_col1:
        st.markdown("#### Group contributions (stacked bar)")
        if group_cols and HAS_PLOTLY:
            bar_data = top_n[group_cols].copy()
            bar_data.index = labels
            bar_data.columns = [group_labels.get(c, c) for c in bar_data.columns]
            fig = px.bar(
                bar_data.reset_index().melt(id_vars="index", var_name="Group", value_name="Score"),
                x="index", y="Score", color="Group",
                title=f"Top {n_top} — group score contributions",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                height=360,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#a0aec0", xaxis_tickangle=-30,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            if group_cols:
                chart_df = top_n[group_cols].copy()
                chart_df.index = labels
                chart_df.columns = [group_labels.get(c, c) for c in chart_df.columns]
                st.bar_chart(chart_df, use_container_width=True, height=320)

    with b_col2:
        st.markdown("#### Pick a candidate")
        cid_list   = top_n_idx.tolist() if hasattr(top_n_idx, "tolist") else list(top_n_idx)
        sel_cid    = st.selectbox("Candidate", cid_list, key="break_cid")
        sel_row    = combined.loc[sel_cid] if sel_cid in combined.index else None

        if sel_row is not None and sub_score_cols:
            st.markdown(f"**Score: {_score_badge(float(sel_row.get('composite_score', 0)))}**",
                        unsafe_allow_html=True)
            st.markdown(f"**Trap penalty: {_trap_badge(float(sel_row.get('trap_penalty', 1.0)))}**",
                        unsafe_allow_html=True)
            for k in sub_score_cols:
                v = float(sel_row.get(k, 0))
                label = k.replace("_score", "").replace("_", " ").title()
                st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
                st.markdown(_pct_bar(v), unsafe_allow_html=True)

    st.divider()
    st.markdown("#### All sub-scores — top candidates")

    heat_df = top_n[sub_score_cols].copy() if sub_score_cols else pd.DataFrame()
    heat_df.index = labels
    heat_df.columns = [c.replace("_score", "").replace("_", " ").title() for c in heat_df.columns]

    if not heat_df.empty and HAS_PLOTLY:
        fig2 = px.imshow(
            heat_df,
            aspect="auto",
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            title=f"Sub-score heatmap — top {n_top}",
            height=max(300, n_top * 26),
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0",
        )
        st.plotly_chart(fig2, use_container_width=True)
    elif not heat_df.empty:
        st.dataframe(heat_df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                     use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 3 — Trap Risk
# ════════════════════════════════════════════════════════

with tab_trap:
    st.markdown("### Trap Risk Analysis")

    if "trap_penalty" not in ss_df.columns:
        st.info("Trap penalty column not available in this dataset.")
    else:
        trap_series = ss_df["trap_penalty"]
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Honeypots (< 0.10)",  int((trap_series < 0.10).sum()))
        t2.metric("High risk (< 0.50)",  int((trap_series < 0.50).sum()))
        t3.metric("Partial penalty (< 1.0)", int((trap_series < 1.0).sum()))
        t4.metric("Clean (= 1.0)",       int((trap_series == 1.0).sum()))

        st.divider()

        # Trap labels distribution
        if "trap_labels" in ss_df.columns:
            from collections import Counter
            import ast

            label_counts: Counter = Counter()
            for raw in ss_df["trap_labels"].dropna():
                try:
                    labels_list = json.loads(str(raw))
                    label_counts.update(labels_list)
                except Exception:
                    pass

            if label_counts:
                lc_df = pd.DataFrame(
                    label_counts.most_common(10),
                    columns=["Trap Label", "Count"],
                )
                lc_col, list_col = st.columns([2, 1])
                with lc_col:
                    st.markdown("#### Trap label frequency (top 10)")
                    if HAS_PLOTLY:
                        fig3 = px.bar(
                            lc_df, x="Count", y="Trap Label",
                            orientation="h", color="Count",
                            color_continuous_scale="Reds",
                            height=340,
                            title="How often each detector fired",
                        )
                        fig3.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#a0aec0", showlegend=False,
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.bar_chart(lc_df.set_index("Trap Label"), use_container_width=True)

                with list_col:
                    st.markdown("#### Detector legend")
                    definitions = {
                        "keyword_stuffing":          "Skill list may be inflated",
                        "fake_ai_profile":           "Zero-duration expert claims",
                        "generic_chatgpt_user":      "AI-generated profile language",
                        "research_only":             "No production deployment evidence",
                        "low_quality_profile":       "Thin / unverifiable profile",
                        "inactive_candidate":        "Long inactivity period",
                        "inconsistent_career":       "Frequent unexplained role changes",
                        "suspicious_timeline":       "Timeline anomalies detected",
                        "ai_keywords_no_production": "AI vocab without production proof",
                        "behavioral_trust_issues":   "Platform engagement anomalies",
                    }
                    for lbl, desc in definitions.items():
                        st.caption(f"**{lbl}**  \n{desc}")

        st.divider()

        # Flagged candidates table
        st.markdown("#### Flagged candidates (trap_penalty < 0.80)")
        flagged_src = ss_df[ss_df["trap_penalty"] < 0.80].copy()

        if mode == "📦 Precomputed (100K)":
            flagged_src = flagged_src.join(meta_df[["current_title", "location"]], how="left")

        show_cols = ["composite_score", "trap_penalty"] + (
            ["current_title", "location"] if "current_title" in flagged_src.columns else []
        ) + (["trap_labels"] if "trap_labels" in flagged_src.columns else [])

        flagged_show = flagged_src[[c for c in show_cols if c in flagged_src.columns]].sort_values("trap_penalty")
        flagged_show.index.name = "candidate_id"
        st.dataframe(
            flagged_show.head(200),
            use_container_width=True,
            height=380,
            column_config={
                "trap_penalty": st.column_config.ProgressColumn("Trap Penalty", min_value=0, max_value=1, format="%.3f"),
                "composite_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.4f"),
            },
        )


# ════════════════════════════════════════════════════════
# TAB 4 — Reasoning
# ════════════════════════════════════════════════════════

with tab_reason:
    st.markdown("### Candidate Reasoning")

    if "reasoning" not in combined.columns and len(sub_df) == 0:
        st.info("Reasoning not available — run `python run.py` or switch to Live Demo mode.")
    else:
        reason_col = "reasoning"
        if reason_col not in combined.columns and "reasoning" in sub_df.columns:
            sub_reason = sub_df.set_index("candidate_id")["reasoning"]
            combined["reasoning"] = combined.index.map(sub_reason)

        # Candidate selector
        cand_ids = combined.index.tolist()
        rank_map = combined["rank"].to_dict() if "rank" in combined.columns else {}
        cand_options = [
            f"#{int(rank_map.get(c, 0))} — {c}"
            for c in cand_ids[:100]
        ]
        sel_idx = st.selectbox("Select candidate", range(len(cand_options)),
                               format_func=lambda i: cand_options[i], key="reason_sel")

        if cand_options:
            sel_cid_r = cand_ids[sel_idx]
            sel_row_r = combined.loc[sel_cid_r]
            reasoning_text = str(sel_row_r.get("reasoning", "No reasoning available.") or "No reasoning available.")

            # ── Candidate card ───────────────────────────────────────────────
            r1, r2 = st.columns([3, 1])
            with r1:
                title = str(sel_row_r.get("current_title") or sel_row_r.get("most_recent_title") or "Unknown")
                yoe   = sel_row_r.get("years_of_experience")
                loc   = str(sel_row_r.get("location") or "")
                yoe_s = f"{int(yoe)} yrs" if yoe else "—"

                st.markdown(f'<div class="cand-header">{title}</div>', unsafe_allow_html=True)
                st.caption(f"**{sel_cid_r}** &nbsp;·&nbsp; {yoe_s} &nbsp;·&nbsp; {loc}")

                score_v = float(sel_row_r.get("composite_score", sel_row_r.get("score", 0)))
                trap_v  = float(sel_row_r.get("trap_penalty", 1.0))
                st.markdown(
                    f"Score: {_score_badge(score_v)} &nbsp; Trap: {_trap_badge(trap_v)}",
                    unsafe_allow_html=True,
                )

            with r2:
                skills_raw = sel_row_r.get("skill_names", [])
                if isinstance(skills_raw, str):
                    try:
                        skills_raw = json.loads(skills_raw)
                    except Exception:
                        skills_raw = []
                if skills_raw:
                    st.markdown("**Skills**")
                    st.caption(", ".join(str(s) for s in skills_raw[:12]))

            st.markdown(
                f'<div class="reasoning-card">{reasoning_text}</div>',
                unsafe_allow_html=True,
            )

            # Sub-scores for this candidate
            sub_score_vals = {k: float(sel_row_r.get(k, 0)) for k in SCORE_WEIGHTS if k in sel_row_r.index}
            if sub_score_vals and HAS_PLOTLY:
                st.markdown("#### Sub-score radar")
                labels_r = [k.replace("_score", "").replace("_", " ").title() for k in sub_score_vals]
                values_r = list(sub_score_vals.values()) + [list(sub_score_vals.values())[0]]
                labels_r_c = labels_r + [labels_r[0]]
                fig_r = go.Figure(go.Scatterpolar(
                    r=values_r, theta=labels_r_c, fill="toself",
                    line_color="#4299e1", fillcolor="rgba(66,153,225,0.15)",
                ))
                fig_r.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1], color="#718096")),
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0",
                    height=360, margin=dict(l=40, r=40, t=40, b=40),
                )
                st.plotly_chart(fig_r, use_container_width=True)

        st.divider()
        st.markdown("#### All reasoning strings (top candidates)")
        reason_df = combined[combined["reasoning"].notna()].head(100) if "reasoning" in combined.columns else pd.DataFrame()
        if not reason_df.empty:
            for _, row_r in reason_df.head(20).iterrows():
                r_cid   = row_r.get("candidate_id", row_r.name)
                r_rank  = int(row_r.get("rank", 0)) if pd.notna(row_r.get("rank", None)) else "—"
                r_score = float(row_r.get("composite_score", row_r.get("score", 0)))
                r_text  = str(row_r.get("reasoning", ""))
                r_title = str(row_r.get("current_title") or "")
                with st.expander(f"#{r_rank}  {r_cid}  ·  {r_title}  ·  score={r_score:.4f}"):
                    st.markdown(f'<div class="reasoning-card">{r_text}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 5 — Dashboard
# ════════════════════════════════════════════════════════

with tab_dash:
    st.markdown("### Metrics Dashboard")

    # ── Score statistics ──────────────────────────────────────────────────
    if "composite_score" in ss_df.columns:
        scores_all = ss_df["composite_score"].dropna()

        d1, d2 = st.columns(2)

        with d1:
            st.markdown("#### Score distribution")
            if HAS_PLOTLY:
                fig_h = px.histogram(
                    scores_all, nbins=80, title="Composite score — all candidates",
                    color_discrete_sequence=["#4299e1"],
                    labels={"value": "Composite Score", "count": "Candidates"},
                    height=300,
                )
                fig_h.add_vline(x=float(scores_all.quantile(0.99)), line_dash="dash",
                                line_color="#f6ad55", annotation_text="p99")
                fig_h.add_vline(x=float(scores_all.quantile(0.999)), line_dash="dash",
                                line_color="#fc8181", annotation_text="p99.9")
                fig_h.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#a0aec0", showlegend=False,
                )
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.bar_chart(scores_all.value_counts(bins=40).sort_index(), use_container_width=True)

        with d2:
            st.markdown("#### Score percentiles")
            pcts = [50, 75, 90, 95, 99, 99.9, 100]
            pct_data = {f"p{p:.0f}": round(float(scores_all.quantile(p / 100)), 4) for p in pcts}
            pct_df = pd.DataFrame.from_dict(pct_data, orient="index", columns=["Score"])
            pct_df.index.name = "Percentile"
            st.dataframe(pct_df, use_container_width=True)

            st.markdown("#### Score statistics")
            stat_data = {
                "Mean":   round(float(scores_all.mean()), 4),
                "Median": round(float(scores_all.median()), 4),
                "Std":    round(float(scores_all.std()), 4),
                "Min":    round(float(scores_all.min()), 4),
                "Max":    round(float(scores_all.max()), 4),
            }
            st.dataframe(
                pd.DataFrame.from_dict(stat_data, orient="index", columns=["Value"]),
                use_container_width=True,
            )

    st.divider()
    d3, d4 = st.columns(2)

    # ── Location distribution ─────────────────────────────────────────────
    with d3:
        # In precomputed mode meta_df has location; in live mode use combined
        _loc_src = combined if "location" in combined.columns else (
            meta_df if (mode == "📦 Precomputed (100K)" and "location" in meta_df.columns) else None
        )
        if _loc_src is not None and "location" in _loc_src.columns:
            st.markdown("#### Location distribution (top 15)")
            loc_counts = (
                _loc_src["location"].dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA).dropna()
                .value_counts()
                .head(15)
            )
            loc_df = loc_counts.reset_index()
            loc_df.columns = ["Location", "Count"]
            if HAS_PLOTLY:
                fig_loc = px.bar(
                    loc_df, x="Count", y="Location",
                    orientation="h",
                    color_discrete_sequence=["#68d391"],
                    height=380,
                )
                fig_loc.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#a0aec0", yaxis_title="", xaxis_title="Candidates",
                )
                st.plotly_chart(fig_loc, use_container_width=True)
            else:
                st.bar_chart(loc_counts, use_container_width=True)

    # ── YoE distribution ──────────────────────────────────────────────────
    with d4:
        yoe_col = "years_of_experience"
        _yoe_candidates = [combined, ss_df] + ([meta_df] if mode == "📦 Precomputed (100K)" else [])
        yoe_src = next((df for df in _yoe_candidates if yoe_col in df.columns), None)
        if yoe_src is not None and yoe_col in yoe_src.columns:
            st.markdown("#### Years of experience distribution")
            yoe_vals = pd.to_numeric(yoe_src[yoe_col], errors="coerce").dropna()
            yoe_capped = yoe_vals.clip(0, 30)
            if HAS_PLOTLY:
                fig_yoe = px.histogram(
                    yoe_capped, nbins=30,
                    title="YoE distribution",
                    color_discrete_sequence=["#b794f4"],
                    labels={"value": "Years of Experience"},
                    height=380,
                )
                fig_yoe.add_vline(x=7, line_dash="dash", line_color="#f6ad55",
                                  annotation_text="JD target (7 yrs)")
                fig_yoe.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#a0aec0", showlegend=False,
                )
                st.plotly_chart(fig_yoe, use_container_width=True)
            else:
                st.bar_chart(yoe_capped.value_counts().sort_index(), use_container_width=True)

    st.divider()

    # ── Sub-score means ───────────────────────────────────────────────────
    sub_score_cols_all = [k for k in SCORE_WEIGHTS if k in ss_df.columns]
    if sub_score_cols_all:
        st.markdown("#### Average sub-scores — all candidates vs top 100")
        means_all  = ss_df[sub_score_cols_all].mean()
        means_top  = combined[sub_score_cols_all].mean() if all(c in combined.columns for c in sub_score_cols_all) else means_all

        cmp_df = pd.DataFrame({
            "All Candidates": means_all,
            "Top 100": means_top,
        })
        cmp_df.index = [c.replace("_score", "").replace("_", " ").title() for c in cmp_df.index]

        if HAS_PLOTLY:
            fig_cmp = px.bar(
                cmp_df.reset_index().melt(id_vars="index", var_name="Group", value_name="Avg Score"),
                x="index", y="Avg Score", color="Group", barmode="group",
                color_discrete_map={"All Candidates": "#4a5568", "Top 100": "#4299e1"},
                height=340,
                title="Sub-score comparison: population vs top-100",
            )
            fig_cmp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#a0aec0", xaxis_tickangle=-30,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.bar_chart(cmp_df, use_container_width=True)

    # ── Pipeline metadata ─────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Pipeline metadata")
    meta_info = {
        "Mode":               mode.split()[1],
        "Total candidates":   f"{n_total:,}",
        "Honeypots excluded": f"{n_honeypots:,}",
        "Partial penalties":  f"{n_penalised:,}",
        "Top score":          f"{top_score:.6f}",
        "Pipeline time":      f"{elapsed:.2f}s" if elapsed else "< 1s (precomputed)",
        "Throughput":         f"{n_total/elapsed:,.0f} cands/sec" if elapsed else "N/A",
        "Eval metric":        "0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10",
        "Team":               "INDIARUNS",
        "Model":              "TF-IDF (offline) + all-MiniLM-L6-v2 embeddings (optional)",
    }
    st.dataframe(
        pd.DataFrame.from_dict(meta_info, orient="index", columns=["Value"]),
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# § 10  Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Redrob Ranker AI · INDIARUNS · Redrob Data & AI Challenge 2026 · "
    "CPU-only · NDCG@10 optimised · `streamlit run streamlit_app.py`"
)
