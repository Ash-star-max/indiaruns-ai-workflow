# Redrob Ranker AI

> **Redrob Data & AI Challenge 2026** — Team **INDIARUNS**
> Rank the 100 best Senior AI Engineer candidates from a pool of 100,000.

---

## Table of Contents

1. [Challenge Overview](#challenge-overview)
2. [System Architecture](#system-architecture)
3. [Ranking Methodology](#ranking-methodology)
4. [Behavioral Intelligence](#behavioral-intelligence)
5. [Trap Detection](#trap-detection)
6. [Explainability Engine](#explainability-engine)
7. [Runtime Compliance](#runtime-compliance)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Docker](#docker)
11. [Testing](#testing)
12. [Demo](#demo)
13. [AI Tools Declaration](#ai-tools-declaration)

---

## Challenge Overview

| Item | Detail |
|------|--------|
| **Challenge** | Redrob Data & AI Challenge 2026 |
| **Task** | Select the top-100 candidates for a Senior AI Engineer role from 100,000 profiles |
| **Evaluation** | `0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10` |
| **Constraints** | <= 5 min wall-clock · <= 16 GB RAM · CPU-only · No network · <= 5 GB disk |
| **Dataset** | `data/raw/candidates.jsonl` — 100,000 profiles, ~465 MB |
| **Output** | `outputs/submission.csv` — ranked top-100, columns: `candidate_id, rank, score, reasoning` |

The evaluation metric is **rank-weighted** — getting position #1 wrong is penalised far more than getting position #50 wrong. Our scoring architecture is designed specifically to front-load the signals that discriminate _who can actually ship retrieval and ranking systems_ rather than who merely lists the right keywords.

---

## System Architecture

### Two-Phase Pipeline

```
Phase 1 — Offline Precompute (run once, ~13 min on 100K)
──────────────────────────────────────────────────────────
  data/raw/candidates.jsonl
        │
        v
  load_data.py            Parse + validate 100K profiles (Pydantic v2)
  text_features.py        TF-IDF fit on corpus + JD; cosine similarity scores
  feature_engineering.py  27 numeric features per candidate
  behavioral_signals.py   18 platform signal dimensions → behavioral_score
  trap_detection.py       10 detectors → trap_penalty in [0.05, 1.0]
  scoring.py              10 sub-scores → composite_score = weighted_sum × trap_penalty
        │
        v
  data/processed/sub_scores.parquet       (100K x 25 columns)
  data/processed/candidates_meta.parquet  (100K x 40 columns)
  data/artifacts/honeypot_flags.npy
  data/artifacts/candidate_ids.npy

Phase 2 — Online Ranking (< 1 second)
──────────────────────────────────────
  Precomputed artifacts
        │
        v
  ranker.py     Sort by composite_score, exclude honeypots, select top 100
  reasoning.py  Generate 1-2 sentence fact-grounded explanations
        │
        v
  outputs/submission.csv
  outputs/score_breakdown.csv
```

### Directory Layout

```
redrob-ranker-ai/
├── src/
│   ├── config.py              Single source of truth — all weights & paths
│   ├── load_data.py           Pydantic v2 schema, flatten_candidate()
│   ├── jd_understanding.py    JD parsing, must-have skill group extraction
│   ├── text_features.py       TF-IDF semantic scorer, keyword boost
│   ├── feature_engineering.py 27 numeric features → normalised feature matrix
│   ├── behavioral_signals.py  18-signal behavioral scorer
│   ├── trap_detection.py      10 trap detectors, compound penalty
│   ├── scoring.py             10 sub-scores → composite, CandidateScore
│   ├── reasoning.py           Fact-grounded explanation generator
│   ├── ranker.py              Top-100 selection, submission builder
│   ├── embedder.py            sentence-transformers/all-MiniLM-L6-v2
│   └── validate_submission.py 8-check submission validator
├── scripts/
│   └── precompute.py          Phase 1 offline pipeline (run once)
├── tests/                     1,013 pytest tests
├── app/
│   └── demo.py                Basic Streamlit demo
├── streamlit_app.py           Professional 5-tab Streamlit dashboard
├── run.py                     Phase 2 entrypoint (< 1 s)
├── Dockerfile
└── requirements.txt
```

---

## Ranking Methodology

### Composite Score Formula

```
composite_score = weighted_sum × trap_penalty

where:

weighted_sum =
    (0.15 × jd_semantic_score)       +   # TF-IDF cosine sim to JD text
    (0.12 × must_have_skill_score)   +   # 4 mandatory skill groups
    (0.08 × retrieval_ranking_score) +   # vector-search / LTR depth
    (0.15 × production_ml_score)     +   # ML shipped to real users
    (0.10 × product_shipper_score)   +   # product vs consulting
    (0.20 × behavioral_signal_score) +   # platform availability & trust
    (0.09 × location_score)          +   # India / Bangalore preferred
    (0.03 × salary_score)            +   # salary fit
    (0.05 × experience_score)        +   # Gaussian centred on 7 yrs
    (0.03 × education_score)             # tier bonus

trap_penalty in [0.05, 1.0]   (multiplicative; < 0.10 → honeypot exclusion)
```

### Five Score Groups

| Group | Weight | Sub-scores |
|-------|--------|------------|
| **Career Relevance** | 35% | jd_semantic · must_have_skills · retrieval_ranking |
| **Skill Depth** | 25% | production_ml · product_shipper |
| **Behavioral** | 20% | behavioral_signal |
| **Location** | 12% | location · salary |
| **Experience Fit** | 8% | experience · education |

### Sub-Score Design Rationale

**`jd_semantic_score` (0.15)** — TF-IDF cosine similarity against the exact JD text. Trained on the full 100K corpus so rarer JD terms (e.g. "dense retrieval", "NDCG") receive higher IDF weight. An optional keyword boost amplifies tier-1 skill signals. This is the single best global discriminator.

**`must_have_skill_score` (0.12)** — The JD specifies four hard requirement clusters: (1) embeddings & retrieval, (2) vector databases (Pinecone/Weaviate/Qdrant/FAISS), (3) Python engineering, (4) ranking & evaluation metrics (NDCG/MAP/MRR). Each cluster is scored independently; zero on any cluster penalises the group score.

**`retrieval_ranking_score` (0.08)** — Domain-specific depth in vector search and learning-to-rank. High signal when present; absent for strong generalists, hence lower weight.

**`production_ml_score` (0.15)** — Combines feature-engineered signals (tier-1 skill density, duration-quality, Redrob assessment scores) with text evidence of ML systems deployed to real users. Penalised by JD-derived disqualifiers.

**`product_shipper_score` (0.10)** — The JD explicitly targets "fast-moving startups" and "candidates who ship working systems over perfect architecture." Preference for product companies, right company size, and AI/tech industry. Consulting-only careers receive a 0.15x penalty.

**`behavioral_signal_score` (0.20)** — Highest-weighted single sub-score because this is a founding-team hire at an early-stage startup. See [Behavioral Intelligence](#behavioral-intelligence).

**`experience_score` (0.05)** — Gaussian score peaked at 7 years (sigma = 2.5). A 5-year specialist with perfect skills outranks a 10-year generalist. Low weight by design.

**`education_score` (0.03)** — Lowest weight. Tier-based bonus (PhD > Masters > Top-tier Bachelor > Other). Absence of degree does not disqualify a self-taught engineer with deep production ML history.

### Tie-Breaking

Ties on composite score are broken deterministically:

1. `composite_score` DESC
2. `recruiter_response_rate` DESC (more responsive = better founding-team fit)
3. `candidate_id` ASC (lexicographic, ensures reproducible ordering)

### Hard Disqualifiers (Multiplicative Penalties)

| Condition | Penalty Factor |
|-----------|---------------|
| Consulting-only career (TCS, Infosys, Wipro, etc.) | x 0.15 |
| Pure research — no production deployment evidence | x 0.20 |
| Non-technical job title | x 0.25 |
| Outside India, not willing to relocate | x 0.40 |
| Notice period > 120 days | x 0.60 |
| Only LangChain skills, no Tier-1 ML skills | x 0.70 |

### YoE Gaussian Fit

```
experience_score = exp( -(yoe - 7.0)^2 / (2 * 2.5^2) )
```

Peaks at 7 years per the JD. Score at 5 yrs = 0.85, at 10 yrs = 0.74, at 15 yrs = 0.26.

---

## Behavioral Intelligence

The behavioral layer converts **18 raw Redrob platform signals** into a single `behavioral_signal_score in [0, 1]` across three dimensions.

### Signal Architecture

```
behavioral_signal_score
        │
        ├── availability_score (45%)
        │       open_to_work_flag              (40%)
        │       days_since_last_active          (35%)  exponential decay, half-life = 30 days
        │       notice_period_days              (25%)
        │
        ├── engagement_score (35%)
        │       recruiter_response_rate         (40%)
        │       avg_response_time_hours         (25%)  lower is better
        │       applications_submitted_30d      (20%)
        │       interview_completion_rate       (15%)
        │
        └── trust_score (20%)
                email + phone + LinkedIn verification  (35%)
                profile_completeness_score             (30%)
                connection_count + endorsements        (20%)
                github_activity_score                  (15%)
```

### Modifiers

| Modifier | Effect |
|----------|--------|
| **Market demand bonus** | Additive <= +0.08 from `saved_by_recruiters_30d`, `profile_views_30d`, `search_appearance_30d` |
| **Hiring history multiplier** | x [0.90, 1.10] from `offer_acceptance_rate` |
| **Inactivity hard cap** | If inactive > 90 days AND `open_to_work = False` → score capped at 0.30 |

---

## Trap Detection

Ten independent detectors flag manipulated, fabricated, or low-signal profiles. Detected traps apply **multiplicative penalties** that compound: a profile triggering two traps at 0.70x and 0.60x receives a 0.42x penalty on its final score. Candidates with a compound penalty below 0.10 are treated as **honeypots** and excluded from the ranked output entirely.

### Detector Catalogue

| Detector | Penalty | What It Catches |
|----------|---------|----------------|
| `fake_ai_profile` | **x 0.20** | Expert/advanced skills with zero usage duration — fabricated competency claims |
| `research_only` | **x 0.30** | Academic-only profile; no evidence of systems shipped to users |
| `suspicious_timeline` | x 0.50 | Timeline anomalies: career months exceeding years x 12 x 1.5, impossible overlap |
| `inconsistent_career` | x 0.60 | Excessive role churn, unexplained gaps, non-matching title/skill patterns |
| `inactive_candidate` | x 0.65 | Long platform inactivity combined with low engagement signals |
| `behavioral_trust_issues` | x 0.60 | Platform engagement anomalies: zero response rate, zero views despite active flag |
| `keyword_stuffing` | x 0.70 | Skill list disproportionate to experience; expert claims without duration |
| `ai_keywords_no_production` | x 0.55 | AI vocabulary without any production deployment evidence in career text |
| `low_quality_profile` | x 0.55 | Thin, unverifiable profile: low completeness, no connections, no verifications |
| `generic_chatgpt_user` | x 0.75 | Template language in summary (regex-detected ChatGPT writing patterns) |

### Compound Penalty Floor

```python
trap_penalty = max(product(penalty_factors), MIN_COMPOUND_PENALTY=0.05)
```

No candidate reaches a score of zero, preserving rank ordering integrity even in adversarial cases.

### Honeypot Exclusion

```python
if trap_penalty < 0.10:
    # excluded from submission — never appears in the top-100
```

In our 100K run: **305 honeypots excluded** (0.3% of the pool).

---

## Explainability Engine

Every candidate in the submission receives a **1-2 sentence, fact-grounded explanation** generated by `src/reasoning.py`.

### Design Principles

- **No hallucinations** — explanations reference only data present in the candidate's profile and score breakdown
- **Rank-band tone** — language adapts to position:
  - Ranks 1-10: confident, superlative framing
  - Ranks 11-30: strong endorsement with nuance
  - Ranks 31-60: positive framing with context
  - Ranks 61-100: honest, appropriately hedged
- **Natural variety** — 6 template variants per tone band; seed derived from `candidate_id` for deterministic output
- **Concern surfacing** — lowest sub-score, inactivity, notice period, and location gaps mentioned when material
- **Grammar safety** — `_apply_template()` applies 9 cleanup regex patterns to eliminate artifacts from empty optional fields

### Example Outputs

```
Rank #1 (score=0.9547):
"A Senior ML Engineer with 7 years of focused retrieval and ranking expertise across
vector search and production recommendation systems — the closest technical match to
the JD's primary ask; strong behavioral signals suggest immediate availability."

Rank #47 (score=0.6812):
"Solid Python and NLP background with tier-1 skill coverage in embeddings and
vector search; limited evidence of end-to-end production ML deployment may temper
expectations for a founding-engineer role."

Rank #91 (score=0.5203):
"Borderline inclusion — covers core Python and some ML skills but lacks the
retrieval-specific depth or startup product-shipping history the JD prioritises;
worth reviewing if top-80 candidates are unavailable."
```

---

## Runtime Compliance

| Constraint | Limit | Our Performance |
|------------|-------|----------------|
| Wall-clock (online step) | <= 5 min | **< 1 second** |
| RAM | <= 16 GB | ~2 GB peak (precomputed path) |
| CPU only | Required | No GPU used anywhere |
| Network access | None | No external calls |
| Disk | <= 5 GB | ~1.8 GB total (data + artifacts) |

**How we achieve < 1 second:** All 100,000 candidate scores are precomputed and stored as a columnar Parquet file. The online `run.py` step loads the pre-scored file (~0.3 s), sorts, selects top-100, generates reasoning strings, and writes the CSV — all in under a second on commodity hardware.

**Precompute step** (`scripts/precompute.py`) runs offline once (~13 min on 100K). It is not counted against the 5-minute online budget.

---

## Installation

### Prerequisites

- Python 3.11+
- ~3 GB disk for data + artifacts
- No GPU required

### Setup

```bash
# 1. Clone
git clone <repo-url>
cd redrob-ranker-ai

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place dataset
cp /path/to/candidates.jsonl    data/raw/candidates.jsonl
cp /path/to/job_description.txt data/raw/job_description.txt
```

---

## Usage

### Step 1 — Precompute (run once, ~13 min)

Scores all 100,000 candidates and writes artifacts to `data/processed/` and `data/artifacts/`.

```bash
python scripts/precompute.py
```

Produces:

- `data/processed/sub_scores.parquet` — composite + 10 sub-scores for all 100K candidates
- `data/processed/candidates_meta.parquet` — metadata for reasoning and display
- `data/artifacts/honeypot_flags.npy` — boolean honeypot mask
- `data/artifacts/candidate_ids.npy` — aligned candidate ID array

### Step 2 — Generate Submission (< 1 second)

```bash
python run.py
# or with explicit paths:
python run.py \
  --candidates data/raw/candidates.jsonl \
  --out outputs/submission.csv \
  --processed data/processed \
  --artifacts data/artifacts
```

Output: `outputs/submission.csv` with columns `candidate_id, rank, score, reasoning`.

### Step 3 — Validate Submission

```bash
python -m src.validate_submission outputs/submission.csv
```

Runs 8 checks: schema, row count (exactly 100), rank sequence (1-100, no gaps), unique IDs, score range [0,1], monotonic scores, non-empty reasoning, UTF-8 encoding.

```
----------------------------------------------------------------
Submission Validator -- Redrob AI Challenge
  File   : outputs/submission.csv
  Checks : 8 passed, 0 failed
----------------------------------------------------------------
  [OK]   Utf8 Encoding         Valid UTF-8
  [OK]   Schema                Schema valid
  [OK]   Row Count             100 rows (expected 100)
  [OK]   Ranks                 Ranks 1-100 valid
  [OK]   Unique Ids            All IDs unique
  [OK]   Score Range           All scores in [0, 1]
  [OK]   Monotonic Scores      Scores non-increasing
  [OK]   Reasoning             All reasoning strings non-empty
----------------------------------------------------------------
  SUBMISSION VALID
----------------------------------------------------------------
```

### Step 4 — Launch Dashboard

```bash
# Professional 5-tab dashboard (recommended)
streamlit run streamlit_app.py

# Basic demo
streamlit run app/demo.py
```

The dashboard opens at `http://localhost:8501`.

| Tab | Contents |
|-----|----------|
| **Rankings** | Searchable top-100 table, download buttons, inline validation |
| **Score Breakdown** | Per-candidate sub-score bars, group stacked chart, heatmap |
| **Trap Risk** | Honeypot/risk counts, detector frequency chart, flagged candidates |
| **Reasoning** | Radar chart, reasoning card, expandable list of all explanations |
| **Dashboard** | Score histogram, location distribution, YoE distribution, population vs top-100 comparison |

Two operating modes (sidebar):

- **Precomputed** — instant load from cached artifacts, shows full 100K statistics
- **Live Demo** — upload any JSON/JSONL (<= 500 candidates) + JD text → runs full pipeline on the fly

### CLI Reference

```
Usage: python run.py [OPTIONS]

Options:
  -c, --candidates PATH    Path to candidates.jsonl  [default: data/raw/candidates.jsonl]
  -o, --out PATH           Output CSV path           [default: outputs/submission.csv]
  --artifacts PATH         Precomputed artifacts dir  [default: data/artifacts]
  --processed PATH         Processed features dir     [default: data/processed]
  --dry-run                Load and validate only; do not write output
  --help                   Show this message and exit.
```

---

## Docker

### Build

```bash
docker build -t redrob-ranker .
```

### Run — Online Ranking Step

Mount your data and outputs directories. Artifacts must have been precomputed beforehand.

```bash
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  redrob-ranker \
  python run.py \
    --candidates data/raw/candidates.jsonl \
    --out outputs/submission.csv
```

### Run — Full Precompute + Ranking

```bash
# Precompute (offline, ~13 min on 100K)
docker run \
  -v $(pwd)/data:/app/data \
  redrob-ranker \
  python scripts/precompute.py

# Then rank (< 1 second)
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  redrob-ranker \
  python run.py --out outputs/submission.csv
```

### Run — Streamlit Dashboard

```bash
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -p 8501:8501 \
  redrob-ranker \
  streamlit run streamlit_app.py --server.address 0.0.0.0
```

Then visit `http://localhost:8501`.

---

## Testing

```bash
# Run all 1,013 tests
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Specific modules
pytest tests/test_scoring.py -v
pytest tests/test_trap_detection.py -v
pytest tests/test_reasoning.py -v
pytest tests/test_validate_submission.py -v
```

### Test Coverage by Module

| Module | Tests | What Is Covered |
|--------|-------|----------------|
| `scoring.py` | 200+ | Sub-score functions, composite formula, determinism, edge cases |
| `trap_detection.py` | 150+ | All 10 detectors, compound penalty, honeypot threshold |
| `behavioral_signals.py` | 100+ | All 18 signals, hard cap, market demand bonus |
| `text_features.py` | 80+ | TF-IDF pipeline, keyword boost, scoring alignment |
| `reasoning.py` | 51 | Helpers, fact grounding, rank bands, concern surfacing, graceful degradation |
| `validate_submission.py` | 61 | 8 check functions, real submission CSV, encoding edge cases |
| Other modules | 370+ | load_data, feature_engineering, ranker, config |

---

## Demo

### Rankings Tab

![Rankings tab showing top-100 candidates with score progress bars and trap penalty indicators](.github/screenshots/tab_rankings.png)

*Top-100 candidates ranked by composite score. Score and trap penalty rendered as inline progress bars. The "Validate Submission" button runs all 8 checks live without leaving the browser.*

---

### Score Breakdown Tab

![Score breakdown showing stacked group contribution bars and per-candidate sub-score mini-bars](.github/screenshots/tab_breakdown.png)

*Stacked bar chart decomposes each candidate's score into the five group contributions. The right panel shows per-candidate sub-score mini-bars and a colour-coded heatmap for the top-N comparison.*

---

### Trap Risk Tab

![Trap risk tab showing flagged candidates and detector frequency chart](.github/screenshots/tab_trap.png)

*Horizontal bar chart shows which of the 10 trap detectors fired most often across the full 100K pool. Flagged candidates table is sortable by penalty severity, making adversarial profiles easy to audit.*

---

### Reasoning Tab

![Reasoning tab showing radar chart and fact-grounded explanation card](.github/screenshots/tab_reasoning.png)

*Per-candidate radar chart spans all 10 sub-scores, immediately revealing which dimensions drove the ranking. Below it, the 1-2 sentence fact-grounded explanation is rendered in a styled card. An expandable list shows all top-20 reasoning strings for quick review.*

---

### Dashboard Tab

![Dashboard showing score histogram, location distribution, YoE distribution, and sub-score comparison](.github/screenshots/tab_dashboard.png)

*Score histogram with p99 and p99.9 markers shows the score distribution across all 100K candidates. The population-vs-top-100 sub-score comparison reveals exactly what separates shortlisted candidates from the field.*

---

## AI Tools Declaration

This project was built with the assistance of AI tools, in accordance with the Redrob Challenge guidelines on transparency.

| Tool | Usage |
|------|-------|
| **Claude Sonnet (Anthropic)** | Architecture design, code generation, test authoring, README writing, debugging |
| **GitHub Copilot** | Inline autocomplete during development |

All generated code was reviewed, tested against 1,013 automated tests, and validated against the hackathon constraints by the team. The scoring weights, trap detector logic, behavioral signal model, and ranking design reflect deliberate domain decisions made by the team — not defaults produced by any AI tool. The final submission (`outputs/submission.csv`) was produced by deterministic, reproducible code from this repository.

---

## Project Metadata

| Field | Value |
|-------|-------|
| **Team** | INDIARUNS |
| **Challenge** | Redrob Data & AI Challenge 2026 |
| **Submission file** | `outputs/submission.csv` |
| **Submission command** | `python run.py --out outputs/submission.csv` |
| **Online runtime** | < 1 second |
| **Tests passing** | 1,013 |
| **Top candidate** | CAND_0018499, score = 0.9547 |
| **Honeypots excluded** | 305 of 100,000 |
| **Python version** | 3.11 |
| **Hardware** | CPU-only (no GPU) |
