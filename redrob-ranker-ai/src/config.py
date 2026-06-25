"""
config.py — All paths, model names, weights, and hyperparameters.

Single source of truth. Change weights here; nothing else needs editing.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
ARTIFACTS_DIR   = DATA_DIR / "artifacts"
OUTPUTS_DIR     = ROOT / "outputs"

CANDIDATES_FILE        = RAW_DIR / "candidates.jsonl"
JD_FILE                = RAW_DIR / "job_description.txt"
CANDIDATE_SCHEMA_FILE  = RAW_DIR / "candidate_schema.json"
SAMPLE_CANDIDATES_FILE = RAW_DIR / "sample_candidates.json"

EMBEDDINGS_FILE     = ARTIFACTS_DIR / "embeddings_combined.npy"
CANDIDATE_IDS_FILE  = ARTIFACTS_DIR / "candidate_ids.npy"
HONEYPOT_FLAGS_FILE = ARTIFACTS_DIR / "honeypot_flags.npy"
JD_EMBEDDING_FILE   = ARTIFACTS_DIR / "jd_embedding.npy"
FEATURES_FILE       = PROCESSED_DIR / "features.parquet"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
EMBEDDING_BATCH = 256          # batch size for CPU inference

# ---------------------------------------------------------------------------
# Composite score weights (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_CAREER_RELEVANCE  = 0.35
WEIGHT_SKILL_DEPTH       = 0.25
WEIGHT_BEHAVIORAL        = 0.20
WEIGHT_LOCATION          = 0.12
WEIGHT_EXPERIENCE_FIT    = 0.08

assert abs(
    WEIGHT_CAREER_RELEVANCE + WEIGHT_SKILL_DEPTH + WEIGHT_BEHAVIORAL
    + WEIGHT_LOCATION + WEIGHT_EXPERIENCE_FIT - 1.0
) < 1e-9, "Weights must sum to 1.0"

# ---------------------------------------------------------------------------
# Career relevance sub-weights
# ---------------------------------------------------------------------------
CAREER_SEMANTIC_WEIGHT      = 0.50
CAREER_COMPANY_TYPE_WEIGHT  = 0.30
CAREER_TITLE_RECENCY_WEIGHT = 0.20

# ---------------------------------------------------------------------------
# Skill depth sub-weights
# ---------------------------------------------------------------------------
SKILL_CORE_MATCH_WEIGHT        = 0.40
SKILL_COHERENCE_WEIGHT         = 0.25
SKILL_ASSESSMENT_WEIGHT        = 0.20
SKILL_DURATION_QUALITY_WEIGHT  = 0.15

# ---------------------------------------------------------------------------
# Behavioral multiplier sub-weights
# ---------------------------------------------------------------------------
BEHAVIORAL_AVAILABILITY_WEIGHT = 0.45
BEHAVIORAL_ENGAGEMENT_WEIGHT   = 0.35
BEHAVIORAL_TRUST_WEIGHT        = 0.20

# ---------------------------------------------------------------------------
# Experience fit (Gaussian params)
# ---------------------------------------------------------------------------
EXP_FIT_MU    = 7.0    # peak years of experience
EXP_FIT_SIGMA = 2.5    # spread

# ---------------------------------------------------------------------------
# Disqualifier penalty factors (multiplicative)
# ---------------------------------------------------------------------------
PENALTY_CONSULTING_ONLY         = 0.15
PENALTY_PURE_RESEARCH           = 0.20
PENALTY_NONTECHNICAL_TITLE      = 0.25
PENALTY_OUTSIDE_INDIA_NO_RELOC  = 0.40
PENALTY_NOTICE_LONG             = 0.60   # notice > 120 days
PENALTY_LANGCHAIN_NO_TIER1      = 0.70

# ---------------------------------------------------------------------------
# Honeypot detection thresholds
# ---------------------------------------------------------------------------
HONEYPOT_EXPERT_ZERO_DURATION_THRESHOLD = 3   # ≥N expert skills with 0 months
HONEYPOT_CAREER_OVERLAP_FACTOR          = 1.5  # career months > exp * 12 * factor

# ---------------------------------------------------------------------------
# Behavioral decay
# ---------------------------------------------------------------------------
AVAILABILITY_DECAY_HALFLIFE_DAYS = 30    # exponential decay halflife
INACTIVE_CAP_THRESHOLD_DAYS      = 90    # days since last active to cap score
INACTIVE_CAP_VALUE               = 0.30  # max behavioral multiplier if inactive

# ---------------------------------------------------------------------------
# Location preferences
# ---------------------------------------------------------------------------
PREFERRED_COUNTRIES = {"India"}
PREFERRED_CITIES    = {
    "Noida", "Pune", "Hyderabad", "Mumbai", "Delhi", "Bengaluru", "Bangalore",
    "Gurgaon", "Gurugram", "Delhi NCR", "New Delhi",
}

# ---------------------------------------------------------------------------
# Consulting firm list (entire career here = disqualifier)
# ---------------------------------------------------------------------------
CONSULTING_FIRMS = {
    "TCS", "Tata Consultancy Services",
    "Infosys",
    "Wipro",
    "Accenture",
    "Cognizant", "Cognizant Technology Solutions",
    "Capgemini",
    "HCL", "HCL Technologies",
    "Tech Mahindra",
    "Hexaware",
    "Mphasis",
    "Persistent Systems",
    "L&T Infotech", "LTIMindtree",
}

# ---------------------------------------------------------------------------
# AI/ML skill taxonomy
# ---------------------------------------------------------------------------
TIER1_SKILLS = {
    # Retrieval / search
    "embeddings", "vector search", "hybrid search", "dense retrieval",
    "sentence-transformers", "sentence transformers", "sbert",
    "bge", "e5", "openai embeddings",
    # Vector databases
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch", "chroma",
    # Ranking / evaluation
    "ndcg", "mrr", "map", "learning to rank", "ltr", "ranking",
    "retrieval", "information retrieval",
    # Core
    "python", "pytorch", "machine learning", "deep learning",
    "nlp", "natural language processing",
}

TIER2_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "rag", "retrieval augmented generation",
    "xgboost", "lightgbm",
    "a/b testing", "ab testing",
    "distributed systems", "large scale inference",
    "recommendation systems", "recommender",
    "transformers", "bert", "gpt",
    "mlops", "kubeflow", "mlflow", "weights & biases",
}

PENALTY_SKILLS = {
    "langchain",   # red flag if ONLY langchain without tier1
}

# ---------------------------------------------------------------------------
# Title classification (lowercase keyword matching)
# ---------------------------------------------------------------------------
NONTECHNICAL_TITLE_KEYWORDS = {
    "hr manager", "hr", "human resources", "marketing manager", "marketing",
    "sales executive", "sales", "accountant", "accounting", "content writer",
    "graphic designer", "customer support", "operations manager", "operations",
    "business analyst", "project manager", "civil engineer", "mechanical engineer",
    "finance manager", "finance", "supply chain", "procurement",
}

TECHNICAL_AI_TITLE_KEYWORDS = {
    "ml engineer", "machine learning engineer", "ai engineer", "data scientist",
    "nlp engineer", "research scientist", "applied scientist", "deep learning engineer",
    "research engineer", "ai researcher", "senior ml", "staff ml", "principal ml",
    "junior ml engineer", "senior machine learning",
}

ADJACENT_TECHNICAL_TITLE_KEYWORDS = {
    "data engineer", "backend engineer", "analytics engineer", "software engineer",
    "platform engineer", "python developer", "full stack", "software developer",
    "data analyst", "research analyst",
}

# ---------------------------------------------------------------------------
# Reference date — used for days_since_last_active and other time deltas
# Today's date: 2026-06-15
# ---------------------------------------------------------------------------
REFERENCE_DATE = date(2026, 6, 15)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
LOAD_SKIP_INVALID   = True    # skip malformed records instead of crashing
LOAD_SHOW_PROGRESS  = True    # show tqdm progress bar during load
MAX_CANDIDATE_TEXT_CHARS = 2048  # truncate combined text to keep embeddings focused
