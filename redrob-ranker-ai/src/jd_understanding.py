"""
jd_understanding.py — Structured extraction of the Job Description

Everything the ranker needs to know about what this JD actually wants
is centralised here as plain Python dicts and scoring functions.

Public constants (importable as data)
--------------------------------------
    MUST_HAVE_SKILLS        — 4 mandatory skill groups
    NICE_TO_HAVE_SKILLS     — 5 preferred skill groups
    VECTOR_DB_PATTERNS      — vector database / ANN index signals
    RETRIEVAL_PATTERNS      — embedding retrieval signals
    RANKING_PATTERNS        — ranking / LTR / matching signals
    EVALUATION_PATTERNS     — offline + online eval signals
    PRODUCTION_ML_PATTERNS  — "deployed to real users" signals
    SHIPPER_PATTERNS        — scrappy product-engineering culture signals
    STARTUP_PATTERNS        — Series A / founding-team culture signals
    LOCATION_REQUIREMENTS   — geography constraints
    EXPERIENCE_REQUIREMENTS — YoE + company-type expectations
    SALARY_EXPECTATIONS     — derived from market + JD context
    DISQUALIFIER_PATTERNS   — hard and soft elimination signals
    JD_REQUIREMENTS         — aggregated master dict (all of the above)

Public functions
----------------
    get_jd_embedding_text() -> str
    score_skills_match(skill_names, career_text, summary_text) -> dict[str,float]
    detect_disqualifiers(flat) -> list[dict]
    score_location_match(country, location, willing_to_relocate) -> float
    score_experience_fit(years_of_experience) -> float
    score_notice_period(notice_period_days) -> float
"""

from __future__ import annotations

import math
import re
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# § 1  JD Metadata
# ─────────────────────────────────────────────────────────────────────────────

JD_METADATA: dict[str, Any] = {
    "title":        "Senior AI Engineer — Founding Team",
    "company":      "Redrob AI",
    "stage":        "Series A",
    "domain":       "AI-native talent intelligence platform",
    "employment":   "Full-time",
    "location":     "Pune / Noida, India (Hybrid)",
    "yoe_stated":   "5–9 years",
    "yoe_ideal":    "6–8 years",
    "notice_ideal": 30,          # days
    "notice_max":   90,          # days (bar gets higher beyond this)
    "visa_sponsor": False,
    "remote":       False,
    "hybrid":       True,
    "team_size_now":   4,
    "team_size_target": 12,
}

# ─────────────────────────────────────────────────────────────────────────────
# § 2  Must-Have Skills  ("Things you absolutely need")
# ─────────────────────────────────────────────────────────────────────────────
# Each group is a skill cluster where the JD requires demonstrated production
# experience, not just awareness.  Matching ANY keyword in the cluster counts
# as evidence for that group; the scorer combines all four groups.
# ─────────────────────────────────────────────────────────────────────────────

MUST_HAVE_SKILLS: dict[str, dict[str, Any]] = {

    "embeddings_retrieval": {
        "keywords": [
            "sentence-transformers", "sentence transformers", "sbert",
            "bge", "e5", "openai embeddings", "cohere embeddings",
            "embedding model", "embeddings", "text embeddings",
            "bi-encoder", "biencoder", "dense retrieval",
            "semantic search", "semantic similarity",
            "vector retrieval", "approximate nearest neighbor", "ann",
            "embedding drift", "index refresh", "retrieval quality",
        ],
        "phrases": [
            "embeddings-based retrieval", "embedding based retrieval",
            "retrieval system", "deployed to real users",
            "production retrieval", "sentence transformer",
        ],
        "threshold": 2,      # need ≥2 matches for full score
        "weight": 1.0,
        "required": True,
        "jd_quote": (
            "Production experience with embeddings-based retrieval systems "
            "(sentence-transformers, OpenAI embeddings, BGE, E5, or similar) "
            "deployed to real users."
        ),
    },

    "vector_databases": {
        "keywords": [
            "pinecone", "weaviate", "qdrant", "milvus", "faiss",
            "opensearch", "elasticsearch", "chroma", "chromadb",
            "pgvector", "redis vector", "typesense",
            "vector database", "vector db", "vector store",
            "hybrid search", "ann index", "hnsw",
        ],
        "phrases": [
            "vector database", "hybrid search infrastructure",
            "vector search", "approximate nearest neighbor search",
            "knn search", "k-nn search",
        ],
        "threshold": 1,      # any vector DB counts — they're alternatives
        "weight": 1.0,
        "required": True,
        "jd_quote": (
            "Production experience with vector databases or hybrid search "
            "infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, "
            "Elasticsearch, FAISS, or something similar."
        ),
    },

    "python_engineering": {
        "keywords": [
            "python", "pytorch", "numpy", "pandas", "scikit-learn",
            "fastapi", "flask", "pydantic", "sqlalchemy", "asyncio",
            "pytest", "unit testing", "code quality", "type hints",
        ],
        "phrases": [
            "strong python", "production python", "python code quality",
        ],
        "threshold": 1,
        "weight": 0.8,
        "required": True,
        "jd_quote": "Strong Python. Yes really, we care about code quality.",
    },

    "ranking_evaluation": {
        "keywords": [
            "ndcg", "mrr", "map", "precision", "recall",
            "a/b testing", "ab testing", "a/b test",
            "offline evaluation", "online evaluation",
            "evaluation framework", "evaluation infrastructure",
            "ranking metrics", "relevance judgment",
            "click-through rate", "ctr", "engagement metrics",
            "offline benchmark",
        ],
        "phrases": [
            "ndcg evaluation", "offline-to-online correlation",
            "a/b test interpretation", "evaluation framework for ranking",
            "ranking evaluation", "search evaluation",
        ],
        "threshold": 2,
        "weight": 1.0,
        "required": True,
        "jd_quote": (
            "Hands-on experience designing evaluation frameworks for ranking "
            "systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test "
            "interpretation."
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# § 3  Nice-to-Have Skills  ("Things we'd like but won't reject you for")
# ─────────────────────────────────────────────────────────────────────────────

NICE_TO_HAVE_SKILLS: dict[str, dict[str, Any]] = {

    "llm_finetuning": {
        "keywords": [
            "lora", "qlora", "peft", "fine-tuning", "finetuning", "fine tuning",
            "sft", "rlhf", "dpo", "instruction tuning",
            "hugging face", "transformers", "llm", "large language model",
        ],
        "phrases": [
            "llm fine-tuning", "fine-tune", "parameter efficient",
            "low-rank adaptation", "quantized lora",
        ],
        "threshold": 1,
        "weight": 0.7,
        "required": False,
        "jd_quote": "LLM fine-tuning experience (LoRA, QLoRA, PEFT)",
    },

    "learning_to_rank": {
        "keywords": [
            "learning to rank", "ltr", "xgboost", "lightgbm", "lambdamart",
            "listwise", "pairwise", "pointwise", "ranknet",
            "gradient boosting", "gbm", "feature-based ranking",
        ],
        "phrases": [
            "learning-to-rank", "xgboost ranking", "neural ltr",
            "xgboost-based", "learning to rank model",
        ],
        "threshold": 1,
        "weight": 0.6,
        "required": False,
        "jd_quote": "Experience with learning-to-rank models (XGBoost-based or neural)",
    },

    "hrtech_marketplace": {
        "keywords": [
            "hr-tech", "hrtech", "hr tech", "recruiting", "recruitment",
            "marketplace", "talent acquisition", "ats", "candidate search",
            "job board", "talent platform", "people analytics",
        ],
        "phrases": [
            "recruiting technology", "talent intelligence", "job matching",
            "candidate ranking", "hr platform",
        ],
        "threshold": 1,
        "weight": 0.5,
        "required": False,
        "jd_quote": "Prior exposure to HR-tech, recruiting tech, or marketplace products",
    },

    "distributed_systems": {
        "keywords": [
            "distributed systems", "kafka", "spark", "ray", "dask",
            "kubernetes", "docker", "mlops", "kubeflow", "airflow",
            "large-scale inference", "model serving", "triton", "torchserve",
            "latency optimization", "throughput optimization",
        ],
        "phrases": [
            "distributed inference", "large scale ml", "ml infrastructure",
            "inference optimization", "model optimization",
        ],
        "threshold": 2,
        "weight": 0.5,
        "required": False,
        "jd_quote": "Background in distributed systems or large-scale inference optimization",
    },

    "open_source_ml": {
        "keywords": [
            "open source", "github", "open-source", "oss",
            "hugging face", "kaggle", "research paper", "arxiv",
            "blog post", "conference", "mlops", "weights & biases",
        ],
        "phrases": [
            "open-source contributions", "ml community", "published paper",
            "open source ai", "contributed to",
        ],
        "threshold": 1,
        "weight": 0.4,
        "required": False,
        "jd_quote": "Open-source contributions in the AI/ML space",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# § 4  Technical Domain Patterns
# ─────────────────────────────────────────────────────────────────────────────

VECTOR_DB_PATTERNS: dict[str, Any] = {
    "keywords": [
        "pinecone", "weaviate", "qdrant", "milvus", "faiss",
        "opensearch", "elasticsearch", "chroma", "chromadb",
        "pgvector", "redis vector", "typesense", "vespa",
        "annoy", "scann", "nmslib", "hnswlib",
        "vector database", "vector db", "vector store",
        "ann", "hnsw", "ivf", "product quantization", "pq",
    ],
    "phrases": [
        "vector search", "approximate nearest neighbor",
        "semantic search index", "vector index",
        "hybrid search infrastructure",
    ],
    "synonyms": {
        "faiss": ["facebook ai similarity search", "faiss index"],
        "elasticsearch": ["elastic search", "opensearch"],
        "vector database": ["vector db", "vector store", "ann database"],
    },
    "weight": 1.0,
    "required": True,
    "description": "Operational experience with vector database or hybrid search tech.",
}

RETRIEVAL_PATTERNS: dict[str, Any] = {
    "keywords": [
        "retrieval", "information retrieval", "semantic search",
        "sentence-transformers", "sentence transformers", "sbert",
        "bge", "e5", "openai embeddings", "text embeddings",
        "dense retrieval", "sparse retrieval", "hybrid retrieval",
        "bm25", "tf-idf", "tfidf", "inverted index",
        "bi-encoder", "cross-encoder", "reranking", "re-ranking",
        "embedding model", "embedding drift", "index refresh",
        "query expansion", "chunking", "rag",
    ],
    "phrases": [
        "embeddings-based retrieval", "retrieval system",
        "retrieval quality regression", "dense + sparse",
        "dense and sparse", "hybrid search",
        "retrieval augmented generation",
    ],
    "synonyms": {
        "retrieval": ["search", "information retrieval", "ir"],
        "bm25": ["okapi bm25", "bm 25", "sparse retrieval"],
        "reranking": ["re-ranking", "cross-encoder reranking", "llm reranker"],
    },
    "weight": 1.0,
    "required": True,
    "description": "Experience building and operating retrieval pipelines.",
}

RANKING_PATTERNS: dict[str, Any] = {
    "keywords": [
        "ranking", "ranker", "learning to rank", "ltr",
        "recommendation", "recommender system", "recommendation engine",
        "candidate matching", "jd matching", "relevance scoring",
        "candidate ranking", "search ranking", "result ranking",
        "personalization", "feed ranking", "xgboost", "lightgbm",
        "lambdamart", "ranknet", "listnet", "neural ranking",
        "pointwise", "pairwise", "listwise",
    ],
    "phrases": [
        "ranking system", "ranking and retrieval", "ranking pipeline",
        "candidate-jd matching", "end-to-end ranking",
        "recruiter search ranking", "search and ranking",
    ],
    "synonyms": {
        "ranking": ["ranker", "rank system"],
        "recommendation": ["recommender", "recommendation system"],
        "learning to rank": ["ltr", "learning-to-rank"],
    },
    "weight": 1.0,
    "required": True,
    "description": "Building and owning ranking, search, or recommendation systems.",
}

EVALUATION_PATTERNS: dict[str, Any] = {
    "keywords": [
        "ndcg", "normalized discounted cumulative gain",
        "mrr", "mean reciprocal rank",
        "map", "mean average precision",
        "precision@k", "recall@k", "f1",
        "a/b testing", "ab test", "a/b test",
        "online evaluation", "offline evaluation",
        "evaluation framework", "eval framework",
        "relevance judgment", "click-through rate", "ctr",
        "engagement rate", "dwell time",
        "offline benchmark", "live evaluation",
    ],
    "phrases": [
        "offline-to-online correlation", "a/b test interpretation",
        "evaluation infrastructure", "evaluation pipeline",
        "ranking evaluation", "search evaluation metrics",
        "ndcg@10", "ndcg@50", "map evaluation",
    ],
    "synonyms": {
        "ndcg": ["nDCG", "normalized dcg", "discounted cumulative gain"],
        "a/b testing": ["ab testing", "split testing", "online experiment"],
        "map": ["mean average precision", "average precision"],
    },
    "weight": 1.0,
    "required": True,
    "description": "Ability to design and interpret ranking evaluation frameworks.",
}

PRODUCTION_ML_PATTERNS: dict[str, Any] = {
    "keywords": [
        "production", "deployed", "shipped", "at scale",
        "real users", "live traffic", "serving",
        "inference", "latency", "throughput",
        "monitoring", "observability", "drift detection",
        "model registry", "mlflow", "weights & biases",
        "ci/cd", "ml pipeline", "feature store",
        "online prediction", "real-time scoring",
    ],
    "phrases": [
        "deployed to production", "production deployment",
        "shipped to users", "production system",
        "real-world deployment", "end-to-end ml",
        "production ml", "ml at scale",
        "serving at scale", "low-latency inference",
        "production-grade", "product company",
    ],
    "negative_keywords": [
        "academic", "research lab", "research-only",
        "thesis", "dissertation", "benchmark only",
    ],
    "weight": 1.0,
    "required": True,
    "description": "Evidence of shipping ML systems to real users in production.",
}

# ─────────────────────────────────────────────────────────────────────────────
# § 5  Culture / Mindset Patterns
# ─────────────────────────────────────────────────────────────────────────────

SHIPPER_PATTERNS: dict[str, Any] = {
    "keywords": [
        "shipped", "launched", "deployed", "delivered", "built and deployed",
        "v1", "v2", "mvp", "prototype", "fast iteration",
        "product", "users", "user feedback", "metrics",
        "impact", "business impact", "revenue impact",
        "scrappy", "pragmatic", "iterate",
    ],
    "phrases": [
        "shipped to users", "launched in production",
        "built from scratch", "shipped a ranking system",
        "delivered end-to-end", "built and shipped",
        "improved metrics", "drove engagement",
        "working but not great", "learn from real users",
        "ship fast", "product-engineering",
    ],
    "anti_phrases": [
        "only prototypes", "never deployed", "research only",
        "academic setting", "theoretical",
    ],
    "weight": 0.7,
    "required": False,
    "description": (
        "Scrappy product-engineering attitude — willing to ship working but "
        "suboptimal solutions to learn from real users."
    ),
}

STARTUP_PATTERNS: dict[str, Any] = {
    "keywords": [
        "startup", "early stage", "series a", "series b",
        "founding team", "greenfield", "0 to 1",
        "ambiguous", "autonomy", "ownership", "self-directed",
        "cross-functional", "full-stack", "generalist",
        "async", "remote-first", "fast-paced",
    ],
    "phrases": [
        "early stage startup", "founding engineer",
        "product company", "growth stage",
        "built the team", "wore many hats",
        "0-to-1 product", "built from scratch",
        "small team", "fast-moving team",
    ],
    "weight": 0.4,
    "required": False,
    "description": (
        "Series A founding-team culture: async-first, disagree-then-commit, "
        "move fast, 3+ year commitment expected."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# § 6  Practical Requirements
# ─────────────────────────────────────────────────────────────────────────────

LOCATION_REQUIREMENTS: dict[str, Any] = {
    "required_countries":      ["India"],
    "preferred_cities":        ["Noida", "Pune"],
    "acceptable_cities": [
        "Hyderabad", "Mumbai", "Delhi", "Delhi NCR", "New Delhi",
        "Bengaluru", "Bangalore", "Gurgaon", "Gurugram",
    ],
    "accepts_relocation":      True,
    "visa_sponsorship":        False,
    "work_model":              "Hybrid",
    "office_locations":        ["Noida", "Pune"],
    "quarterly_travel":        True,
    "scores": {
        "preferred_country_preferred_city": 1.00,
        "preferred_country_acceptable_city": 0.80,
        "preferred_country_other_city":      0.65,
        "outside_country_willing_relocate":  0.45,
        "outside_country_not_relocate":      0.10,
    },
    "jd_quote": (
        "Location: Pune/Noida-preferred but flexible. Candidates in Hyderabad, "
        "Pune, Mumbai, Delhi NCR welcome to apply. Outside India: case-by-case, "
        "but we don't sponsor work visas."
    ),
}

EXPERIENCE_REQUIREMENTS: dict[str, Any] = {
    "stated_range_min":   5,
    "stated_range_max":   9,
    "ideal_range_min":    6,
    "ideal_range_max":    8,
    "gaussian_mu":        7.0,    # peak of Gaussian fit
    "gaussian_sigma":     2.5,    # spread of Gaussian fit
    "applied_ml_min_yrs": 4,      # minimum years in applied ML/AI roles
    "applied_ml_ideal":   5,      # ideal years in applied ML/AI at product cos
    "product_company_required": True,
    "production_code_max_gap_months": 18,  # gap = disqualifier
    "hard_min_yrs":       3,      # below this → very unlikely
    "hard_max_yrs":       16,     # above this → likely overqualified for culture
    "jd_quote": (
        "6-8 years total experience, of which 4-5 are in applied ML/AI roles "
        "at product companies (not pure services)."
    ),
}

SALARY_EXPECTATIONS: dict[str, Any] = {
    "currency":           "INR",
    "unit":               "LPA",   # Lakhs Per Annum
    "market_range_min":   30,      # LPA — derived from Series A Sr AI Eng market
    "market_range_max":   80,      # LPA — upper end for founding team hire
    "sweet_spot_min":     40,
    "sweet_spot_max":     65,
    "notice_budget_days": 30,      # company willing to buy out up to 30 days
    "notes": (
        "JD does not state compensation explicitly. Derived from: Series A "
        "company, Sr AI Engineer title, India market, 6-8 YoE. "
        "Candidates expecting >80 LPA may negotiate well; candidates asking "
        "<25 LPA may be under-levelled or signalling low confidence."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# § 7  Disqualifier Patterns
# ─────────────────────────────────────────────────────────────────────────────
# Each disqualifier has:
#   penalty_factor  : multiplied into final score (0.0 = elimination)
#   detectable      : can we detect this from flat candidate dict?
#   detection_fields: which flat dict fields to inspect
# ─────────────────────────────────────────────────────────────────────────────

DISQUALIFIER_PATTERNS: dict[str, dict[str, Any]] = {

    "pure_research_no_production": {
        "penalty_factor":     0.15,
        "detectable":         True,
        "detection_fields":   ["career_titles", "career_descriptions_text", "current_title"],
        "title_signals":      ["researcher", "research scientist", "phd student",
                               "postdoc", "postdoctoral", "research intern"],
        "anti_title_signals": ["engineer", "ml engineer", "senior", "staff"],
        "jd_quote": (
            "If you've spent your career in pure research environments "
            "(academic labs, research-only roles) without any production deployment "
            "— we will not move forward."
        ),
    },

    "langchain_only_no_tier1": {
        "penalty_factor":     0.70,
        "detectable":         True,
        "detection_fields":   ["skill_names", "tier1_skill_count"],
        "trigger_skill":      "langchain",
        "require_tier1_min":  1,  # must have ≥1 tier-1 skill alongside LangChain
        "jd_quote": (
            "If your 'AI experience' consists primarily of recent (under 12 months) "
            "projects using LangChain to call OpenAI — we will probably not move forward."
        ),
    },

    "consulting_only_career": {
        "penalty_factor":     0.15,
        "detectable":         True,
        "detection_fields":   ["is_consulting_only"],
        "consulting_firms": [
            "TCS", "Tata Consultancy Services", "Infosys", "Wipro", "Accenture",
            "Cognizant", "Capgemini", "HCL", "Tech Mahindra", "Hexaware",
            "Mphasis", "Persistent Systems", "L&T Infotech", "LTIMindtree",
        ],
        "jd_quote": (
            "People who have only worked at consulting firms (TCS, Infosys, Wipro, "
            "Accenture, Cognizant, Capgemini, etc.) in their entire career."
        ),
    },

    "nontechnical_primary_title": {
        "penalty_factor":     0.25,
        "detectable":         True,
        "detection_fields":   ["current_title", "most_recent_title"],
        "nontechnical_titles": [
            "hr manager", "human resources", "marketing manager", "marketing",
            "sales executive", "sales manager", "accountant", "accounting",
            "content writer", "graphic designer", "customer support",
            "operations manager", "operations", "business analyst",
            "project manager", "civil engineer", "mechanical engineer",
            "supply chain", "procurement", "finance manager",
        ],
        "jd_quote": (
            "A candidate who has all the AI keywords listed as skills but whose "
            "title is 'Marketing Manager' is not a fit, no matter how perfect "
            "their skill list looks."
        ),
    },

    "cv_speech_robotics_no_nlp": {
        "penalty_factor":     0.50,
        "detectable":         True,
        "detection_fields":   ["skill_names", "tier1_skill_count"],
        "cv_speech_keywords": [
            "computer vision", "image classification", "object detection",
            "speech recognition", "tts", "text-to-speech", "asr",
            "robotics", "ros", "slam", "autonomous", "opencv",
        ],
        "nlp_ir_keywords": [
            "nlp", "natural language processing", "text", "retrieval",
            "embedding", "transformer", "bert", "gpt", "llm",
        ],
        "jd_quote": (
            "People whose primary expertise is computer vision, speech, or "
            "robotics without significant NLP/IR exposure."
        ),
    },

    "title_chaser": {
        "penalty_factor":     0.60,
        "detectable":         True,
        "detection_fields":   ["n_career_roles", "total_career_months"],
        "avg_tenure_threshold_months": 18,  # avg < 18 months + multiple companies = flag
        "min_roles_to_check": 3,            # only check if ≥3 roles
        "jd_quote": (
            "Title-chasers. If your career trajectory shows you optimizing for "
            "'Senior' → 'Staff' → 'Principal' titles by switching companies "
            "every 1.5 years, we're not a fit."
        ),
    },

    "outside_india_no_relocation": {
        "penalty_factor":     0.40,
        "detectable":         True,
        "detection_fields":   ["country", "willing_to_relocate"],
        "jd_quote": (
            "Outside India: case-by-case, but we don't sponsor work visas."
        ),
    },

    "long_notice_period": {
        "penalty_factor":     0.70,   # soft — JD says "bar gets higher"
        "detectable":         True,
        "detection_fields":   ["notice_period_days"],
        "threshold_days":     90,
        "jd_quote": (
            "30+ day notice candidates are still in scope but the bar gets higher."
        ),
    },

    "inactive_not_available": {
        "penalty_factor":     0.30,
        "detectable":         True,
        "detection_fields":   ["days_since_last_active", "open_to_work_flag",
                               "recruiter_response_rate"],
        "inactive_days_threshold":     90,
        "low_response_rate_threshold": 0.10,
        "jd_quote": (
            "A perfect-on-paper candidate who hasn't logged in for 6 months "
            "and has a 5% recruiter response rate is, for hiring purposes, "
            "not actually available."
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# § 8  Master aggregation
# ─────────────────────────────────────────────────────────────────────────────

JD_REQUIREMENTS: dict[str, Any] = {
    "metadata":              JD_METADATA,
    "must_have_skills":      MUST_HAVE_SKILLS,
    "nice_to_have_skills":   NICE_TO_HAVE_SKILLS,
    "vector_db_patterns":    VECTOR_DB_PATTERNS,
    "retrieval_patterns":    RETRIEVAL_PATTERNS,
    "ranking_patterns":      RANKING_PATTERNS,
    "evaluation_patterns":   EVALUATION_PATTERNS,
    "production_ml_patterns": PRODUCTION_ML_PATTERNS,
    "shipper_patterns":      SHIPPER_PATTERNS,
    "startup_patterns":      STARTUP_PATTERNS,
    "location_requirements": LOCATION_REQUIREMENTS,
    "experience_requirements": EXPERIENCE_REQUIREMENTS,
    "salary_expectations":   SALARY_EXPECTATIONS,
    "disqualifier_patterns": DISQUALIFIER_PATTERNS,
}

# ─────────────────────────────────────────────────────────────────────────────
# § 9  JD Embedding Text
# ─────────────────────────────────────────────────────────────────────────────
# Distilled semantic description of the ideal candidate for embedding.
# This gets embedded once and used as the query vector for all candidates.
# It intentionally captures INTENT (what the JD means) not just KEYWORDS.
# ─────────────────────────────────────────────────────────────────────────────

_JD_EMBEDDING_TEXT = """
Senior AI Engineer with 6-8 years of experience building production machine learning systems.
Deep expertise in embeddings-based retrieval and semantic search.
Has shipped end-to-end ranking, search, or recommendation systems to real users at meaningful scale.
Strong hands-on experience with vector databases such as FAISS, Elasticsearch, Pinecone, Weaviate, Qdrant or Milvus.
Expert in sentence-transformers, BGE, or E5 embedding models deployed in production.
Has handled embedding drift, index refresh, and retrieval quality regression in live systems.
Designed evaluation frameworks using NDCG, MRR, MAP, offline-to-online correlation, and A/B testing.
Strong Python engineering skills with production code quality.
Applied machine learning and NLP engineering at product companies, not pure consulting or research.
Experience with hybrid retrieval combining BM25 sparse search with dense vector search.
Understands LLM-based reranking and when to fine-tune versus prompt.
Values shipping working systems over perfect architecture.
Comfortable in a fast-moving early-stage startup environment.
Located in India, preferably Noida, Pune, Hyderabad, Mumbai, or Delhi NCR.
Available within 30 days.
Actively seeking new opportunities.
""".strip()


def get_jd_embedding_text() -> str:
    """Return the optimised JD text for embedding model input."""
    return _JD_EMBEDDING_TEXT


# ─────────────────────────────────────────────────────────────────────────────
# § 10  Scoring Functions
# ─────────────────────────────────────────────────────────────────────────────

def _count_keyword_hits(
    keywords: list[str],
    phrases: list[str],
    skills_lower: frozenset[str],
    text_lower: str,
) -> int:
    """Count how many keywords / phrases appear in skills or free text."""
    hits = sum(1 for k in keywords if k in skills_lower or k in text_lower)
    hits += sum(1 for p in phrases if p in text_lower)
    return hits


def score_skills_match(
    skill_names: list[str],
    career_text: str = "",
    summary_text: str = "",
) -> dict[str, float]:
    """
    Score how well a candidate's declared skills + career text match the JD.

    Each returned score is in [0, 1].

    Parameters
    ----------
    skill_names  : Candidate's skill names (from flat["skill_names"]).
    career_text  : Concatenated career history descriptions.
    summary_text : Profile summary.

    Returns
    -------
    dict with keys:
        must_have_<group>      — score per must-have group (0-1)
        nice_<group>           — score per nice-to-have group (0-1)
        vector_db              — vector DB pattern match
        retrieval              — retrieval pattern match
        ranking                — ranking pattern match
        evaluation             — evaluation pattern match
        production_ml          — production ML evidence
        overall_must_have      — weighted average of 4 must-have groups (0-1)
        overall_nice           — average of nice-to-have groups (0-1)
        composite_skill_score  — combined score (0-1)
    """
    skills_lower   = frozenset(s.lower() for s in skill_names)
    text_lower     = (career_text + " " + summary_text).lower()
    scores: dict[str, float] = {}

    # ── Must-have groups ─────────────────────────────────────────────────────
    must_scores: list[float] = []
    for group_name, group in MUST_HAVE_SKILLS.items():
        hits = _count_keyword_hits(
            group["keywords"], group.get("phrases", []), skills_lower, text_lower
        )
        threshold = max(group["threshold"], 1)
        s = min(1.0, hits / threshold)
        scores[f"must_have_{group_name}"] = round(s, 4)
        must_scores.append(s * group["weight"])

    scores["overall_must_have"] = round(
        sum(must_scores) / sum(g["weight"] for g in MUST_HAVE_SKILLS.values()), 4
    )

    # ── Nice-to-have groups ───────────────────────────────────────────────────
    nice_scores: list[float] = []
    for group_name, group in NICE_TO_HAVE_SKILLS.items():
        hits = _count_keyword_hits(
            group["keywords"], group.get("phrases", []), skills_lower, text_lower
        )
        threshold = max(group["threshold"], 1)
        s = min(1.0, hits / threshold)
        scores[f"nice_{group_name}"] = round(s, 4)
        nice_scores.append(s)

    scores["overall_nice"] = round(
        sum(nice_scores) / len(nice_scores) if nice_scores else 0.0, 4
    )

    # ── Technical domain patterns ─────────────────────────────────────────────
    for domain, pattern in [
        ("vector_db",     VECTOR_DB_PATTERNS),
        ("retrieval",     RETRIEVAL_PATTERNS),
        ("ranking",       RANKING_PATTERNS),
        ("evaluation",    EVALUATION_PATTERNS),
        ("production_ml", PRODUCTION_ML_PATTERNS),
    ]:
        hits = _count_keyword_hits(
            pattern["keywords"], pattern.get("phrases", []), skills_lower, text_lower
        )
        # Domain patterns: 3 hits = full score (they're diverse signals)
        s = min(1.0, hits / 3.0)
        scores[domain] = round(s, 4)

    # ── Composite ─────────────────────────────────────────────────────────────
    scores["composite_skill_score"] = round(
        0.50 * scores["overall_must_have"]
        + 0.25 * scores["production_ml"]
        + 0.15 * scores["overall_nice"]
        + 0.10 * scores["ranking"],
        4,
    )

    return scores


def detect_disqualifiers(flat: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Check a flat candidate dict against all detectable disqualifier patterns.

    Returns a list of triggered disqualifiers, each as:
    {
        "name"           : str   — disqualifier key
        "description"    : str   — human-readable reason
        "penalty_factor" : float — multiply into final score
        "evidence"       : str   — what triggered it
    }
    If the list is empty, no disqualifiers fired.
    """
    triggered: list[dict[str, Any]] = []
    disqs = DISQUALIFIER_PATTERNS

    # ── 1. Consulting-only career ─────────────────────────────────────────────
    if flat.get("is_consulting_only"):
        triggered.append({
            "name":           "consulting_only_career",
            "description":    "Entire career at IT services / consulting firms.",
            "penalty_factor": disqs["consulting_only_career"]["penalty_factor"],
            "evidence":       f"companies: {flat.get('career_companies', [])}",
        })

    # ── 2. Non-technical primary title ────────────────────────────────────────
    title = (flat.get("most_recent_title") or flat.get("current_title") or "").lower()
    nt_titles = disqs["nontechnical_primary_title"]["nontechnical_titles"]
    if any(nt in title for nt in nt_titles):
        triggered.append({
            "name":           "nontechnical_primary_title",
            "description":    f"Current title '{title}' is not a technical ML/AI role.",
            "penalty_factor": disqs["nontechnical_primary_title"]["penalty_factor"],
            "evidence":       f"current_title='{title}'",
        })

    # ── 3. LangChain-only without Tier 1 skills ───────────────────────────────
    skill_names_lower = {s.lower() for s in (flat.get("skill_names") or [])}
    has_langchain = "langchain" in skill_names_lower
    tier1_count   = flat.get("tier1_skill_count", 0)
    if has_langchain and tier1_count < disqs["langchain_only_no_tier1"]["require_tier1_min"]:
        triggered.append({
            "name":           "langchain_only_no_tier1",
            "description":    "LangChain listed but no Tier-1 retrieval/vector skills.",
            "penalty_factor": disqs["langchain_only_no_tier1"]["penalty_factor"],
            "evidence":       f"has_langchain=True, tier1_skill_count={tier1_count}",
        })

    # ── 4. CV/Speech/Robotics primary without NLP/IR ─────────────────────────
    cv_kws  = set(disqs["cv_speech_robotics_no_nlp"]["cv_speech_keywords"])
    nlp_kws = set(disqs["cv_speech_robotics_no_nlp"]["nlp_ir_keywords"])
    cv_hits  = len(skill_names_lower & cv_kws)
    nlp_hits = len(skill_names_lower & nlp_kws) + (tier1_count > 0)
    if cv_hits >= 2 and nlp_hits == 0:
        triggered.append({
            "name":           "cv_speech_robotics_no_nlp",
            "description":    "Primary expertise in CV/Speech/Robotics with no NLP/IR signals.",
            "penalty_factor": disqs["cv_speech_robotics_no_nlp"]["penalty_factor"],
            "evidence":       f"cv_hits={cv_hits}, nlp_hits={nlp_hits}",
        })

    # ── 5. Title-chaser (high company-switch velocity) ────────────────────────
    n_roles = flat.get("n_career_roles", 0)
    total_months = flat.get("total_career_months", 0)
    min_roles = disqs["title_chaser"]["min_roles_to_check"]
    if n_roles >= min_roles and total_months > 0:
        avg_tenure = total_months / n_roles
        if avg_tenure < disqs["title_chaser"]["avg_tenure_threshold_months"]:
            triggered.append({
                "name":           "title_chaser",
                "description":    f"Average tenure {avg_tenure:.0f} months across {n_roles} roles suggests job-hopping.",
                "penalty_factor": disqs["title_chaser"]["penalty_factor"],
                "evidence":       f"avg_tenure={avg_tenure:.1f}m, n_roles={n_roles}",
            })

    # ── 6. Outside India + not willing to relocate ────────────────────────────
    country = (flat.get("country") or "").lower()
    willing = flat.get("willing_to_relocate", False)
    if "india" not in country and not willing:
        triggered.append({
            "name":           "outside_india_no_relocation",
            "description":    "Candidate outside India and not willing to relocate.",
            "penalty_factor": disqs["outside_india_no_relocation"]["penalty_factor"],
            "evidence":       f"country='{flat.get('country')}', willing_to_relocate={willing}",
        })

    # ── 7. Long notice period ─────────────────────────────────────────────────
    notice = flat.get("notice_period_days", 0)
    if notice > disqs["long_notice_period"]["threshold_days"]:
        triggered.append({
            "name":           "long_notice_period",
            "description":    f"Notice period {notice} days exceeds threshold.",
            "penalty_factor": disqs["long_notice_period"]["penalty_factor"],
            "evidence":       f"notice_period_days={notice}",
        })

    # ── 8. Inactive / not available ───────────────────────────────────────────
    days_inactive   = flat.get("days_since_last_active", 0)
    open_to_work    = flat.get("open_to_work_flag", True)
    response_rate   = flat.get("recruiter_response_rate", 1.0)
    inactive_thresh = disqs["inactive_not_available"]["inactive_days_threshold"]
    rr_thresh       = disqs["inactive_not_available"]["low_response_rate_threshold"]

    if days_inactive > inactive_thresh and not open_to_work and response_rate < rr_thresh:
        triggered.append({
            "name":           "inactive_not_available",
            "description":    "Candidate appears behaviorally unavailable.",
            "penalty_factor": disqs["inactive_not_available"]["penalty_factor"],
            "evidence":       (
                f"days_since_active={days_inactive}, "
                f"open_to_work={open_to_work}, "
                f"response_rate={response_rate:.2f}"
            ),
        })

    return triggered


def compute_disqualifier_penalty(flat: dict[str, Any]) -> float:
    """
    Compound all triggered disqualifier penalty factors multiplicatively.

    Returns a float in (0, 1].  1.0 = no disqualifiers.  0.0 = elimination.
    """
    triggered = detect_disqualifiers(flat)
    penalty = 1.0
    for d in triggered:
        penalty *= d["penalty_factor"]
    return round(penalty, 6)


def score_location_match(
    country: str,
    location: str,
    willing_to_relocate: bool,
) -> float:
    """
    Return a location fit score in [0, 1].

    1.00  — India + preferred city (Noida, Pune)
    0.80  — India + acceptable city (Hyderabad, Mumbai, Delhi NCR, Bangalore)
    0.65  — India + other city
    0.45  — Outside India + willing to relocate
    0.10  — Outside India + not willing to relocate
    """
    loc   = LOCATION_REQUIREMENTS
    scores = loc["scores"]

    country_lower  = country.lower().strip()
    location_lower = location.lower().strip()

    is_india = "india" in country_lower

    preferred_lower  = {c.lower() for c in loc["preferred_cities"]}
    acceptable_lower = {c.lower() for c in loc["acceptable_cities"]}

    in_preferred  = any(c in location_lower for c in preferred_lower)
    in_acceptable = any(c in location_lower for c in acceptable_lower)

    if is_india and in_preferred:
        return scores["preferred_country_preferred_city"]
    if is_india and in_acceptable:
        return scores["preferred_country_acceptable_city"]
    if is_india:
        return scores["preferred_country_other_city"]
    if willing_to_relocate:
        return scores["outside_country_willing_relocate"]
    return scores["outside_country_not_relocate"]


def score_experience_fit(years_of_experience: float) -> float:
    """
    Gaussian fit centred on the JD ideal (7 years, σ=2.5).

    Returns score in (0, 1].  Hard floor: < 2 years → 0.05.
    """
    mu    = EXPERIENCE_REQUIREMENTS["gaussian_mu"]
    sigma = EXPERIENCE_REQUIREMENTS["gaussian_sigma"]
    yoe   = max(0.0, years_of_experience)

    if yoe < 2.0:
        return 0.05

    score = math.exp(-0.5 * ((yoe - mu) / sigma) ** 2)
    return round(score, 4)


def score_notice_period(notice_period_days: int) -> float:
    """
    Score the notice period against JD preference (sub-30 days ideal).

    1.0   — ≤ 30 days  (ideal: can buy out)
    0.80  — 31–60 days
    0.60  — 61–90 days
    0.40  — 91–120 days (bar gets significantly higher)
    0.20  — > 120 days
    """
    if notice_period_days <= 30:
        return 1.00
    if notice_period_days <= 60:
        return 0.80
    if notice_period_days <= 90:
        return 0.60
    if notice_period_days <= 120:
        return 0.40
    return 0.20
