"""
text_features.py — TF-IDF semantic scoring of candidates against the JD.

Design
------
Offline (precompute.py calls fit_transform once on all 100K candidates):
  1. Clean & normalise candidate texts           vectorised pandas str ops
  2. Build expanded JD query                     must-have keywords ×3, domain ×2
  3. Fit TfidfVectorizer on corpus + JD query    ngrams(1,2), 60K features
  4. Compute cosine similarity (sparse matmul)   <5 s for 100K
  5. Boost via TF-IDF column subset              no second vectoriser needed
  6. Calibrate to [0,1] (percentile min-max)     consistent across calls
  7. Save pipeline via joblib                    reusable for new candidates

Online (ranker.py loads pre-saved .npy — no TF-IDF at query time):
  Precomputed jd_semantic_score loaded directly from ARTIFACTS_DIR.

Public API
----------
    clean_text(text)                       -> str
    batch_clean_texts(texts)               -> list[str]
    build_jd_query_text()                  -> str
    TextFeaturePipeline.fit(corpus)        -> self
    TextFeaturePipeline.transform(corpus)  -> np.ndarray   float32 in [0,1]
    TextFeaturePipeline.fit_transform(c)   -> np.ndarray
    TextFeaturePipeline.save(path)
    TextFeaturePipeline.load(path)         classmethod
    compute_jd_semantic_scores(texts, ...) -> tuple[np.ndarray, TextFeaturePipeline]
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import ARTIFACTS_DIR
from src.jd_understanding import (
    EVALUATION_PATTERNS,
    MUST_HAVE_SKILLS,
    NICE_TO_HAVE_SKILLS,
    PRODUCTION_ML_PATTERNS,
    RANKING_PATTERNS,
    RETRIEVAL_PATTERNS,
    VECTOR_DB_PATTERNS,
    get_jd_embedding_text,
)

# ─────────────────────────────────────────────────────────────────────────────
# § 1  Constants
# ─────────────────────────────────────────────────────────────────────────────

TFIDF_PIPELINE_FILE = ARTIFACTS_DIR / "tfidf_pipeline.joblib"
JD_SEMANTIC_SCORES_FILE = ARTIFACTS_DIR / "jd_semantic_scores.npy"

# TF-IDF hyperparameters — calibrated for 100K candidate corpus
_TFIDF_PARAMS: dict = {
    "sublinear_tf":   True,         # log(tf+1) — better for long documents
    "min_df":         2,            # ignore terms appearing in only 1 doc
    "max_df":         0.95,         # ignore near-universal terms
    "ngram_range":    (1, 2),       # unigrams + bigrams for compound skills
    "max_features":   60_000,       # vocab cap; covers full skills vocabulary
    "dtype":          np.float32,
    "strip_accents":  "ascii",
    "analyzer":       "word",
    # Require ≥2 chars, allow hyphens inside tokens for "sentence-transformers"
    "token_pattern":  r"(?u)\b[a-z][a-z0-9\-]+\b",
}

# Calibration percentile window — maps p05→0, p95→1
_CALIB_LOW_PCT  = 5.0
_CALIB_HIGH_PCT = 95.0

# Keyword boost budget — candidate hitting 30 % of boost keywords gets full boost
_BOOST_HIT_FRACTION = 0.30
# Default max additive boost above the base cosine similarity score
DEFAULT_KEYWORD_BOOST_MAX = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# § 2  Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

# Compiled patterns — created once at import time
_RE_URL    = re.compile(r"https?://\S+|www\.\S+")
_RE_HTML   = re.compile(r"<[^>]+>")
_RE_KEEP   = re.compile(r"[^a-zA-Z0-9\s\-]")   # keep letters, digits, spaces, hyphens
_RE_SPACE  = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """
    Clean a single candidate text string.

    Steps: None-guard → unicode normalise → strip URLs → strip HTML →
           remove non-alphanumeric (keep hyphens) → collapse whitespace → lowercase.
    Hyphens are preserved so 'sentence-transformers' stays as one token.
    """
    if not text:
        return ""
    # Unicode normalise (NFKD) then drop non-ASCII — handles accented chars
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = _RE_URL.sub(" ", text)
    text = _RE_HTML.sub(" ", text)
    text = _RE_KEEP.sub(" ", text)
    text = _RE_SPACE.sub(" ", text)
    return text.strip().lower()


def batch_clean_texts(texts: Iterable[str | None]) -> list[str]:
    """
    Vectorised cleaning for large corpora using pandas string ops (C-speed).

    Significantly faster than a Python loop for N > 1000.
    Returns a list of cleaned strings in the same order.
    """
    s = pd.Series(list(texts), dtype="object").fillna("").astype(str)
    # Remove URLs
    s = s.str.replace(_RE_URL.pattern,  " ", regex=True)
    # Remove HTML tags
    s = s.str.replace(_RE_HTML.pattern, " ", regex=True)
    # Unicode normalisation: strip accented chars via encode/decode
    s = s.apply(
        lambda t: unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    )
    # Non-alphanumeric (keep hyphens)
    s = s.str.replace(_RE_KEEP.pattern, " ", regex=True)
    # Collapse whitespace + lowercase
    s = s.str.replace(_RE_SPACE.pattern, " ", regex=True).str.strip().str.lower()
    return s.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# § 3  JD query builder
# ─────────────────────────────────────────────────────────────────────────────

def build_jd_query_text() -> str:
    """
    Assemble an expanded JD text for use as the TF-IDF query vector.

    Must-have keywords are repeated 3× (inflates their IDF-weighted TF in
    the query, increasing their contribution to cosine similarity).
    Domain pattern keywords are repeated 2×.
    Nice-to-have keywords appear once.
    The base JD semantic description anchors the overall intent.
    """
    parts: list[str] = [get_jd_embedding_text()]

    # Must-have: 3× repetition for implicit query-side weighting
    for group in MUST_HAVE_SKILLS.values():
        parts.extend(group["keywords"] * 3)
        parts.extend(group.get("phrases", []) * 2)

    # Domain patterns: 2×
    for pattern in [VECTOR_DB_PATTERNS, RETRIEVAL_PATTERNS,
                    RANKING_PATTERNS, EVALUATION_PATTERNS,
                    PRODUCTION_ML_PATTERNS]:
        parts.extend(pattern["keywords"] * 2)
        parts.extend(pattern.get("phrases", []) * 2)

    # Nice-to-have: 1×
    for group in NICE_TO_HAVE_SKILLS.values():
        parts.extend(group["keywords"])
        parts.extend(group.get("phrases", []))

    raw = " ".join(str(p) for p in parts if p)
    return clean_text(raw)


# ─────────────────────────────────────────────────────────────────────────────
# § 4  Boost keyword list (derived from JD patterns, cleaned once)
# ─────────────────────────────────────────────────────────────────────────────

def _build_boost_keywords() -> list[str]:
    """Return unique, cleaned tier-1 boost keywords derived from must-have groups."""
    kws: list[str] = []
    for group in MUST_HAVE_SKILLS.values():
        kws.extend(group["keywords"])
    for pattern in [VECTOR_DB_PATTERNS, RETRIEVAL_PATTERNS, EVALUATION_PATTERNS]:
        kws.extend(pattern["keywords"])
    cleaned = [clean_text(k) for k in kws]
    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for k in cleaned:
        if k and k not in seen:
            seen.add(k)
            result.append(k)
    return result


_BOOST_KEYWORDS: list[str] = _build_boost_keywords()


# ─────────────────────────────────────────────────────────────────────────────
# § 5  TextFeaturePipeline
# ─────────────────────────────────────────────────────────────────────────────

class TextFeaturePipeline:
    """
    Fits a TF-IDF vectoriser on a candidate corpus, then scores all candidates
    against an expanded JD query using cosine similarity + keyword boosting.

    Serialisable via TextFeaturePipeline.save() / TextFeaturePipeline.load().

    Attributes (post-fit)
    ---------------------
    vectorizer         : fitted TfidfVectorizer
    keyword_boost_max  : max additive boost from keyword hits
    _jd_vec_norm       : L2-normalised JD sparse row vector (1, vocab)
    _boost_indices     : int32 indices into vocabulary for boost keywords
    _cal_p05           : calibration lower bound (computed during fit_transform)
    _cal_p95           : calibration upper bound
    """

    def __init__(
        self,
        *,
        keyword_boost_max: float = DEFAULT_KEYWORD_BOOST_MAX,
        tfidf_params: dict | None = None,
    ) -> None:
        self.keyword_boost_max = keyword_boost_max
        _params = dict(_TFIDF_PARAMS)
        if tfidf_params:
            _params.update(tfidf_params)
        self.vectorizer = TfidfVectorizer(**_params)

        # Set after fit()
        self._jd_vec_norm:    object | None = None
        self._boost_indices:  np.ndarray | None = None
        self._cal_p05:        float = 0.0
        self._cal_p95:        float = 1.0
        self._fitted:         bool = False

    # ── Fit ─────────────────────────────────────────────────────────────────

    def fit(self, corpus: list[str]) -> "TextFeaturePipeline":
        """
        Fit the TF-IDF vectoriser on corpus + JD query.

        The JD query is appended to the corpus before fitting so its terms are
        included in the vocabulary and IDF is computed correctly.
        """
        jd_clean = build_jd_query_text()
        clean_corpus = batch_clean_texts(corpus)
        combined = clean_corpus + [jd_clean]

        self.vectorizer.fit(combined)

        # Store L2-normalised JD vector for cosine similarity
        jd_tfidf = self.vectorizer.transform([jd_clean])
        self._jd_vec_norm = normalize(jd_tfidf, norm="l2", copy=True)

        # Precompute boost keyword indices in vocabulary
        feature_names = self.vectorizer.get_feature_names_out()
        vocab_map = {name: idx for idx, name in enumerate(feature_names)}
        self._boost_indices = np.array(
            [vocab_map[kw] for kw in _BOOST_KEYWORDS if kw in vocab_map],
            dtype=np.int32,
        )

        self._fitted = True
        return self

    # ── Transform ────────────────────────────────────────────────────────────

    def transform(self, corpus: list[str]) -> np.ndarray:
        """
        Score each candidate in corpus against the JD.

        Returns float32 array of shape (n,) in [0, 1].
        Uses saved calibration parameters if available (set during fit_transform).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() or fit_transform() before transform().")

        clean_corpus = batch_clean_texts(corpus)

        # TF-IDF matrix: sparse (n, vocab), dtype=float32
        tfidf_mat = self.vectorizer.transform(clean_corpus)

        # Cosine similarity = dot product of L2-normalised vectors
        tfidf_norm = normalize(tfidf_mat, norm="l2", copy=True)
        cos_sim = np.asarray(
            (tfidf_norm @ self._jd_vec_norm.T).todense()
        ).flatten().astype(np.float32)

        # Keyword boost using pre-identified vocabulary column indices
        boost = self._compute_keyword_boost(tfidf_mat)
        raw = cos_sim * boost

        # Calibrate to [0, 1]
        span = max(self._cal_p95 - self._cal_p05, 1e-6)
        scores = (raw - self._cal_p05) / span
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    # ── Fit + transform (with calibration fitting) ────────────────────────────

    def fit_transform(self, corpus: list[str]) -> np.ndarray:
        """
        Fit on corpus and return calibrated scores in one pass.
        Calibration parameters (p05, p95) are saved for subsequent transform() calls.
        """
        jd_clean = build_jd_query_text()
        clean_corpus = batch_clean_texts(corpus)
        combined = clean_corpus + [jd_clean]

        # Fit on combined corpus
        self.vectorizer.fit(combined)

        # JD vector
        jd_tfidf = self.vectorizer.transform([jd_clean])
        self._jd_vec_norm = normalize(jd_tfidf, norm="l2", copy=True)

        # Boost indices
        feature_names = self.vectorizer.get_feature_names_out()
        vocab_map = {name: idx for idx, name in enumerate(feature_names)}
        self._boost_indices = np.array(
            [vocab_map[kw] for kw in _BOOST_KEYWORDS if kw in vocab_map],
            dtype=np.int32,
        )
        self._fitted = True

        # Transform corpus (not combined — don't include JD query in scores)
        tfidf_mat = self.vectorizer.transform(clean_corpus)
        tfidf_norm = normalize(tfidf_mat, norm="l2", copy=True)
        cos_sim = np.asarray(
            (tfidf_norm @ self._jd_vec_norm.T).todense()
        ).flatten().astype(np.float32)

        boost = self._compute_keyword_boost(tfidf_mat)
        raw = cos_sim * boost

        # Fit calibration: map [p05, p95] → [0, 1]
        self._cal_p05 = float(np.percentile(raw, _CALIB_LOW_PCT))
        self._cal_p95 = float(np.percentile(raw, _CALIB_HIGH_PCT))
        span = max(self._cal_p95 - self._cal_p05, 1e-6)
        scores = (raw - self._cal_p05) / span
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    # ── Keyword boost (internal) ──────────────────────────────────────────────

    def _compute_keyword_boost(self, tfidf_mat) -> np.ndarray:
        """
        Multiplicative boost based on unique tier-1 keyword hits.

        Reuses already-computed TF-IDF matrix: select boost-keyword columns,
        count rows with any non-zero entry, normalise, scale to max_boost.
        No second vectoriser needed.

        Returns float32 array of shape (n,) with values in [1.0, 1+max_boost].
        """
        if self._boost_indices is None or len(self._boost_indices) == 0:
            return np.ones(tfidf_mat.shape[0], dtype=np.float32)

        # Select boost-keyword columns from TF-IDF matrix (sparse submatrix)
        boost_cols = tfidf_mat[:, self._boost_indices]

        # Count unique boost keywords present per candidate
        unique_hits = np.asarray((boost_cols > 0).sum(axis=1)).flatten().astype(np.float32)

        # Normalise: hitting _BOOST_HIT_FRACTION of boost keywords = full boost
        threshold = max(len(self._boost_indices) * _BOOST_HIT_FRACTION, 1.0)
        normalised = np.minimum(unique_hits / threshold, 1.0)
        return (1.0 + normalised * self.keyword_boost_max).astype(np.float32)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path | str | None = None) -> Path:
        """Serialise pipeline via joblib. Defaults to TFIDF_PIPELINE_FILE."""
        dest = Path(path) if path else TFIDF_PIPELINE_FILE
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        return dest

    @classmethod
    def load(cls, path: Path | str | None = None) -> "TextFeaturePipeline":
        """Deserialise a previously saved pipeline."""
        src = Path(path) if path else TFIDF_PIPELINE_FILE
        return joblib.load(src)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.vectorizer.vocabulary_) if self._fitted else 0

    @property
    def n_boost_keywords_in_vocab(self) -> int:
        return len(self._boost_indices) if self._boost_indices is not None else 0

    def __repr__(self) -> str:
        status = f"fitted, vocab={self.vocab_size}" if self._fitted else "unfitted"
        return f"TextFeaturePipeline({status}, boost_max={self.keyword_boost_max})"


# ─────────────────────────────────────────────────────────────────────────────
# § 6  Public convenience function
# ─────────────────────────────────────────────────────────────────────────────

def compute_jd_semantic_scores(
    candidate_texts: list[str] | pd.Series,
    *,
    pipeline: TextFeaturePipeline | None = None,
    keyword_boost: bool = True,
    keyword_boost_max: float = DEFAULT_KEYWORD_BOOST_MAX,
    tfidf_params: dict | None = None,
) -> tuple[np.ndarray, TextFeaturePipeline]:
    """
    Compute jd_semantic_score for a corpus of candidate texts.

    Parameters
    ----------
    candidate_texts  : list or pd.Series of raw candidate text strings.
    pipeline         : Optional pre-fitted TextFeaturePipeline.
                       If provided, calls transform() instead of fit_transform().
    keyword_boost    : Apply tier-1 keyword multiplicative boost (default True).
    keyword_boost_max: Max additive boost factor (default 0.25).
    tfidf_params     : Override TF-IDF hyperparameters for experimentation.

    Returns
    -------
    scores    : np.ndarray float32, shape (n,), values in [0, 1].
                Higher = more semantically relevant to the JD.
    pipeline  : The fitted TextFeaturePipeline (pass back in for incremental use).

    Notes
    -----
    - Calling with pipeline=None fits a new pipeline (use for the full 100K corpus).
    - Calling with a fitted pipeline calls transform() only (for new candidates).
    - For offline precompute, the returned pipeline should be saved via pipeline.save().
    """
    if isinstance(candidate_texts, pd.Series):
        corpus = candidate_texts.tolist()
    else:
        corpus = list(candidate_texts)

    if pipeline is not None:
        # Re-use existing fitted pipeline
        scores = pipeline.transform(corpus)
        return scores, pipeline

    # Fit fresh pipeline
    p = TextFeaturePipeline(
        keyword_boost_max=keyword_boost_max if keyword_boost else 0.0,
        tfidf_params=tfidf_params,
    )
    scores = p.fit_transform(corpus)
    return scores, p
