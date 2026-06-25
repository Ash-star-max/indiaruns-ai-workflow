"""
embedder.py — Sentence-transformer embedding pipeline

Generates dense L2-normalised vector representations of candidate profiles
and the JD query using all-MiniLM-L6-v2 (22 MB, CPU inference, 384 dims).

All embeddings are precomputed offline. This module is NOT called during
the ≤5-min online ranking step.

Public API
----------
    load_model(model_name) -> SentenceTransformer
    embed_texts(texts, model, batch_size, show_progress) -> np.ndarray  (N, 384)
    embed_jd(model) -> np.ndarray                                        (384,)
    embed_candidates(flat_rows, model, ...) -> np.ndarray               (N, 384)
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_BATCH, EMBEDDING_MODEL
from src.jd_understanding import get_jd_embedding_text


def load_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Load the sentence-transformer model.
    Downloads once to the local HuggingFace cache; subsequent calls are instant.
    """
    return SentenceTransformer(model_name)


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = EMBEDDING_BATCH,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of strings.

    Returns float32 array of shape (len(texts), model_dim).
    Vectors are L2-normalised so cosine similarity = dot product.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_jd(model: SentenceTransformer) -> np.ndarray:
    """
    Embed the JD text for semantic similarity scoring.

    Returns float32 array of shape (EMBEDDING_DIM,).
    Uses get_jd_embedding_text() as the canonical JD representation.
    """
    jd_text = get_jd_embedding_text()
    emb = model.encode(
        [jd_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb[0].astype(np.float32)


def embed_candidates(
    flat_rows: list[dict],
    model: SentenceTransformer,
    batch_size: int = EMBEDDING_BATCH,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed candidate profiles from a list of flat dicts.

    Uses the pre-built ``candidate_text`` field produced by
    load_data.flatten_candidate(). Falls back to summary + skill_names
    when candidate_text is absent.

    Returns float32 array of shape (N, EMBEDDING_DIM).
    """
    texts: list[str] = []
    for flat in flat_rows:
        text = str(flat.get("candidate_text") or "").strip()
        if not text:
            summary = str(flat.get("summary") or "")
            skills  = " ".join(flat.get("skill_names") or [])
            text    = f"{summary} {skills}".strip()
        texts.append(text)

    return embed_texts(texts, model, batch_size=batch_size, show_progress=show_progress)
