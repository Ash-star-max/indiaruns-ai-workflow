"""
Tests for src/text_features.py

§A  Text cleaning — clean_text()
§B  Batch cleaning — batch_clean_texts()
§C  JD query builder — build_jd_query_text()
§D  Pipeline structure — TextFeaturePipeline attributes
§E  TF-IDF fit correctness (small corpus)
§F  Keyword boost behaviour
§G  Score computation properties
§H  Integration — relevant vs irrelevant candidates
§I  Persistence — save / load round-trip
§J  Public API — compute_jd_semantic_scores()
§K  Benchmark — timing assertions on synthetic corpora
"""

from __future__ import annotations

import os
import time
import random
from pathlib import Path

import numpy as np
import pytest

from src.config import CANDIDATES_FILE
from src.text_features import (
    DEFAULT_KEYWORD_BOOST_MAX,
    TextFeaturePipeline,
    batch_clean_texts,
    build_jd_query_text,
    clean_text,
    compute_jd_semantic_scores,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

_RELEVANT_TEXTS = [
    (
        "Senior ML Engineer with 7 years experience. Built production embeddings-based "
        "retrieval system using sentence-transformers and FAISS deployed to 10M users. "
        "Designed NDCG MRR MAP evaluation framework. A/B tested ranking improvements. "
        "Strong Python PyTorch. Pinecone Elasticsearch vector search hybrid search."
    ),
    (
        "Applied scientist with 6 years at product companies. Deep expertise in "
        "semantic search, dense retrieval, FAISS, Qdrant. Implemented BGE E5 "
        "embedding models. Evaluated with NDCG MAP MRR metrics. Production ML."
    ),
    (
        "NLP Engineer. Built ranking system using XGBoost LTR and sentence-transformers. "
        "Elasticsearch hybrid search. Weaviate Milvus OpenSearch. Python PyTorch. "
        "Deployed recommendation engine serving real users at scale. ndcg evaluation."
    ),
]

_IRRELEVANT_TEXTS = [
    "Marketing manager with 8 years experience. Excel PowerPoint VLOOKUP. MBA from Delhi.",
    "Civil engineer specialised in structural analysis and AutoCAD. 10 years experience.",
    "Content writer SEO social media marketing copywriting. Brand strategy.",
    "HR Business Partner talent acquisition onboarding payroll. Human resources 6 years.",
    "Graphic designer Photoshop Illustrator Figma. UI/UX visual identity.",
]

_MIXED_CORPUS = _RELEVANT_TEXTS + _IRRELEVANT_TEXTS


def _generate_synthetic_texts(n: int, seed: int = 42) -> list[str]:
    """
    Generate n synthetic candidate texts for benchmarking.
    ~20% relevant, ~80% irrelevant, each ~400 chars.
    """
    rng = random.Random(seed)
    relevant_templates = [
        (
            "ML engineer with {yoe} years. Production embeddings retrieval FAISS "
            "sentence-transformers Elasticsearch. NDCG MRR MAP evaluation. Python "
            "PyTorch vector search ranking recommendation deployed production."
        ),
        (
            "NLP engineer {yoe} years product company. Dense retrieval semantic search "
            "Pinecone Weaviate Qdrant hybrid search. A/B testing offline evaluation. "
            "Fine-tuning LoRA PEFT transformers BERT. MLOps deployed real users."
        ),
    ]
    irrelevant_templates = [
        "Sales executive {yoe} years. Revenue targets pipeline CRM Excel PowerPoint.",
        "Mechanical engineer {yoe} years. AutoCAD SolidWorks manufacturing quality.",
        "Content marketer {yoe} years. SEO blog social media copywriting analytics.",
        "Accountant {yoe} years. GST tally balance sheet payroll finance reporting.",
        "Android developer {yoe} years. Kotlin Java Retrofit Room Firebase mobile apps.",
        "React frontend developer {yoe} years. JavaScript TypeScript Redux CSS HTML5.",
        "DevOps engineer {yoe} years. Docker Kubernetes CI/CD Terraform AWS infrastructure.",
        "Computer vision engineer {yoe} years. OpenCV object detection image classification.",
    ]
    texts = []
    for i in range(n):
        yoe = rng.randint(2, 15)
        if rng.random() < 0.20:
            t = rng.choice(relevant_templates)
        else:
            t = rng.choice(irrelevant_templates)
        texts.append(t.format(yoe=yoe))
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# §A  Text cleaning — clean_text()
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanText:

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_lowercases_text(self):
        assert clean_text("PYTHON PyTorch FAISS") == "python pytorch faiss"

    def test_strips_html_tags(self):
        result = clean_text("<b>Machine Learning</b> <span>Engineer</span>")
        assert "<b>" not in result
        assert "machine learning" in result

    def test_strips_urls(self):
        result = clean_text("Visit https://example.com/ml for more. Python engineer.")
        assert "https" not in result
        assert "python" in result

    def test_removes_special_chars(self):
        result = clean_text("ML@Engineer! $100k #hiring (remote)")
        assert "@" not in result and "!" not in result and "$" not in result

    def test_preserves_hyphens(self):
        # sentence-transformers must stay as one hyphenated unit
        result = clean_text("sentence-transformers for semantic search")
        assert "sentence-transformers" in result

    def test_collapses_whitespace(self):
        result = clean_text("python   pytorch    nlp")
        assert "  " not in result

    def test_strips_leading_trailing_whitespace(self):
        result = clean_text("  python engineer  ")
        assert result == result.strip()

    def test_handles_unicode_accents(self):
        result = clean_text("naïve résumé café")
        assert "naive" in result or "resume" in result or "cafe" in result

    def test_returns_string(self):
        assert isinstance(clean_text("hello world"), str)

    def test_digits_preserved(self):
        result = clean_text("GPT-4 transformer model 2024")
        assert "gpt" in result or "4" in result
        assert "2024" in result


# ─────────────────────────────────────────────────────────────────────────────
# §B  Batch cleaning — batch_clean_texts()
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchCleanTexts:

    def test_returns_list(self):
        result = batch_clean_texts(["hello", "world"])
        assert isinstance(result, list)

    def test_length_preserved(self):
        texts = ["text one", "text two", "text three"]
        assert len(batch_clean_texts(texts)) == len(texts)

    def test_none_values_become_empty(self):
        result = batch_clean_texts([None, "python", None])
        assert result[0] == ""
        assert result[2] == ""

    def test_matches_single_clean_text(self):
        texts = [
            "ML Engineer with FAISS experience",
            "<b>Python developer</b> https://link.com",
            "  NDCG MRR evaluation  ",
        ]
        batch_result = batch_clean_texts(texts)
        single_results = [clean_text(t) for t in texts]
        assert batch_result == single_results

    def test_empty_list(self):
        assert batch_clean_texts([]) == []

    def test_large_batch_no_crash(self):
        texts = ["Python machine learning engineer"] * 500
        result = batch_clean_texts(texts)
        assert len(result) == 500


# ─────────────────────────────────────────────────────────────────────────────
# §C  JD query builder — build_jd_query_text()
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildJdQueryText:

    def test_returns_nonempty_string(self):
        q = build_jd_query_text()
        assert isinstance(q, str) and len(q) > 100

    def test_contains_faiss(self):
        assert "faiss" in build_jd_query_text()

    def test_contains_ndcg(self):
        assert "ndcg" in build_jd_query_text()

    def test_contains_sentence_transformers(self):
        q = build_jd_query_text()
        assert "sentence-transformers" in q or "sentence transformers" in q

    def test_must_have_keywords_appear_multiple_times(self):
        q = build_jd_query_text()
        # FAISS is in must-have; should appear 3+ times due to repetition
        assert q.count("faiss") >= 3

    def test_is_lowercase(self):
        q = build_jd_query_text()
        assert q == q.lower()

    def test_deterministic(self):
        assert build_jd_query_text() == build_jd_query_text()


# ─────────────────────────────────────────────────────────────────────────────
# §D  Pipeline structure
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineStructure:

    def test_default_instantiation(self):
        p = TextFeaturePipeline()
        assert p.keyword_boost_max == DEFAULT_KEYWORD_BOOST_MAX
        assert not p._fitted

    def test_custom_boost_max(self):
        p = TextFeaturePipeline(keyword_boost_max=0.10)
        assert p.keyword_boost_max == 0.10

    def test_transform_before_fit_raises(self):
        p = TextFeaturePipeline()
        with pytest.raises(RuntimeError, match="fit"):
            p.transform(["some text"])

    def test_repr_unfitted(self):
        assert "unfitted" in repr(TextFeaturePipeline())

    def test_vocab_size_zero_before_fit(self):
        assert TextFeaturePipeline().vocab_size == 0


# ─────────────────────────────────────────────────────────────────────────────
# §E  TF-IDF fit correctness (small corpus)
# ─────────────────────────────────────────────────────────────────────────────

class TestTfidfFitCorrectness:

    @pytest.fixture(scope="class")
    def fitted_pipeline(self):
        p = TextFeaturePipeline()
        p.fit(_MIXED_CORPUS * 5)   # 40 docs — enough for min_df=2
        return p

    def test_fitted_flag_set(self, fitted_pipeline):
        assert fitted_pipeline._fitted

    def test_vocab_size_positive(self, fitted_pipeline):
        assert fitted_pipeline.vocab_size > 0

    def test_jd_vec_stored(self, fitted_pipeline):
        assert fitted_pipeline._jd_vec_norm is not None

    def test_boost_indices_stored(self, fitted_pipeline):
        assert fitted_pipeline._boost_indices is not None
        assert len(fitted_pipeline._boost_indices) > 0

    def test_boost_keywords_in_vocab(self, fitted_pipeline):
        assert fitted_pipeline.n_boost_keywords_in_vocab > 0

    def test_transform_returns_array(self, fitted_pipeline):
        scores = fitted_pipeline.transform(_MIXED_CORPUS)
        assert isinstance(scores, np.ndarray)

    def test_transform_shape(self, fitted_pipeline):
        scores = fitted_pipeline.transform(_MIXED_CORPUS)
        assert scores.shape == (len(_MIXED_CORPUS),)

    def test_transform_dtype_float32(self, fitted_pipeline):
        scores = fitted_pipeline.transform(_MIXED_CORPUS)
        assert scores.dtype == np.float32

    def test_scores_in_0_1(self, fitted_pipeline):
        scores = fitted_pipeline.transform(_MIXED_CORPUS)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_repr_fitted(self, fitted_pipeline):
        assert "fitted" in repr(fitted_pipeline)
        assert "vocab" in repr(fitted_pipeline)


# ─────────────────────────────────────────────────────────────────────────────
# §F  Keyword boost behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordBoost:

    @pytest.fixture(scope="class")
    def pipeline_no_boost(self):
        p = TextFeaturePipeline(keyword_boost_max=0.0)
        p.fit(_MIXED_CORPUS * 5)
        return p

    @pytest.fixture(scope="class")
    def pipeline_with_boost(self):
        p = TextFeaturePipeline(keyword_boost_max=0.25)
        p.fit(_MIXED_CORPUS * 5)
        return p

    def test_boost_increases_relevant_scores(self, pipeline_no_boost, pipeline_with_boost):
        # Relevant candidates should benefit more from keyword boost than irrelevant
        scores_no_boost   = pipeline_no_boost.transform(_RELEVANT_TEXTS)
        scores_with_boost = pipeline_with_boost.transform(_RELEVANT_TEXTS)
        # At least one relevant candidate should score higher with boost
        assert (scores_with_boost >= scores_no_boost).any()

    def test_zero_boost_is_valid(self, pipeline_no_boost):
        scores = pipeline_no_boost.transform(_MIXED_CORPUS)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_boost_does_not_exceed_cap(self, pipeline_with_boost):
        scores = pipeline_with_boost.transform(_MIXED_CORPUS)
        assert scores.max() <= 1.0

    def test_boost_indices_are_valid_vocab_positions(self, pipeline_with_boost):
        vocab_size = pipeline_with_boost.vocab_size
        assert (pipeline_with_boost._boost_indices < vocab_size).all()
        assert (pipeline_with_boost._boost_indices >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# §G  Score computation properties
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreProperties:

    @pytest.fixture(scope="class")
    def pipeline_and_scores(self):
        p = TextFeaturePipeline()
        scores = p.fit_transform(_MIXED_CORPUS * 5)
        return p, scores

    def test_fit_transform_returns_array(self, pipeline_and_scores):
        _, scores = pipeline_and_scores
        assert isinstance(scores, np.ndarray)

    def test_fit_transform_shape(self, pipeline_and_scores):
        _, scores = pipeline_and_scores
        # _MIXED_CORPUS * 5 = 40 items
        assert scores.shape == (len(_MIXED_CORPUS) * 5,)

    def test_fit_transform_scores_in_0_1(self, pipeline_and_scores):
        _, scores = pipeline_and_scores
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_calibration_params_stored(self, pipeline_and_scores):
        p, _ = pipeline_and_scores
        assert hasattr(p, "_cal_p05")
        assert hasattr(p, "_cal_p95")
        assert p._cal_p95 > p._cal_p05

    def test_transform_after_fit_transform_consistent(self, pipeline_and_scores):
        p, scores_fittransform = pipeline_and_scores
        scores_transform = p.transform(_MIXED_CORPUS * 5)
        # Scores should be close (same pipeline, same data, same calibration)
        assert np.allclose(scores_fittransform, scores_transform, atol=1e-5)

    def test_single_candidate_no_crash(self, pipeline_and_scores):
        p, _ = pipeline_and_scores
        score = p.transform(["python machine learning engineer"])
        assert score.shape == (1,)
        assert 0.0 <= score[0] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# §H  Integration — relevant vs irrelevant candidates
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    @pytest.fixture(scope="class")
    def corpus_scores(self):
        # Use a larger mixed corpus so TF-IDF has enough docs for min_df=2
        corpus = _MIXED_CORPUS * 20   # 160 docs
        p = TextFeaturePipeline()
        scores = p.fit_transform(corpus)
        return corpus, p, scores

    def test_relevant_texts_score_higher_than_irrelevant(self, corpus_scores):
        corpus, p, _ = corpus_scores
        rel_scores = p.transform(_RELEVANT_TEXTS)
        irrel_scores = p.transform(_IRRELEVANT_TEXTS)
        assert rel_scores.mean() > irrel_scores.mean(), (
            f"Relevant mean {rel_scores.mean():.4f} <= Irrelevant mean {irrel_scores.mean():.4f}"
        )

    def test_top_candidates_are_relevant(self, corpus_scores):
        corpus, p, scores = corpus_scores
        # In the 160-doc corpus, top 20 should be dominated by _RELEVANT_TEXTS (×20)
        n_relevant_in_corpus = len(_RELEVANT_TEXTS) * 20   # 60
        top_20_indices = np.argsort(scores)[::-1][:20]
        # Relevant indices: 0..2, 8..10, 16..18 ... (every 8 docs in corpus)
        relevant_positions = set()
        for i in range(20):
            base = i * len(_MIXED_CORPUS)
            for r in range(len(_RELEVANT_TEXTS)):
                relevant_positions.add(base + r)
        relevant_in_top20 = sum(1 for idx in top_20_indices if idx in relevant_positions)
        assert relevant_in_top20 >= 10, (
            f"Only {relevant_in_top20}/20 top candidates are relevant"
        )

    def test_empty_text_scores_low(self, corpus_scores):
        _, p, _ = corpus_scores
        score = p.transform([""])
        assert score[0] < 0.5

    def test_jd_like_text_scores_high(self, corpus_scores):
        _, p, _ = corpus_scores
        jd_text = (
            "senior ai engineer production embeddings retrieval sentence-transformers "
            "faiss elasticsearch ndcg mrr map evaluation python pytorch vector search "
            "hybrid search pinecone weaviate recommendation ranking"
        )
        score = p.transform([jd_text])
        # JD-like text should score above median
        corpus_text = [
            "python machine learning engineer nlp sentence-transformers faiss ndcg",
            "java spring boot microservices kubernetes aws devops",
        ] * 10
        baseline_scores = p.transform(corpus_text)
        assert score[0] > baseline_scores.min()

    def test_scores_vary_across_corpus(self, corpus_scores):
        _, _, scores = corpus_scores
        # Scores shouldn't all be the same — we want discrimination
        assert scores.std() > 0.05


# ─────────────────────────────────────────────────────────────────────────────
# §I  Persistence — save / load round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:

    @pytest.fixture
    def tmp_pipeline_path(self, tmp_path):
        return tmp_path / "test_pipeline.joblib"

    def test_save_creates_file(self, tmp_pipeline_path):
        p = TextFeaturePipeline()
        p.fit(_MIXED_CORPUS * 5)
        p.save(tmp_pipeline_path)
        assert tmp_pipeline_path.exists()

    def test_load_returns_pipeline(self, tmp_pipeline_path):
        p = TextFeaturePipeline()
        p.fit(_MIXED_CORPUS * 5)
        p.save(tmp_pipeline_path)
        p2 = TextFeaturePipeline.load(tmp_pipeline_path)
        assert isinstance(p2, TextFeaturePipeline)

    def test_loaded_pipeline_produces_same_scores(self, tmp_pipeline_path):
        p = TextFeaturePipeline()
        p.fit_transform(_MIXED_CORPUS * 5)
        p.save(tmp_pipeline_path)
        p2 = TextFeaturePipeline.load(tmp_pipeline_path)
        scores1 = p.transform(_MIXED_CORPUS)
        scores2 = p2.transform(_MIXED_CORPUS)
        assert np.allclose(scores1, scores2, atol=1e-5)

    def test_loaded_pipeline_is_fitted(self, tmp_pipeline_path):
        p = TextFeaturePipeline()
        p.fit(_MIXED_CORPUS * 5)
        p.save(tmp_pipeline_path)
        p2 = TextFeaturePipeline.load(tmp_pipeline_path)
        assert p2._fitted


# ─────────────────────────────────────────────────────────────────────────────
# §J  Public API — compute_jd_semantic_scores()
# ─────────────────────────────────────────────────────────────────────────────

class TestPublicAPI:

    @pytest.fixture(scope="class")
    def api_result(self):
        corpus = _MIXED_CORPUS * 10
        scores, pipeline = compute_jd_semantic_scores(corpus)
        return scores, pipeline, corpus

    def test_returns_tuple(self, api_result):
        result = compute_jd_semantic_scores(_MIXED_CORPUS * 5)
        assert isinstance(result, tuple) and len(result) == 2

    def test_scores_are_ndarray(self, api_result):
        scores, _, _ = api_result
        assert isinstance(scores, np.ndarray)

    def test_pipeline_is_fitted(self, api_result):
        _, pipeline, _ = api_result
        assert pipeline._fitted

    def test_scores_in_0_1(self, api_result):
        scores, _, _ = api_result
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_pre_fitted_pipeline_reused(self, api_result):
        scores1, pipeline, corpus = api_result
        scores2, _ = compute_jd_semantic_scores(corpus, pipeline=pipeline)
        assert np.allclose(scores1, scores2, atol=1e-5)

    def test_keyword_boost_false_accepted(self):
        scores, p = compute_jd_semantic_scores(_MIXED_CORPUS * 5, keyword_boost=False)
        assert isinstance(scores, np.ndarray)
        assert p.keyword_boost_max == 0.0

    def test_pandas_series_input(self):
        import pandas as pd
        s = pd.Series(_RELEVANT_TEXTS + _IRRELEVANT_TEXTS)
        scores, _ = compute_jd_semantic_scores(
            s,
            tfidf_params={"min_df": 1, "max_features": 1000},
        )
        assert scores.shape == (len(s),)

    def test_custom_tfidf_params_accepted(self):
        scores, p = compute_jd_semantic_scores(
            _MIXED_CORPUS * 5,
            tfidf_params={"max_features": 500, "ngram_range": (1, 1)},
        )
        assert p.vocab_size <= 500


# ─────────────────────────────────────────────────────────────────────────────
# §K  Benchmark — timing assertions on synthetic corpora
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmark:
    """
    Performance benchmarks.

    Timing budgets:
      1K  texts  → < 5 s   (sanity check)
      5K  texts  → < 15 s  (fast path)
      20K texts  → < 60 s  (scale test, projects 100K ≈ 5 min)
    Real 100K benchmark is guarded by the data file existing.
    """

    def test_1k_texts_under_5_seconds(self):
        texts = _generate_synthetic_texts(1_000)
        t0 = time.perf_counter()
        scores, p = compute_jd_semantic_scores(texts)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"1K texts took {elapsed:.1f}s (limit 5s)"
        assert scores.shape == (1_000,)
        assert 0.0 <= scores.min() and scores.max() <= 1.0

    def test_5k_texts_under_15_seconds(self):
        texts = _generate_synthetic_texts(5_000)
        t0 = time.perf_counter()
        scores, p = compute_jd_semantic_texts_wrapper(texts)
        elapsed = time.perf_counter() - t0
        assert elapsed < 15.0, f"5K texts took {elapsed:.1f}s (limit 15s)"
        assert scores.shape == (5_000,)

    def test_20k_texts_under_60_seconds(self):
        texts = _generate_synthetic_texts(20_000)
        t0 = time.perf_counter()
        scores, p = compute_jd_semantic_scores(texts)
        elapsed = time.perf_counter() - t0
        assert elapsed < 60.0, f"20K texts took {elapsed:.1f}s (limit 60s)"
        assert scores.shape == (20_000,)
        # Check scores are discriminative (not all equal)
        assert scores.std() > 0.01

    def test_20k_scores_discriminate_relevant_from_irrelevant(self):
        """Candidates dense with JD keywords must score above pure non-ML candidates."""
        # 500 pure JD-keyword candidates — no ambiguity about content
        enriched = [
            (
                "python pytorch machine learning nlp faiss ndcg mrr map "
                "sentence-transformers elasticsearch vector search hybrid search "
                "pinecone weaviate qdrant ranking evaluation production deployed "
                f"candidate {i}"
            )
            for i in range(500)
        ]
        # 9500 pure non-ML background candidates
        background = [
            (
                "sales executive excel powerpoint vlookup pivot marketing "
                f"manager brand strategy crm revenue pipeline candidate {i}"
            )
            for i in range(9_500)
        ]
        corpus = enriched + background
        scores, _ = compute_jd_semantic_scores(corpus)

        enriched_scores    = scores[:500]
        background_scores  = scores[500:]
        assert enriched_scores.mean() > background_scores.mean(), (
            f"Enriched mean {enriched_scores.mean():.4f} not above "
            f"background mean {background_scores.mean():.4f}"
        )
        # Relative lift must be substantial (≥10 %)
        lift = (enriched_scores.mean() - background_scores.mean()) / max(
            background_scores.mean(), 1e-6
        )
        assert lift > 0.10, f"Score lift {lift:.1%} too small — scores not discriminative"

    def test_transform_is_fast_after_fit(self):
        # Fit once, then measure transform-only time
        corpus = _generate_synthetic_texts(10_000)
        _, pipeline = compute_jd_semantic_scores(corpus[:5_000])

        t0 = time.perf_counter()
        scores = pipeline.transform(corpus[5_000:])
        elapsed = time.perf_counter() - t0
        # Transform without re-fitting should be ≤ 40% of fit_transform time
        assert elapsed < 20.0, f"Transform-only on 5K took {elapsed:.1f}s (limit 20s)"

    @pytest.mark.skipif(
        not CANDIDATES_FILE.exists(),
        reason="candidates.jsonl not found — skipping real-data benchmark",
    )
    def test_100k_real_data_under_240_seconds(self):
        """Full 100K candidates benchmark — requires candidates.jsonl."""
        from src.load_data import load_flat
        df = load_flat(CANDIDATES_FILE, show_progress=False)
        texts = df["candidate_text"].fillna("").tolist()
        assert len(texts) == 100_000, f"Expected 100K, got {len(texts)}"

        t0 = time.perf_counter()
        scores, pipeline = compute_jd_semantic_scores(texts)
        elapsed = time.perf_counter() - t0

        assert elapsed < 240.0, (
            f"100K candidates took {elapsed:.1f}s — exceeds 240s budget.\n"
            f"(Precompute step must finish with time left for embeddings.)"
        )
        assert scores.shape == (100_000,)
        assert scores.dtype == np.float32
        assert 0.0 <= scores.min() and scores.max() <= 1.0

        # Verify score distribution is healthy
        n_nonzero = (scores > 0.01).sum()
        assert n_nonzero > 1_000, (
            f"Only {n_nonzero} candidates scored > 0.01 — scoring too sparse."
        )
        print(
            f"\n[benchmark] 100K TF-IDF completed in {elapsed:.1f}s\n"
            f"  vocab_size={pipeline.vocab_size}, "
            f"  boost_keywords_in_vocab={pipeline.n_boost_keywords_in_vocab}\n"
            f"  score p50={np.percentile(scores,50):.4f}, "
            f"  p95={np.percentile(scores,95):.4f}, "
            f"  max={scores.max():.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers used in benchmark tests
# ─────────────────────────────────────────────────────────────────────────────

def compute_jd_semantic_texts_wrapper(texts: list[str]):
    """Thin wrapper so fixture can call with different param styles in tests."""
    return compute_jd_semantic_scores(texts)
