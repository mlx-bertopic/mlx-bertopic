"""Tests for mlx-keybert."""
import numpy as np
import pytest
from mlx_keybert import KeyBERTInspired, MaximalMarginalRelevance


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_docs = 100
    n_words = 500
    dim = 64
    doc_emb = np.random.randn(n_docs, dim).astype(np.float32)
    word_emb = np.random.randn(n_words, dim).astype(np.float32)
    topics = np.array([i % 5 for i in range(n_docs)])
    vocab = [f"word_{i}" for i in range(n_words)]
    return doc_emb, word_emb, topics, vocab


def test_keybert_basic(sample_data):
    doc_emb, word_emb, topics, vocab = sample_data
    kb = KeyBERTInspired(top_n_words=5)
    result = kb.extract_topics(
        topic_ids=[0, 1, 2],
        doc_embeddings=doc_emb,
        topic_assignments=topics,
        word_embeddings=word_emb,
        vocab=vocab,
    )
    assert set(result.keys()) == {0, 1, 2}
    for tid in [0, 1, 2]:
        assert len(result[tid]) == 5
        for word, score in result[tid]:
            assert isinstance(word, str)
            assert isinstance(score, float)


def test_keybert_scores_descending(sample_data):
    doc_emb, word_emb, topics, vocab = sample_data
    kb = KeyBERTInspired(top_n_words=10)
    result = kb.extract_topics(
        topic_ids=[0],
        doc_embeddings=doc_emb,
        topic_assignments=topics,
        word_embeddings=word_emb,
        vocab=vocab,
    )
    scores = [s for _, s in result[0]]
    assert scores == sorted(scores, reverse=True)


def test_mmr_basic(sample_data):
    doc_emb, word_emb, topics, vocab = sample_data
    mmr = MaximalMarginalRelevance(top_n_words=5, diversity=0.3)
    result = mmr.extract_topics(
        topic_ids=[0, 1],
        doc_embeddings=doc_emb,
        topic_assignments=topics,
        word_embeddings=word_emb,
        vocab=vocab,
    )
    assert set(result.keys()) == {0, 1}
    for tid in [0, 1]:
        assert len(result[tid]) == 5


def test_mmr_diversity_effect(sample_data):
    """Higher diversity should produce more different keywords."""
    doc_emb, word_emb, topics, vocab = sample_data

    low_div = MaximalMarginalRelevance(top_n_words=10, diversity=0.0)
    high_div = MaximalMarginalRelevance(top_n_words=10, diversity=0.9)

    r_low = low_div.extract_topics(
        topic_ids=[0], doc_embeddings=doc_emb,
        topic_assignments=topics, word_embeddings=word_emb, vocab=vocab,
    )
    r_high = high_div.extract_topics(
        topic_ids=[0], doc_embeddings=doc_emb,
        topic_assignments=topics, word_embeddings=word_emb, vocab=vocab,
    )

    words_low = set(w for w, _ in r_low[0])
    words_high = set(w for w, _ in r_high[0])
    # Should pick at least some different words
    assert words_low != words_high


def test_empty_topic(sample_data):
    doc_emb, word_emb, topics, vocab = sample_data
    kb = KeyBERTInspired(top_n_words=5)
    result = kb.extract_topics(
        topic_ids=[99],  # non-existent topic
        doc_embeddings=doc_emb,
        topic_assignments=topics,
        word_embeddings=word_emb,
        vocab=vocab,
    )
    assert result[99] == []
