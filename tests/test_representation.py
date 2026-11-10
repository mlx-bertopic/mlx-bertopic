"""MlxKeyBERT / MlxMMR must work as BERTopic `representation_model=`.

Regression: they used to be plain classes, so BERTopic 0.17's isinstance check
in `_extract_words_per_topic` raised
``TypeError: unsupported type MlxKeyBERT for representation_model``.
"""

import warnings
import zlib

import numpy as np
import pytest
from bertopic import BERTopic
from bertopic.representation import BaseRepresentation
from hdbscan import HDBSCAN

from mlx_bertopic import MlxCTFIDF, MlxUMAPWrapper
from mlx_bertopic.representation import MlxKeyBERT, MlxMMR, _get_embeddings


class FakeEmbedder:
    """Offline deterministic embedder: CRC32-seeded vector per string."""

    def embed(self, documents, verbose=False):
        out = np.zeros((len(documents), 8), dtype=np.float32)
        for i, doc in enumerate(documents):
            out[i] = np.random.default_rng(zlib.crc32(doc.encode())).standard_normal(8)
        return out


DOCS = (
    [f"football match report {i}" for i in range(6)]
    + [f"stock market earnings {i}" for i in range(6)]
    + [f"python library release {i}" for i in range(6)]
    + [f"weather storm warning {i}" for i in range(6)]
)


def _fit(representation_model):
    embedder = FakeEmbedder()
    embeddings = embedder.embed(DOCS)
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=MlxUMAPWrapper(n_components=5, n_neighbors=5),
        hdbscan_model=HDBSCAN(min_cluster_size=4, min_samples=2),
        ctfidf_model=MlxCTFIDF(),
        representation_model=representation_model,
        calculate_probabilities=False,
        verbose=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        topics, _ = topic_model.fit_transform(DOCS, embeddings=embeddings)
    skipped = [str(w.message) for w in caught if "skipping GPU representation" in str(w.message)]
    assert not skipped, skipped
    return topic_model, list(topics)


@pytest.mark.parametrize(
    "representation_model",
    [MlxKeyBERT(top_n_words=5), MlxMMR(top_n_words=5, diversity=0.2)],
    ids=["MlxKeyBERT", "MlxMMR"],
)
def test_representation_model_is_accepted_by_bertopic(representation_model):
    assert isinstance(representation_model, BaseRepresentation)
    topic_model, topics = _fit(representation_model)
    assert len(topics) == len(DOCS)
    topic_ids = sorted({t for t in topics if t >= 0})
    assert topic_ids, "no clusters found"
    for tid in topic_ids:
        words = topic_model.get_topic(tid)
        assert words, f"topic {tid} has no words"
        assert all(isinstance(word, str) and isinstance(score, float) for word, score in words)


def test_get_embeddings_falls_back_to_embedding_model():
    """BERTopic >= 0.17 keeps no embeddings on the model; re-embed instead."""

    class StubModel:
        embedding_model = FakeEmbedder()

    import pandas as pd

    documents = pd.DataFrame({"Document": DOCS, "Topic": [0] * len(DOCS)})
    embeddings = _get_embeddings(StubModel(), documents)
    assert embeddings.shape == (len(DOCS), 8)
    assert np.array_equal(embeddings, FakeEmbedder().embed(DOCS))


def test_get_embeddings_prefers_cached_embeddings():
    cached = np.ones((3, 4), dtype=np.float32)

    class StubModel:
        _embeddings = cached
        embedding_model = FakeEmbedder()

    assert _get_embeddings(StubModel(), None) is cached


def test_get_embeddings_without_embedding_model_returns_none():
    class StubModel:
        embedding_model = None

    assert _get_embeddings(StubModel(), None) is None
