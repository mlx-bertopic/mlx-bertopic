"""NeuralDTM: schema, drift ranking, changepoints, keyword extraction.

The class shipped in 0.6.0 with no tests at all; these pin the observable
behaviour (not any claim about semantic quality).
"""

import numpy as np
import pandas as pd
import pytest

from mlx_dtm import NeuralDTM

N_BINS = 10
DOCS_PER_BIN = 40
DIM = 16
TOPICS = (0, 1, 2)


def _embeddings():
    """Topic 0 turns sharply between bins 4 and 5; topic 1 is static."""
    rng = np.random.default_rng(0)
    rows, topics, stamps = [], [], []
    dates = pd.date_range("2024-01-01", periods=N_BINS, freq="MS")
    for bin_idx, date in enumerate(dates):
        for tid in TOPICS:
            base = np.zeros(DIM, dtype=np.float32)
            base[0] = 1.0
            if tid == 0 and bin_idx >= 5:
                base[:] = 0.0
                base[1] = 1.0          # 90-degree turn, one step only
            elif tid == 2:
                base[2] = 0.01         # barely moves
            noise = rng.normal(scale=0.01, size=(DOCS_PER_BIN, DIM)).astype(np.float32)
            if tid == 0 and bin_idx >= 5:
                noise[:, 0] = 0.0
            rows.append(base + noise)
            topics.extend([tid] * DOCS_PER_BIN)
            stamps.extend([date] * DOCS_PER_BIN)
    return np.vstack(rows).astype(np.float32), np.array(topics), np.array(stamps)


@pytest.fixture(scope="module")
def fitted():
    embeddings, topics, stamps = _embeddings()
    return NeuralDTM(
        n_bins=N_BINS, min_docs_per_bin=10, smooth=False, verbose=False
    ).fit(embeddings, topics, stamps)


def test_drift_dataframe_schema(fitted):
    df = fitted.drift_dataframe()
    assert list(df.columns) == ["topic", "bin", "timestamp", "drift"]
    assert len(df) == len(TOPICS) * (N_BINS - 1)
    assert set(df["topic"]) == set(TOPICS)
    assert (df["drift"] >= -1e-6).all()
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_centroid_shape(fitted):
    assert fitted.get_centroids(0).shape == (N_BINS, DIM)


def test_get_drift_is_keyed_by_topic(fitted):
    drift = fitted.get_drift(0)
    assert list(drift) == [0]
    assert drift[0].shape == (N_BINS - 1,)
    assert set(fitted.get_drift()) == set(TOPICS)


def test_rotating_topic_has_the_largest_drift(fitted):
    ranking = fitted.top_drifting_topics(3)
    assert ranking[0][0] == 0, ranking
    total = dict(ranking)
    assert total[1] < 0.01, total        # static topic
    assert total[0] > 0.9, total         # ~1.0 cosine distance once at the turn


def test_changepoint_lands_on_the_turn(fitted):
    assert fitted.changepoints(0) == [4], fitted.changepoints(0)


def test_static_topics_report_no_changepoints(fitted):
    """Numerical noise must not be reported as a semantic shift."""
    assert fitted.changepoints(1) == []
    assert fitted.changepoints(2) == []


def test_keywords_over_time(fitted):
    rng = np.random.default_rng(1)
    vocab = [f"w{i}" for i in range(12)]
    vocab_emb = np.vstack(
        [np.eye(DIM)[i % DIM] for i in range(len(vocab))]
    ).astype(np.float32) + rng.normal(scale=0.01, size=(len(vocab), DIM)).astype(np.float32)

    # w0 aligns with the pre-turn centroid, w1 with the post-turn one.
    keywords = fitted.keywords_over_time(0, vocab_emb, vocab, top_k=3)
    assert set(keywords) == set(range(N_BINS))
    assert all(len(words) == 3 for words in keywords.values())
    assert keywords[0][0][0] == "w0", keywords[0]
    assert keywords[N_BINS - 1][0][0] == "w1", keywords[N_BINS - 1]
    assert all(isinstance(score, float) for _, score in keywords[0])


def test_topic_ids_restrict_tracking():
    embeddings, topics, stamps = _embeddings()
    model = NeuralDTM(n_bins=N_BINS, min_docs_per_bin=10, smooth=False, verbose=False)
    model.fit(embeddings, topics, stamps, topic_ids=[1])
    assert set(model.get_drift()) == {1}
    assert len(model.drift_dataframe()) == N_BINS - 1


def test_smoothing_path_runs_and_keeps_drift_finite():
    """smooth=True trains the LSTM and recomputes drift from smoothed centroids."""
    embeddings, topics, stamps = _embeddings()
    model = NeuralDTM(
        n_bins=N_BINS, min_docs_per_bin=10, smooth=True,
        hidden_dim=8, n_epochs=2, verbose=False,
    ).fit(embeddings, topics, stamps)
    drift = model.get_drift(0)[0]
    assert drift.shape == (N_BINS - 1,)
    assert np.all(np.isfinite(drift))
    assert model.top_drifting_topics(1)[0][0] == 0


def test_summary_is_readable(fitted):
    text = fitted.summary()
    assert "NeuralDTM" in text
    assert f"{N_BINS} time bins" in text
