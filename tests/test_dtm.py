"""Smoke tests for mlx-dtm: fit → topic_trends → predict → changepoints.

Covers the regression where ``predict()`` used ``mx.no_grad()`` (which does not
exist in MLX) and raised AttributeError.
"""
import numpy as np

from mlx_dtm import DynamicTopics


def _make_data(n=600, n_topics=3, seed=0):
    rng = np.random.default_rng(seed)
    # Drift: topic prevalence shifts over time so trends are non-degenerate.
    topics = []
    for i in range(n):
        weights = np.abs(np.sin(i / 80.0 + np.arange(n_topics)))
        weights /= weights.sum()
        topics.append(int(rng.choice(n_topics, p=weights)))
    timestamps = list(range(n))
    return topics, timestamps


def test_fit_predict_trends_changepoints():
    topics, timestamps = _make_data()
    model = DynamicTopics(
        n_topics=3, hidden_dim=16, window_size=100, stride=50,
        n_epochs=5, verbose=False,
    )
    model.fit(topics, timestamps)

    # topic_trends returns a DataFrame with a timestamp column.
    df = model.topic_trends(smooth=True)
    assert "timestamp" in df.columns
    assert df.shape[0] > 0
    # proportions per row should be non-negative.
    prop_cols = [c for c in df.columns if c.startswith("topic_")]
    assert len(prop_cols) == 3
    assert np.all(df[prop_cols].values >= -1e-6)

    # predict() must not raise (regression: previously mx.no_grad AttributeError).
    preds = model.predict(steps=3)
    assert preds.shape[0] == 3
    assert np.all(np.isfinite(np.array(preds)))

    cps = model.changepoints()
    assert isinstance(cps, list)


def test_raw_trends_and_predict_steps():
    topics, timestamps = _make_data(seed=1)
    model = DynamicTopics(
        n_topics=3, hidden_dim=8, window_size=100, stride=50,
        n_epochs=3, verbose=False,
    )
    model.fit(topics, timestamps)
    raw = model.topic_trends(smooth=False)
    assert raw.shape[0] > 0
    preds = model.predict(steps=5)
    assert preds.shape[0] == 5
