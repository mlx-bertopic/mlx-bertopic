"""Tests for mlx-lda: batch and online fit_transform must both return a valid
document-topic distribution. Covers the regression where online mode returned
np.array(None) because ``_doc_topic`` was None.
"""
import numpy as np

from mlx_lda import LDA


# 3 topics x 12 docs each, with topic-specific vocabulary (min_df=2 satisfied).
DOCS = (
    ["the team won the soccer game last night",
     "soccer players trained for the big game",
     "the team played a great soccer game",
     "players and the team love soccer"]
    + ["the fast chip runs the new software",
     "software runs on the fast new chip",
     "the new chip runs fast software",
     "fast software needs a fast new chip"]
    + ["the chef cooked fresh pasta with garlic",
     "fresh garlic makes the pasta taste great",
     "the chef used fresh garlic in the pasta",
     "great pasta needs fresh garlic and care"]
)


def test_batch_fit_transform():
    m = LDA(n_topics=3, n_iter=15, learning_method="batch", verbose=False)
    dt = m.fit_transform(DOCS)
    assert dt.shape == (len(DOCS), 3)
    assert np.all(np.isfinite(dt))
    np.testing.assert_allclose(dt.sum(axis=1), 1.0, atol=0.05)


def test_online_fit_transform():
    # Previously crashed: returned np.array(None) because _doc_topic is None online.
    m = LDA(n_topics=3, n_iter=3, learning_method="online", batch_size=6, verbose=False)
    dt = m.fit_transform(DOCS)
    assert dt.shape == (len(DOCS), 3)
    assert np.all(np.isfinite(dt))
    np.testing.assert_allclose(dt.sum(axis=1), 1.0, atol=0.05)


def test_auto_fit_transform():
    m = LDA(n_topics=3, n_iter=10, learning_method="auto", verbose=False)
    dt = m.fit_transform(DOCS)
    assert dt.shape == (len(DOCS), 3)
    assert np.all(np.isfinite(dt))


def test_get_topics_and_transform_new_docs():
    m = LDA(n_topics=3, n_iter=15, learning_method="batch", verbose=False)
    m.fit(DOCS)
    topics = m.get_topics(n_words=5)
    assert len(topics) == 3
    for topic in topics:
        assert len(topic) == 5
    new_dt = m.transform(DOCS[:4])
    assert new_dt.shape == (4, 3)
    np.testing.assert_allclose(new_dt.sum(axis=1), 1.0, atol=0.05)


def test_doc_topic_matrix_after_batch():
    m = LDA(n_topics=3, n_iter=10, learning_method="batch", verbose=False)
    m.fit(DOCS)
    dtm = m.doc_topic_matrix()
    assert dtm.shape == (len(DOCS), 3)
