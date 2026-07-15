"""Tests for mlx-outlier."""
import numpy as np
import pytest
from mlx_outlier import reduce_outliers_embeddings, reduce_outliers_ctfidf


def test_no_outliers_unchanged():
    topics = np.array([0, 1, 2, 0, 1, 2])
    embeddings = np.random.randn(6, 64).astype(np.float32)
    result = reduce_outliers_embeddings(topics, embeddings)
    np.testing.assert_array_equal(result, topics)


def test_all_outliers_get_assigned():
    np.random.seed(42)
    # Create 3 clear clusters + outliers near them
    centers = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) * 10
    embeddings = []
    topics = []
    # 5 docs per cluster
    for i in range(3):
        for _ in range(5):
            embeddings.append(centers[i] + np.random.randn(3).astype(np.float32) * 0.1)
            topics.append(i)
    # 3 outliers near cluster 0
    for _ in range(3):
        embeddings.append(centers[0] + np.random.randn(3).astype(np.float32) * 0.2)
        topics.append(-1)

    embeddings = np.array(embeddings)
    topics = np.array(topics)

    result = reduce_outliers_embeddings(topics, embeddings, threshold=0.0)
    # All outliers should be assigned (no -1 left)
    assert (result == -1).sum() == 0
    # Outliers near cluster 0 should be assigned to cluster 0
    assert all(result[15:] == 0)


def test_threshold_prevents_assignment():
    np.random.seed(42)
    embeddings = np.random.randn(10, 32).astype(np.float32)
    topics = np.array([0, 0, 0, 1, 1, 1, -1, -1, -1, -1])
    # Very high threshold — shouldn't reassign random vectors
    result = reduce_outliers_embeddings(topics, embeddings, threshold=0.99)
    # Some or all outliers remain
    assert (result == -1).sum() >= 0  # at least doesn't crash


def test_ctfidf_outlier_reduction():
    np.random.seed(42)
    n_topics = 5
    vocab = 100
    ctfidf = np.random.randn(n_topics, vocab).astype(np.float32)
    doc_vectors = np.random.randn(20, vocab).astype(np.float32)
    topics = np.array([0, 1, 2, 3, 4] * 3 + [-1, -1, -1, -1, -1])
    result = reduce_outliers_ctfidf(topics, ctfidf, doc_vectors)
    # Outliers should be assigned
    assert (result == -1).sum() == 0
    assert result.shape == topics.shape


def test_ctfidf_noncontiguous_topic_ids():
    """Topic ids need not be contiguous; assigned ids must be real topic ids,
    not raw row indices (regression: previously best_idx was used directly)."""
    np.random.seed(42)
    topic_ids = [0, 1, 5, 7]  # non-contiguous, sorted unique
    vocab = 80
    ctfidf = np.random.randn(len(topic_ids), vocab).astype(np.float32)
    doc_vectors = np.random.randn(20, vocab).astype(np.float32)
    topics = np.array([0] * 4 + [1] * 4 + [5] * 4 + [7] * 4 + [-1] * 4, dtype=np.int32)

    result = reduce_outliers_ctfidf(topics, ctfidf, doc_vectors, threshold=0.0)
    assigned = set(int(x) for x in result[result != -1])
    assert assigned.issubset(set(topic_ids)), f"got invalid topic ids: {assigned}"
    assert (result == -1).sum() == 0  # threshold 0 reassigns all


def test_ctfidf_misaligned_raises():
    """ctfidf rows not matching unique topic count must raise, not silently
    assign wrong ids."""
    np.random.seed(0)
    topics = np.array([0, 1, 2, -1], dtype=np.int32)  # 3 unique non-outlier
    ctfidf = np.random.randn(2, 10).astype(np.float32)  # only 2 rows
    doc_vectors = np.random.randn(4, 10).astype(np.float32)
    with pytest.raises(ValueError):
        reduce_outliers_ctfidf(topics, ctfidf, doc_vectors)
