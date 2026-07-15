"""Test mlx-hdbscan agreement with reference hdbscan (Cython)."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


@pytest.fixture
def blob_data():
    """Generate 3 well-separated blobs."""
    X, y_true = make_blobs(
        n_samples=300,
        n_features=3,
        centers=3,
        cluster_std=0.8,
        random_state=42,
    )
    return X.astype(np.float32), y_true


def test_cluster_agreement(blob_data):
    """MLX HDBSCAN should produce similar clusters to reference implementation."""
    import mlx_hdbscan
    import hdbscan

    X, y_true = blob_data

    # Run mlx-hdbscan
    mlx_clusterer = mlx_hdbscan.HDBSCAN(
        min_cluster_size=10, min_samples=5, verbose=False
    )
    mlx_labels = mlx_clusterer.fit_predict(X)

    # Run reference hdbscan
    ref_clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
    ref_labels = ref_clusterer.fit_predict(X)

    # Cluster count should be equal or ±1
    mlx_n = len(set(mlx_labels)) - (1 if -1 in mlx_labels else 0)
    ref_n = len(set(ref_labels)) - (1 if -1 in ref_labels else 0)
    assert abs(mlx_n - ref_n) <= 1, (
        f"Cluster count mismatch: mlx={mlx_n}, ref={ref_n}"
    )

    # ARI between the two labelings should be high
    ari = adjusted_rand_score(ref_labels, mlx_labels)
    assert ari > 0.7, f"ARI too low: {ari:.3f}"


def test_fit_predict_shape(blob_data):
    """fit_predict should return array with correct shape."""
    import mlx_hdbscan

    X, _ = blob_data

    clusterer = mlx_hdbscan.HDBSCAN(
        min_cluster_size=10, min_samples=5, verbose=False
    )
    labels = clusterer.fit_predict(X)

    assert labels.shape == (X.shape[0],)
    assert labels.dtype == np.int32
