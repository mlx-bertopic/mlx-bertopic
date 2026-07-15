"""Non-convex data tests: mlx-hdbscan must agree with the Cython reference
hdbscan on moons / circles (the regime where the condensed-tree stability
computation matters most). These were the cases that exposed the original
over-segmentation bug.
"""
import numpy as np
import pytest
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import adjusted_rand_score


@pytest.mark.parametrize("seed", [1, 3, 7, 11])
def test_moons_agreement(seed):
    import mlx_hdbscan
    import hdbscan

    X, _ = make_moons(400, noise=0.07, random_state=seed)
    X = X.astype(np.float32)

    ref = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5).fit_predict(X)
    mlx = mlx_hdbscan.HDBSCAN(
        min_cluster_size=15, min_samples=5, verbose=False
    ).fit_predict(X)

    # Reference finds 2 crescent clusters; mlx must agree.
    assert adjusted_rand_score(ref, mlx) >= 0.9, (
        f"moons seed={seed}: ARI={adjusted_rand_score(ref, mlx):.3f} < 0.9"
    )


@pytest.mark.parametrize("seed", [1, 5])
def test_circles_agreement(seed):
    import mlx_hdbscan
    import hdbscan

    X, _ = make_circles(400, noise=0.07, factor=0.5, random_state=seed)
    X = X.astype(np.float32)

    ref = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5).fit_predict(X)
    mlx = mlx_hdbscan.HDBSCAN(
        min_cluster_size=15, min_samples=5, verbose=False
    ).fit_predict(X)

    assert adjusted_rand_score(ref, mlx) >= 0.9, (
        f"circles seed={seed}: ARI={adjusted_rand_score(ref, mlx):.3f} < 0.9"
    )


def test_no_oversegmentation_moons():
    """Regression guard: moons must NOT explode into many small clusters."""
    import mlx_hdbscan

    X, _ = make_moons(400, noise=0.07, random_state=7)
    X = X.astype(np.float32)
    labels = mlx_hdbscan.HDBSCAN(
        min_cluster_size=15, min_samples=5, verbose=False
    ).fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    assert n_clusters <= 3, f"over-segmented: {n_clusters} clusters"
    assert n_noise < 50, f"too much noise: {n_noise}/400"
