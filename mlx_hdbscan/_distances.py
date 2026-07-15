"""Distance computation functions for mlx-hdbscan."""

import numpy as np
import mlx.core as mx


def _pairwise_distances_mlx(X: mx.array, batch_size: int = 4096) -> np.ndarray:
    """Compute pairwise Euclidean distances using MLX Metal GPU.

    Uses the identity: ||a-b||² = ||a||² + ||b||² - 2*a·b
    Processes in batches to avoid OOM on large datasets.

    Returns numpy array (n, n) of distances.
    """
    n = X.shape[0]
    # Precompute squared norms
    sq_norms = mx.sum(X * X, axis=1)  # (n,)
    mx.eval(sq_norms)
    sq_norms_np = np.array(sq_norms)

    # Compute full distance matrix in batches
    dist = np.zeros((n, n), dtype=np.float32)

    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        Xi = X[i:end_i]  # (batch, d)
        # dot products: Xi @ X.T → (batch, n)
        dots = Xi @ X.T
        mx.eval(dots)
        dots_np = np.array(dots)

        # distances: sq_norms[i:end_i, None] + sq_norms[None, :] - 2 * dots
        d = sq_norms_np[i:end_i, None] + sq_norms_np[None, :] - 2.0 * dots_np
        np.maximum(d, 0, out=d)  # numerical fix
        dist[i:end_i] = d

    np.sqrt(dist, out=dist)
    return dist


def _core_distances(dist_matrix: np.ndarray, min_samples: int) -> np.ndarray:
    """Compute core distances: distance to the k-th nearest neighbor.
    Uses batch processing for memory efficiency on large matrices.
    """
    n = dist_matrix.shape[0]
    # For moderate n, np.partition on full matrix is fast
    if n <= 30000:
        kth_dists = np.partition(dist_matrix, min_samples, axis=1)[:, min_samples]
        return kth_dists.astype(np.float32)
    # For very large n, process in chunks
    core_dist = np.zeros(n, dtype=np.float32)
    chunk = 4096
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        kth = np.partition(dist_matrix[i:end], min_samples, axis=1)[:, min_samples]
        core_dist[i:end] = kth
    return core_dist


def _mutual_reachability(dist_matrix: np.ndarray, core_distances: np.ndarray) -> np.ndarray:
    """Compute mutual reachability distance matrix.

    mrd(a,b) = max(core_dist(a), core_dist(b), dist(a,b))
    """
    n = dist_matrix.shape[0]
    # Vectorized: mrd = max(dist, core_a broadcast, core_b broadcast)
    mrd = dist_matrix.copy()
    np.maximum(mrd, core_distances[:, None], out=mrd)
    np.maximum(mrd, core_distances[None, :], out=mrd)
    return mrd
