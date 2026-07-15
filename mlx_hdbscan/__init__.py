"""mlx-hdbscan: HDBSCAN clustering on Apple Silicon via MLX.

GPU-accelerated distance computation and mutual reachability graph construction.
The MST and cluster extraction run on CPU (inherently sequential).

References:
- Campello et al., "Density-Based Clustering Based on Hierarchical Density Estimates" (2013)
- McInnes et al., "hdbscan: Hierarchical density based clustering" (JOSS 2017)
"""

__version__ = "0.1.0"

import numpy as np
import mlx.core as mx
from typing import Optional
import time

from mlx_hdbscan._distances import _pairwise_distances_mlx, _core_distances, _mutual_reachability
from mlx_hdbscan._mst import _prim_mst, _build_hierarchy, _sparse_mrd_and_mst
from mlx_hdbscan._cluster import _extract_clusters, _add_points_to_condensed


class HDBSCAN:
    """HDBSCAN clustering with MLX GPU-accelerated distance computation.

    Parameters
    ----------
    min_cluster_size : int
        Minimum number of points to form a cluster.
    min_samples : int or None
        Number of neighbors for core distance. None = min_cluster_size.
    metric : str
        Distance metric. Currently only 'euclidean'.
    algorithm : str
        'dense' (default for n < 30K): full pairwise distance matrix on GPU.
        'sparse': use precomputed KNN graph (for large datasets).
        'auto': choose based on n_samples (dense if < 30K, else sparse).
    knn_k : int
        Number of neighbors for KNN graph (sparse mode). Default: max(min_samples*2, 30).
    batch_size : int
        Batch size for distance computation (controls GPU memory usage).
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        min_cluster_size: int = 15,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        algorithm: str = "auto",
        knn_k: Optional[int] = None,
        batch_size: int = 4096,
        verbose: bool = True,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.metric = metric
        self.algorithm = algorithm
        self.knn_k = knn_k
        self.batch_size = batch_size
        self.verbose = verbose

        self.labels_ = None
        self.probabilities_ = None

    def fit(self, X: np.ndarray) -> "HDBSCAN":
        """Fit HDBSCAN on data matrix X (n_samples, n_features)."""
        n = X.shape[0]

        # Decide algorithm
        algo = self.algorithm
        if algo == "auto":
            algo = "dense" if n <= 30000 else "sparse"

        if algo == "sparse":
            return self._fit_sparse(X)
        else:
            return self._fit_dense(X)

    def fit_knn(self, knn_indices: np.ndarray, knn_distances: np.ndarray) -> "HDBSCAN":
        """Fit HDBSCAN from a precomputed KNN graph.

        Parameters
        ----------
        knn_indices : ndarray of shape (n_samples, k)
            Indices of k nearest neighbors for each point.
        knn_distances : ndarray of shape (n_samples, k)
            Distances to k nearest neighbors for each point.

        Returns
        -------
        self
        """
        t0 = time.time()
        n = knn_indices.shape[0]
        k = knn_indices.shape[1]

        if self.verbose:
            print(f"HDBSCAN (sparse/precomputed KNN): {n:,} points, k={k}, "
                  f"min_cluster={self.min_cluster_size}")

        # Step 1: Core distances (k-th neighbor distance)
        if self.verbose:
            print("  Core distances from KNN...", end=" ", flush=True)
        t1 = time.time()
        # core_dist[i] = distance to min_samples-th neighbor
        ms = min(self.min_samples, k)
        core_dist = knn_distances[:, ms - 1].astype(np.float32)
        if self.verbose:
            print(f"{time.time()-t1:.3f}s")

        # Step 2: Sparse mutual reachability graph
        if self.verbose:
            print("  Building sparse mutual reachability graph...", end=" ", flush=True)
        t1 = time.time()
        mst_edges = _sparse_mrd_and_mst(knn_indices, knn_distances, core_dist)
        if self.verbose:
            print(f"{time.time()-t1:.2f}s")

        # Step 3: Extract clusters
        if self.verbose:
            print("  Extracting clusters (EOM)...", end=" ", flush=True)
        t1 = time.time()
        self.labels_, self.probabilities_ = _extract_clusters(mst_edges, n, self.min_cluster_size)
        if self.verbose:
            print(f"{time.time()-t1:.2f}s")

        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = (self.labels_ == -1).sum()
        if self.verbose:
            print(f"  → {n_clusters} clusters, {n_noise} noise points ({time.time()-t0:.2f}s total)")

        return self

    def _fit_sparse(self, X: np.ndarray) -> "HDBSCAN":
        """Fit using sparse KNN graph (for large datasets)."""
        t0 = time.time()
        n = X.shape[0]
        k = self.knn_k or max(self.min_samples * 2, 30)

        if self.verbose:
            print(f"HDBSCAN (sparse): {n:,} points, {X.shape[1]} dims, k={k}, "
                  f"min_cluster={self.min_cluster_size}")

        # Step 1: Build KNN graph using MLX (via mlx-vis NNDescent or brute-force)
        if self.verbose:
            print("  Building KNN graph (MLX)...", end=" ", flush=True)
        t1 = time.time()
        knn_indices, knn_distances = self._build_knn(X, k)
        if self.verbose:
            print(f"{time.time()-t1:.2f}s")

        # Delegate to fit_knn
        return self.fit_knn(knn_indices, knn_distances)

    def _build_knn(self, X: np.ndarray, k: int) -> tuple:
        """Build KNN graph. Tries mlx-vis NNDescent first, falls back to brute-force MLX."""
        n = X.shape[0]
        try:
            from mlx_vis import NNDescent
            nn = NNDescent(k=k, verbose=False)
            indices, distances = nn.build(X.astype(np.float32))
            return np.array(indices, dtype=np.int32), np.array(distances, dtype=np.float32)
        except ImportError:
            pass

        # Fallback: batched brute-force KNN on MLX
        X_mx = mx.array(X.astype(np.float32))
        sq_norms = np.array(mx.sum(X_mx * X_mx, axis=1))

        indices = np.zeros((n, k), dtype=np.int32)
        distances = np.zeros((n, k), dtype=np.float32)

        batch_size = self.batch_size
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            Xi = X_mx[i:end]
            dots = Xi @ X_mx.T
            mx.eval(dots)
            dots_np = np.array(dots)
            d = sq_norms[i:end, None] + sq_norms[None, :] - 2.0 * dots_np
            np.maximum(d, 0, out=d)
            np.sqrt(d, out=d)
            # Exclude self (set diagonal to inf)
            for j in range(end - i):
                d[j, i + j] = np.inf
            # Top-k
            top_k_idx = np.argpartition(d, k, axis=1)[:, :k]
            for j in range(end - i):
                sorted_local = top_k_idx[j][np.argsort(d[j, top_k_idx[j]])]
                indices[i + j] = sorted_local
                distances[i + j] = d[j, sorted_local]

        return indices, distances

    def _fit_dense(self, X: np.ndarray) -> "HDBSCAN":
        """Fit using full pairwise distance matrix (original algorithm)."""
        t0 = time.time()
        n = X.shape[0]

        if self.verbose:
            print(f"HDBSCAN (dense): {n:,} points, {X.shape[1]} dims, "
                  f"min_cluster={self.min_cluster_size}")

        # Step 1: Pairwise distances (GPU)
        if self.verbose:
            print("  Computing pairwise distances (MLX Metal)...", end=" ", flush=True)
        t1 = time.time()
        X_mx = mx.array(X.astype(np.float32))
        dist_matrix = _pairwise_distances_mlx(X_mx, batch_size=self.batch_size)
        if self.verbose:
            print(f"{time.time()-t1:.2f}s")

        # Step 2: Core distances
        if self.verbose:
            print("  Computing core distances...", end=" ", flush=True)
        t1 = time.time()
        core_dist = _core_distances(dist_matrix, self.min_samples)
        if self.verbose:
            print(f"{time.time()-t1:.2f}s")

        # Step 3: Mutual reachability distance
        if self.verbose:
            print("  Building mutual reachability graph...", end=" ", flush=True)
        t1 = time.time()
        mrd = _mutual_reachability(dist_matrix, core_dist)
        if self.verbose:
            print(f"{time.time()-t1:.2f}s")

        # Step 4: Minimum spanning tree
        if self.verbose:
            print("  Computing MST (Prim's)...", end=" ", flush=True)
        t1 = time.time()
        mst_edges = _prim_mst(mrd)
        if self.verbose:
            print(f"{time.time()-t1:.2f}s")

        # Step 5: Extract clusters
        if self.verbose:
            print("  Extracting clusters (EOM)...", end=" ", flush=True)
        t1 = time.time()
        self.labels_, self.probabilities_ = _extract_clusters(mst_edges, n, self.min_cluster_size)
        if self.verbose:
            print(f"{time.time()-t1:.2f}s")

        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = (self.labels_ == -1).sum()
        if self.verbose:
            print(f"  → {n_clusters} clusters, {n_noise} noise points ({time.time()-t0:.2f}s total)")

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_
