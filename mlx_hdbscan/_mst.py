"""Minimum spanning tree and hierarchy construction for mlx-hdbscan."""

import numpy as np


def _prim_mst(mrd: np.ndarray) -> np.ndarray:
    """Minimum spanning tree via Prim's algorithm.

    Returns edges as (n-1, 3) array: [node_a, node_b, weight].
    """
    n = mrd.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    min_edge = np.full(n, np.inf, dtype=np.float32)
    min_edge_node = np.zeros(n, dtype=np.int32)
    edges = np.zeros((n - 1, 3), dtype=np.float32)

    # Start from node 0
    current = 0
    in_tree[current] = True
    min_edge[0] = 0

    for i in range(n - 1):
        # Update minimum edges from current node
        dists = mrd[current]
        mask = ~in_tree
        update = mask & (dists < min_edge)
        min_edge[update] = dists[update]
        min_edge_node[update] = current

        # Find next node to add (minimum edge not in tree)
        candidates = np.where(~in_tree)[0]
        next_idx = candidates[np.argmin(min_edge[candidates])]

        edges[i] = [min_edge_node[next_idx], next_idx, min_edge[next_idx]]
        in_tree[next_idx] = True
        current = next_idx

    return edges


def _build_hierarchy(mst_edges: np.ndarray) -> np.ndarray:
    """Build HDBSCAN hierarchy from MST.

    Sort edges by weight, then perform single-linkage clustering.
    Returns sorted MST edges.
    """
    sorted_idx = np.argsort(mst_edges[:, 2])
    return mst_edges[sorted_idx]


def _sparse_mrd_and_mst(knn_indices: np.ndarray, knn_distances: np.ndarray,
                         core_dist: np.ndarray) -> np.ndarray:
    """Build sparse mutual reachability graph from KNN and compute MST.

    Parameters
    ----------
    knn_indices : (n, k) int32 — neighbor indices
    knn_distances : (n, k) float32 — neighbor distances
    core_dist : (n,) float32 — core distances

    Returns
    -------
    mst_edges : (n-1, 3) float32 — [node_a, node_b, weight]
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

    n, k = knn_indices.shape

    # Vectorized sparse MRD construction
    # Flatten KNN into edge list
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = knn_indices.ravel().astype(np.int32)
    dists = knn_distances.ravel().astype(np.float32)

    # Filter invalid indices
    valid = (cols >= 0) & (cols < n)
    rows = rows[valid]
    cols = cols[valid]
    dists = dists[valid]

    # MRD: max(core[i], core[j], dist(i,j))
    mrd_vals = np.maximum(np.maximum(core_dist[rows], core_dist[cols]), dists)

    # Build sparse symmetric MRD matrix
    # Combine (i→j) and (j→i) edges, keep minimum for duplicates
    sparse_mrd = csr_matrix((mrd_vals, (rows, cols)), shape=(n, n))
    sparse_t = sparse_mrd.T.tocsr()
    # Symmetric: take the minimum where both directions exist
    # maximum gives the union (sparse addition would double shared edges)
    sparse_sym = sparse_mrd.maximum(sparse_t)

    # Compute MST via scipy (Kruskal's on sparse graph)
    mst_sparse = minimum_spanning_tree(sparse_sym)
    mst_coo = mst_sparse.tocoo()
    n_edges = len(mst_coo.row)

    if n_edges < n - 1:
        # Graph is not fully connected — connect components with high-weight edges
        n_components, comp_labels = connected_components(sparse_sym, directed=False)
        if n_components > 1:
            max_weight = mst_coo.data.max() * 2 if n_edges > 0 else 1.0
            extra_rows = []
            extra_cols = []
            extra_data = []
            prev_rep = np.where(comp_labels == 0)[0][0]
            for c in range(1, n_components):
                comp_rep = np.where(comp_labels == c)[0][0]
                extra_rows.append(prev_rep)
                extra_cols.append(comp_rep)
                extra_data.append(max_weight)
                prev_rep = comp_rep

            all_rows = np.concatenate([mst_coo.row, extra_rows])
            all_cols = np.concatenate([mst_coo.col, extra_cols])
            all_data = np.concatenate([mst_coo.data, extra_data])
            return np.column_stack([all_rows, all_cols, all_data]).astype(np.float32)

    return np.column_stack([
        mst_coo.row.astype(np.float32),
        mst_coo.col.astype(np.float32),
        mst_coo.data.astype(np.float32),
    ])
