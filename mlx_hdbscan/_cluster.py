"""Cluster extraction functions for mlx-hdbscan."""

import numpy as np


def _add_points_to_condensed(node, n_points, dendrogram, size, condensed_tree, parent_cid, lam):
    """Recursively add all points under a dendrogram node to the condensed tree.

    Each point is recorded as falling out of parent_cid at the given lambda.
    """
    if node < n_points:
        condensed_tree.append((parent_cid, node, lam, 1))
        return

    # Internal dendrogram node - recurse into children
    stack = [node]
    while stack:
        n_node = stack.pop()
        if n_node < n_points:
            condensed_tree.append((parent_cid, n_node, lam, 1))
        else:
            idx = n_node - n_points
            if idx < len(dendrogram):
                left = int(dendrogram[idx, 0])
                right = int(dendrogram[idx, 1])
                stack.append(left)
                stack.append(right)


def _extract_clusters(mst_edges: np.ndarray, n_points: int, min_cluster_size: int) -> tuple:
    """Extract flat clusters from MST using the full HDBSCAN stability algorithm.

    Implements Campello et al. 2013:
    1. Build condensed tree from MST via single-linkage + Union-Find
    2. Compute cluster stability (excess of mass)
    3. Extract optimal flat clustering via bottom-up EOM selection
    4. Assign labels and probabilities

    Parameters
    ----------
    mst_edges : ndarray of shape (n_points-1, 3)
        MST edges [node_a, node_b, weight].
    n_points : int
        Total number of data points.
    min_cluster_size : int
        Minimum cluster size for condensed tree.

    Returns
    -------
    labels : ndarray of shape (n_points,)
        Cluster labels (-1 for noise).
    probabilities : ndarray of shape (n_points,)
        Membership probabilities.
    """
    # --- Step 1: Build single-linkage dendrogram ---
    # Sort MST edges by weight (ascending) for single-linkage traversal
    sorted_idx = np.argsort(mst_edges[:, 2])
    sorted_edges = mst_edges[sorted_idx]

    # Union-Find for single-linkage
    parent = np.arange(2 * n_points - 1, dtype=np.intp)
    size = np.ones(2 * n_points - 1, dtype=np.intp)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Build full single-linkage dendrogram
    # Each merge creates a new node (id = n_points, n_points+1, ...)
    # dendrogram[i] = (child_left, child_right, distance, new_size)
    dendrogram = np.zeros((n_points - 1, 4), dtype=np.float64)
    next_node = n_points

    for i in range(len(sorted_edges)):
        a, b, w = int(sorted_edges[i, 0]), int(sorted_edges[i, 1]), sorted_edges[i, 2]
        ra, rb = find(a), find(b)
        if ra == rb:
            continue

        new_size = size[ra] + size[rb]
        dendrogram[next_node - n_points] = [ra, rb, w, new_size]

        # Union under the new node
        parent[ra] = next_node
        parent[rb] = next_node
        size[next_node] = new_size
        next_node += 1

    # --- Step 2: Build condensed tree ---
    # Walk dendrogram top-down. A cluster persists as long as both children
    # of a split have >= min_cluster_size. If one child < min_cluster_size,
    # those points "fall out" as noise and the cluster continues.
    # If both children >= min_cluster_size, the cluster splits into two new clusters.

    # Condensed tree entries: (parent_cluster, child, lambda_val, child_size)
    # child can be a point (< n_points) or a cluster (>= n_points in condensed IDs)
    condensed_tree = []  # list of (parent_cid, child, lambda_val, child_size)

    # Map from dendrogram node → condensed cluster id
    # We assign condensed cluster IDs starting from n_points
    next_condensed_id = n_points
    node_to_condensed = {}  # dendrogram_node → condensed_cluster_id

    # Top-down BFS from root of dendrogram
    root_dendro = next_node - 1  # last node created
    # The root always becomes the first condensed cluster
    node_to_condensed[root_dendro] = next_condensed_id
    condensed_birth = {next_condensed_id: 0.0}  # lambda at birth
    next_condensed_id += 1

    # Stack for top-down traversal: (dendrogram_node, condensed_cluster_id)
    stack = [(root_dendro, node_to_condensed[root_dendro])]

    while stack:
        node, cid = stack.pop()
        if node < n_points:
            # Leaf point - shouldn't happen in stack, but just in case
            continue

        dendro_idx = node - n_points
        if dendro_idx < 0 or dendro_idx >= len(dendrogram):
            continue

        left = int(dendrogram[dendro_idx, 0])
        right = int(dendrogram[dendro_idx, 1])
        dist = dendrogram[dendro_idx, 2]
        lam = 1.0 / max(dist, 1e-10)

        left_size = int(size[left]) if left >= n_points else 1
        right_size = int(size[right]) if right >= n_points else 1

        # Check if children are large enough to be clusters
        left_big = left_size >= min_cluster_size
        right_big = right_size >= min_cluster_size

        if left_big and right_big:
            # Split: current cluster dies, two new clusters born
            left_cid = next_condensed_id
            next_condensed_id += 1
            right_cid = next_condensed_id
            next_condensed_id += 1

            condensed_birth[left_cid] = lam
            condensed_birth[right_cid] = lam

            condensed_tree.append((cid, left_cid, lam, left_size))
            condensed_tree.append((cid, right_cid, lam, right_size))

            node_to_condensed[left] = left_cid
            node_to_condensed[right] = right_cid

            # Continue traversal into children
            if left >= n_points:
                stack.append((left, left_cid))
            else:
                # Single point as cluster - add as point entry
                condensed_tree.append((left_cid, left, lam, 1))

            if right >= n_points:
                stack.append((right, right_cid))
            else:
                condensed_tree.append((right_cid, right, lam, 1))

        elif left_big and not right_big:
            # Right side falls out as noise/points, left continues as same cluster
            _add_points_to_condensed(right, n_points, dendrogram, size, condensed_tree, cid, lam)
            node_to_condensed[left] = cid
            if left >= n_points:
                stack.append((left, cid))
            else:
                # Single point remaining
                condensed_tree.append((cid, left, lam, 1))

        elif right_big and not left_big:
            # Left side falls out, right continues
            _add_points_to_condensed(left, n_points, dendrogram, size, condensed_tree, cid, lam)
            node_to_condensed[right] = cid
            if right >= n_points:
                stack.append((right, cid))
            else:
                condensed_tree.append((cid, right, lam, 1))

        else:
            # Neither child is big enough — all points fall out of this cluster
            _add_points_to_condensed(left, n_points, dendrogram, size, condensed_tree, cid, lam)
            _add_points_to_condensed(right, n_points, dendrogram, size, condensed_tree, cid, lam)

    # Handle edge case: no clusters formed
    if not condensed_birth:
        return -np.ones(n_points, dtype=np.int32), np.zeros(n_points, dtype=np.float32)

    # --- Step 3: Compute cluster stability (reference compute_stability semantics) ---
    # stability(C) = sum over ALL edges (C -> child) of (lambda - birth(C)) * child_size.
    # This MUST include cluster-child edges (not only single point departures), so a
    # cluster is credited for the mass persisting in it until it splits. Omitting the
    # cluster-child term under-counts wide-persistence clusters and causes EOM to
    # over-select their descendants (e.g. catastrophic over-segmentation of non-convex
    # data such as make_moons). Matches hdbscan/_hdbscan_tree.pyx:compute_stability.
    all_cluster_ids = sorted(condensed_birth.keys())
    cluster_stability = {cid: 0.0 for cid in all_cluster_ids}
    cluster_children_map = {cid: [] for cid in all_cluster_ids}

    for (parent_cid, child, lam, child_size) in condensed_tree:
        birth = condensed_birth.get(parent_cid, 0.0)
        cluster_stability[parent_cid] += (lam - birth) * child_size
        if child >= n_points and child in cluster_children_map:
            cluster_children_map[parent_cid].append(child)

    # --- Step 4: EOM extraction (bottom-up) ---
    # Find leaves (clusters with no cluster children)
    selected = {}
    subtree_stability = {}

    # Topological sort: process leaves first
    # Build traversal by walking from leaves to root
    in_degree = {cid: 0 for cid in all_cluster_ids}
    parent_of = {}
    for cid in all_cluster_ids:
        for child in cluster_children_map[cid]:
            if child in in_degree:
                parent_of[child] = cid

    # Process in reverse topological order (leaves first)
    # Simple approach: repeatedly find nodes with no unprocessed children
    processed = set()
    traversal = []
    remaining = set(all_cluster_ids)

    while remaining:
        # Find nodes whose children are all processed
        leaves_now = []
        for cid in remaining:
            children = cluster_children_map[cid]
            valid_children = [c for c in children if c in all_cluster_ids]
            if all(c in processed for c in valid_children):
                leaves_now.append(cid)
        if not leaves_now:
            # Shouldn't happen in a valid tree, but break to avoid infinite loop
            break
        for cid in leaves_now:
            traversal.append(cid)
            processed.add(cid)
            remaining.discard(cid)

    root_cluster_id = all_cluster_ids[0]

    for cid in traversal:
        # Reference get_clusters excludes the root cluster entirely from EOM
        # selection (node_list = sorted(stability, reverse=True)[:-1]). The root
        # must not be a candidate and must not deselect its descendants, otherwise
        # a dataset that forms one wide-stability cluster collapses to all-noise.
        if cid == root_cluster_id:
            continue

        children = [c for c in cluster_children_map[cid] if c in all_cluster_ids]

        if not children:
            # Leaf: always selected
            selected[cid] = True
            subtree_stability[cid] = cluster_stability[cid]
        else:
            children_total = sum(subtree_stability.get(c, 0.0) for c in children)
            own_stability = cluster_stability[cid]

            if own_stability >= children_total:
                # Select this cluster, deselect descendants
                selected[cid] = True
                subtree_stability[cid] = own_stability
                # Deselect all descendants
                q = list(children)
                while q:
                    desc = q.pop()
                    selected[desc] = False
                    q.extend(c for c in cluster_children_map.get(desc, []) if c in all_cluster_ids)
            else:
                selected[cid] = False
                subtree_stability[cid] = children_total

    # --- Step 5: Assign labels and probabilities ---
    # Reference get_clusters excludes the root cluster from selection
    # (node_list = sorted(stability, reverse=True)[:-1]); mirror that here.
    root_cluster_id = all_cluster_ids[0]
    selected_clusters = sorted(
        [cid for cid, s in selected.items() if s and cid != root_cluster_id]
    )

    if not selected_clusters:
        return -np.ones(n_points, dtype=np.int32), np.zeros(n_points, dtype=np.float32)

    # Build map: cluster_id → label
    cluster_to_label = {cid: idx for idx, cid in enumerate(selected_clusters)}

    # For each selected cluster, find max lambda of points in it
    # (needed for probability computation)
    cluster_max_lambda = {cid: 0.0 for cid in selected_clusters}

    # For each point, determine which selected cluster it belongs to and its lambda
    labels = -np.ones(n_points, dtype=np.int32)
    probabilities = np.zeros(n_points, dtype=np.float32)

    # Build point → (cluster, lambda) mapping from condensed tree
    # A point belongs to a cluster if it falls out of that cluster or a descendant
    # We need to find the deepest selected cluster that contains each point

    # For each cluster, collect which selected cluster is its ancestor (or self)
    cluster_to_selected_ancestor = {}
    for cid in all_cluster_ids:
        cur = cid
        found = None
        visited = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            if selected.get(cur, False):
                found = cur
                break
            cur = parent_of.get(cur)
        cluster_to_selected_ancestor[cid] = found

    # Now assign each point
    for (parent_cid, child, lam, child_size) in condensed_tree:
        if child < n_points:
            # This is a point that fell out of parent_cid at lambda=lam
            sel_cluster = cluster_to_selected_ancestor.get(parent_cid)
            if sel_cluster is not None:
                labels[child] = cluster_to_label[sel_cluster]
                # Track the lambda for probability
                birth = condensed_birth[sel_cluster]
                probabilities[child] = lam - birth
                if lam > cluster_max_lambda[sel_cluster]:
                    cluster_max_lambda[sel_cluster] = lam

    # Normalize probabilities
    for cid in selected_clusters:
        label = cluster_to_label[cid]
        mask = labels == label
        max_lam = cluster_max_lambda[cid]
        if max_lam > 0:
            probabilities[mask] = probabilities[mask] / max_lam
        else:
            probabilities[mask] = 1.0
        # Clip to [0, 1]
        probabilities[mask] = np.clip(probabilities[mask], 0.0, 1.0)

    return labels, probabilities
