"""mlx-outlier — BERTopic outlier reduction on Apple Silicon via MLX (Metal GPU).

Embedding-based and centroid-based outlier reduction strategies,
accelerated on Metal GPU through MLX batch cosine similarity.
"""

import mlx.core as mx
import numpy as np

__version__ = "0.1.0"


def reduce_outliers_embeddings(
    topics: np.ndarray,
    embeddings: np.ndarray,
    threshold: float = 0.0,
    batch_size: int = 4096,
) -> np.ndarray:
    """Reduce outliers by assigning them to the nearest topic centroid.

    For each document labeled -1 (outlier), compute cosine similarity
    to all topic centroids and assign to the closest one if above threshold.

    Args:
        topics: (N,) topic assignments, -1 = outlier.
        embeddings: (N, D) document embeddings.
        threshold: minimum cosine similarity to reassign (0.0 = always reassign).
        batch_size: process outliers in batches to limit memory.

    Returns:
        (N,) new topic assignments with reduced outliers.
    """
    topics = np.array(topics, dtype=np.int32)
    new_topics = topics.copy()

    # Find unique non-outlier topics
    unique_topics = np.unique(topics[topics >= 0])
    if len(unique_topics) == 0:
        return new_topics

    # Compute centroids: index with numpy, compute mean with MLX
    emb_f32 = embeddings.astype(np.float32)
    centroids = []
    for t in unique_topics:
        idx = np.where(topics == t)[0]
        centroid = mx.mean(mx.array(emb_f32[idx]), axis=0)
        centroids.append(centroid)

    # Stack centroids: (n_topics, D)
    centroids_mx = mx.stack(centroids, axis=0)
    # L2-normalize centroids
    centroid_norms = mx.maximum(mx.sqrt(mx.sum(centroids_mx * centroids_mx, axis=1, keepdims=True)), 1e-9)
    centroids_mx = centroids_mx / centroid_norms
    mx.eval(centroids_mx)

    # Process outliers in batches
    outlier_idx = np.where(topics == -1)[0]
    if len(outlier_idx) == 0:
        return new_topics

    for i in range(0, len(outlier_idx), batch_size):
        batch_idx = outlier_idx[i:i + batch_size]
        batch_emb = mx.array(emb_f32[batch_idx])

        # L2-normalize batch
        batch_norms = mx.maximum(mx.sqrt(mx.sum(batch_emb * batch_emb, axis=1, keepdims=True)), 1e-9)
        batch_emb = batch_emb / batch_norms

        # Cosine similarity: (batch, n_topics)
        sims = batch_emb @ centroids_mx.T
        mx.eval(sims)

        sims_np = np.array(sims)
        best_topic_idx = sims_np.argmax(axis=1)
        best_sim = sims_np[np.arange(len(best_topic_idx)), best_topic_idx]

        # Assign if above threshold
        assign_mask = best_sim >= threshold
        for j, idx in enumerate(batch_idx):
            if assign_mask[j]:
                new_topics[idx] = unique_topics[best_topic_idx[j]]

    return new_topics


def reduce_outliers_ctfidf(
    topics: np.ndarray,
    ctfidf_matrix: np.ndarray,
    doc_vectors: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """Reduce outliers using c-TF-IDF similarity.

    Args:
        topics: (N,) topic assignments.
        ctfidf_matrix: (n_topics, vocab) c-TF-IDF topic representations.
        doc_vectors: (N, vocab) document term vectors (sparse → dense).
        threshold: minimum similarity for reassignment.

    Returns:
        (N,) new topic assignments.
    """
    topics = np.array(topics, dtype=np.int32)
    new_topics = topics.copy()

    outlier_idx = np.where(topics == -1)[0]
    if len(outlier_idx) == 0:
        return new_topics

    # ctfidf_matrix rows are assumed to align with the sorted unique non-outlier
    # topic ids (the BERTopic convention). Map each row index back to the actual
    # topic id so non-contiguous ids (e.g. 0,1,5,7 after merge/reduce) are handled
    # correctly -- previously the raw row index was used as the topic id.
    unique_topics = np.unique(topics[topics >= 0])
    if ctfidf_matrix.shape[0] != len(unique_topics):
        raise ValueError(
            f"ctfidf_matrix has {ctfidf_matrix.shape[0]} rows but there are "
            f"{len(unique_topics)} unique non-outlier topics; rows must align "
            f"with sorted unique topic ids."
        )

    # Move to MLX
    ctfidf_mx = mx.array(ctfidf_matrix.astype(np.float32))
    # L2-normalize topic vectors
    t_norms = mx.maximum(mx.sqrt(mx.sum(ctfidf_mx * ctfidf_mx, axis=1, keepdims=True)), 1e-9)
    ctfidf_mx = ctfidf_mx / t_norms
    mx.eval(ctfidf_mx)

    # Process in batches
    batch_size = 2048
    for i in range(0, len(outlier_idx), batch_size):
        batch_idx = outlier_idx[i:i + batch_size]
        batch_docs = mx.array(doc_vectors[batch_idx].astype(np.float32))

        d_norms = mx.maximum(mx.sqrt(mx.sum(batch_docs * batch_docs, axis=1, keepdims=True)), 1e-9)
        batch_docs = batch_docs / d_norms

        sims = batch_docs @ ctfidf_mx.T
        mx.eval(sims)

        sims_np = np.array(sims)
        best_idx = sims_np.argmax(axis=1)
        best_sim = sims_np[np.arange(len(best_idx)), best_idx]

        assign_mask = best_sim >= threshold
        for j, idx in enumerate(batch_idx):
            if assign_mask[j]:
                new_topics[idx] = unique_topics[best_idx[j]]

    return new_topics


__all__ = ["reduce_outliers_embeddings", "reduce_outliers_ctfidf"]
