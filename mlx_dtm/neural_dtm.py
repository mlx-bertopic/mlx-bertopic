"""neural_dtm.py — Contextual Neural Dynamic Topic Modeling.

Tracks topic semantic drift using pre-computed embeddings.
Core idea: for each topic, compute centroid per time bin,
measure drift (cosine distance between consecutive centroids),
and extract time-specific keywords by finding words closest to
each temporal centroid.

Requires:
  - Pre-computed embeddings (e.g. from qwen3-embedding-8b)
  - BERTopic topic assignments
  - Timestamps
  - CKIP ws (word segmentation) for keyword extraction

This is conceptually different from Blei DTM (which models word
distributions with a random walk prior). Here we work directly in
the embedding space — faster, more interpretable, and naturally
compatible with any embedding model.
"""

import time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from collections import Counter


class NeuralDTM:
    """Contextual Neural Dynamic Topic Modeling on Apple Silicon.

    Tracks how topic centroids move in embedding space over time,
    detects semantic drift, and extracts time-specific keywords.

    Parameters
    ----------
    n_bins : int
        Number of time bins to divide the data into.
    min_docs_per_bin : int
        Minimum documents per bin per topic to compute a centroid.
    smooth : bool
        Apply LSTM smoothing to centroid trajectories.
    hidden_dim : int
        LSTM hidden dimension (if smooth=True).
    n_epochs : int
        LSTM training epochs.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        n_bins: int = 20,
        min_docs_per_bin: int = 10,
        smooth: bool = True,
        hidden_dim: int = 64,
        n_epochs: int = 30,
        verbose: bool = True,
    ):
        self.n_bins = n_bins
        self.min_docs_per_bin = min_docs_per_bin
        self.smooth = smooth
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.verbose = verbose

        # Fitted state
        self._centroids = None        # dict: topic_id -> (n_bins, embed_dim)
        self._drift = None            # dict: topic_id -> (n_bins-1,) cosine distances
        self._bin_edges = None        # (n_bins+1,) timestamps
        self._bin_labels = None       # (n_bins,) middle timestamps
        self._topic_ids = None
        self._embeddings = None
        self._embed_dim = None

    def fit(
        self,
        embeddings: np.ndarray,
        topics: np.ndarray,
        timestamps: np.ndarray,
        topic_ids: Optional[list[int]] = None,
    ) -> "NeuralDTM":
        """Fit the Neural DTM.

        Parameters
        ----------
        embeddings : ndarray (n_docs, embed_dim)
            Pre-computed document embeddings.
        topics : ndarray (n_docs,)
            Topic assignment per document (from BERTopic). -1 = outlier.
        timestamps : ndarray or list
            Timestamp per document (must be sortable).
        topic_ids : list[int], optional
            Which topics to track. Default: top 50 by frequency.
        """
        t0 = time.time()
        n_docs, self._embed_dim = embeddings.shape

        if self.verbose:
            print(f"NeuralDTM: {n_docs:,} docs, {self._embed_dim}d embeddings", flush=True)

        # Determine topics to track
        topics = np.asarray(topics)
        if topic_ids is None:
            counts = Counter(t for t in topics if t >= 0)
            topic_ids = [t for t, _ in counts.most_common(50)]
        self._topic_ids = topic_ids

        # Convert timestamps to numeric for binning
        ts = pd.to_datetime(timestamps)
        ts_numeric = ts.astype(np.int64) // 10**9  # seconds since epoch

        # Create time bins
        self._bin_edges = np.linspace(ts_numeric.min(), ts_numeric.max(), self.n_bins + 1)
        bin_indices = np.digitize(ts_numeric, self._bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        # Bin labels (midpoints as datetime)
        bin_mids = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2
        self._bin_labels = pd.to_datetime(bin_mids, unit='s')

        if self.verbose:
            print(f"  Time bins: {self.n_bins} ({self._bin_labels[0].strftime('%Y-%m')} to {self._bin_labels[-1].strftime('%Y-%m')})", flush=True)

        # Compute centroids per topic per bin
        if self.verbose:
            print(f"  Computing centroids for {len(topic_ids)} topics...", flush=True)

        self._centroids = {}
        for tid in topic_ids:
            topic_mask = topics == tid
            centroids_t = np.zeros((self.n_bins, self._embed_dim), dtype=np.float32)
            valid_bins = np.zeros(self.n_bins, dtype=bool)

            for b in range(self.n_bins):
                mask = topic_mask & (bin_indices == b)
                n = mask.sum()
                if n >= self.min_docs_per_bin:
                    centroids_t[b] = embeddings[mask].mean(axis=0)
                    valid_bins[b] = True

            # Interpolate missing bins (linear in embedding space)
            if valid_bins.sum() >= 2:
                valid_idx = np.where(valid_bins)[0]
                for b in range(self.n_bins):
                    if not valid_bins[b]:
                        # Find nearest valid bins
                        left = valid_idx[valid_idx < b]
                        right = valid_idx[valid_idx > b]
                        if len(left) > 0 and len(right) > 0:
                            l, r = left[-1], right[0]
                            alpha = (b - l) / (r - l)
                            centroids_t[b] = (1 - alpha) * centroids_t[l] + alpha * centroids_t[r]
                        elif len(left) > 0:
                            centroids_t[b] = centroids_t[left[-1]]
                        elif len(right) > 0:
                            centroids_t[b] = centroids_t[right[0]]

            self._centroids[tid] = centroids_t

        # Compute drift (cosine distance between consecutive centroids)
        if self.verbose:
            print(f"  Computing drift...", flush=True)

        self._drift = {}
        for tid in topic_ids:
            c = self._centroids[tid]
            # Normalize
            norms = np.sqrt((c ** 2).sum(axis=1, keepdims=True) + 1e-8)
            c_norm = c / norms
            # Cosine distance = 1 - cosine_similarity
            drift = np.array([
                1.0 - np.dot(c_norm[t], c_norm[t+1])
                for t in range(self.n_bins - 1)
            ])
            self._drift[tid] = drift

        # Optional: LSTM smoothing of centroid trajectories
        if self.smooth:
            if self.verbose:
                print(f"  LSTM smoothing (epochs={self.n_epochs})...", flush=True)
            self._smooth_centroids()

        elapsed = time.time() - t0
        if self.verbose:
            print(f"  Done: {elapsed:.1f}s", flush=True)

        return self

    def _smooth_centroids(self):
        """Apply LSTM to smooth centroid trajectories in embedding space."""
        # Stack all topic centroids: (n_topics, n_bins, embed_dim)
        # Use PCA to reduce dim for LSTM (embed_dim is too large)
        all_centroids = np.stack([self._centroids[tid] for tid in self._topic_ids])
        # PCA: project to 64d for LSTM
        mean = all_centroids.reshape(-1, self._embed_dim).mean(axis=0)
        centered = all_centroids.reshape(-1, self._embed_dim) - mean
        # Truncated SVD (only need top components)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        n_comp = min(self.hidden_dim, self._embed_dim, centered.shape[0])
        projected = centered @ Vt[:n_comp].T  # (n_topics * n_bins, n_comp)
        projected = projected.reshape(len(self._topic_ids), self.n_bins, n_comp)

        # LSTM: input (n_bins, n_comp) per topic → smoothed output
        from . import TopicLSTM
        model = TopicLSTM(n_comp, hidden_dim=self.hidden_dim, n_layers=1)
        optimizer = optim.Adam(learning_rate=0.001)

        # Self-supervised: predict next step
        X = mx.array(projected[:, :-1, :])  # (n_topics, n_bins-1, n_comp)
        Y = mx.array(projected[:, 1:, :])   # (n_topics, n_bins-1, n_comp)

        def loss_fn(model, x, y):
            # Process each topic independently
            total_loss = mx.array(0.0)
            for i in range(x.shape[0]):
                pred = model(x[i])  # (n_bins-1, n_comp)
                total_loss = total_loss + mx.mean((pred - y[i]) ** 2)
            return total_loss / x.shape[0]

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        for epoch in range(self.n_epochs):
            loss, grads = loss_and_grad(model, X, Y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        # Get smoothed projections
        smoothed_proj = []
        for i in range(len(self._topic_ids)):
            s = model(mx.array(projected[i]))
            mx.eval(s)
            smoothed_proj.append(np.array(s))

        # Reconstruct in original embedding space
        smoothed_proj = np.array(smoothed_proj)  # (n_topics, n_bins, n_comp)
        for i, tid in enumerate(self._topic_ids):
            reconstructed = smoothed_proj[i] @ Vt[:n_comp] + mean
            self._centroids[tid] = reconstructed.astype(np.float32)

        # Recompute drift after smoothing
        for tid in self._topic_ids:
            c = self._centroids[tid]
            norms = np.sqrt((c ** 2).sum(axis=1, keepdims=True) + 1e-8)
            c_norm = c / norms
            self._drift[tid] = np.array([
                1.0 - np.dot(c_norm[t], c_norm[t+1])
                for t in range(self.n_bins - 1)
            ])

    def get_drift(self, topic_id: Optional[int] = None) -> dict:
        """Get semantic drift per topic.

        Returns dict: topic_id → array of cosine distances between consecutive bins.
        """
        if topic_id is not None:
            return {topic_id: self._drift[topic_id]}
        return self._drift

    def get_centroids(self, topic_id: int) -> np.ndarray:
        """Get centroid trajectory for a topic: (n_bins, embed_dim)."""
        return self._centroids[topic_id]

    def top_drifting_topics(self, top_n: int = 10) -> list[tuple[int, float]]:
        """Return topics with highest total drift (most semantically shifted)."""
        total_drift = [(tid, d.sum()) for tid, d in self._drift.items()]
        total_drift.sort(key=lambda x: -x[1])
        return total_drift[:top_n]

    def changepoints(self, topic_id: int, threshold: float = 2.0) -> list[int]:
        """Detect time bins where a topic shifts sharply."""
        drift = self._drift[topic_id]
        mean_d = drift.mean()
        std_d = drift.std()
        return [i for i, d in enumerate(drift) if d > mean_d + threshold * std_d]

    def keywords_over_time(
        self,
        topic_id: int,
        vocab_embeddings: np.ndarray,
        vocab_words: list[str],
        top_k: int = 10,
    ) -> dict:
        """Extract keywords for each time bin by nearest words to centroid.

        Parameters
        ----------
        vocab_embeddings : ndarray (vocab_size, embed_dim)
            Embeddings of vocabulary words.
        vocab_words : list[str]
            Corresponding word strings.
        top_k : int
            Number of keywords per bin.

        Returns
        -------
        dict: bin_index → list of (word, similarity) tuples
        """
        centroids = self._centroids[topic_id]
        # Normalize vocab
        v_norms = np.sqrt((vocab_embeddings ** 2).sum(axis=1, keepdims=True) + 1e-8)
        vocab_norm = vocab_embeddings / v_norms

        result = {}
        for b in range(self.n_bins):
            c = centroids[b]
            c_norm = c / (np.sqrt((c ** 2).sum()) + 1e-8)
            sims = vocab_norm @ c_norm
            top_idx = np.argsort(sims)[::-1][:top_k]
            result[b] = [(vocab_words[i], float(sims[i])) for i in top_idx]

        return result

    def drift_dataframe(self) -> pd.DataFrame:
        """Return drift as a tidy DataFrame for plotting."""
        rows = []
        for tid in self._topic_ids:
            for b, d in enumerate(self._drift[tid]):
                rows.append({
                    "topic": tid,
                    "bin": b,
                    "timestamp": self._bin_labels[b] if b < len(self._bin_labels) else None,
                    "drift": d,
                })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Print summary of fitted model."""
        top = self.top_drifting_topics(5)
        lines = [
            f"NeuralDTM: {len(self._topic_ids)} topics × {self.n_bins} time bins",
            f"Time range: {self._bin_labels[0].strftime('%Y-%m')} → {self._bin_labels[-1].strftime('%Y-%m')}",
            f"Top drifting topics:",
        ]
        for tid, total_d in top:
            lines.append(f"  Topic {tid}: total drift = {total_d:.4f}")
        return "\n".join(lines)
