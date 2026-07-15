"""mlx-lda: Fast LDA topic modeling on Apple Silicon via MLX.

Implements batch Variational EM (Blei et al., 2003) with MLX GPU acceleration.
The E-step (per-doc variational inference) and M-step (topic-word update) both
use Metal GPU via MLX matrix operations.
"""

import numpy as np
import mlx.core as mx
from sklearn.feature_extraction.text import CountVectorizer
from typing import Optional
import time

__version__ = "0.1.0"


class LDA:
    """Latent Dirichlet Allocation on MLX (Metal GPU).

    Parameters
    ----------
    n_topics : int
        Number of topics.
    n_iter : int
        Number of EM iterations (batch) or passes over data (online).
    alpha : float or None
        Document-topic Dirichlet prior. None = 1/n_topics.
    eta : float
        Topic-word Dirichlet prior.
    learning_method : str
        'batch': load all docs in GPU at once (fast, memory-heavy).
        'online': mini-batch variational inference (memory-efficient, scalable).
        'auto': choose based on estimated memory usage.
    batch_size : int
        Mini-batch size for online mode. Default 512.
    memory_limit_gb : float
        GPU memory budget for auto mode. Default 64.
    random_state : int or None
        Random seed.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        n_topics: int = 10,
        n_iter: int = 30,
        alpha: Optional[float] = None,
        eta: float = 0.01,
        learning_method: str = "auto",
        batch_size: int = 512,
        memory_limit_gb: float = 64.0,
        random_state: Optional[int] = 42,
        verbose: bool = True,
    ):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.alpha = alpha or (1.0 / n_topics)
        self.eta = eta
        self.learning_method = learning_method
        self.batch_size = batch_size
        self.memory_limit_gb = memory_limit_gb
        self.random_state = random_state
        self.verbose = verbose

        self._beta = None  # (K, V) normalized topic-word distribution
        self._vocab = None
        self._vectorizer = None
        self._doc_topic = None  # (D, K) from last fit

    def fit(self, documents: list[str]) -> "LDA":
        """Fit LDA model on documents."""
        t0 = time.time()
        K = self.n_topics

        # Vectorize
        if self.verbose:
            print(f"Vectorizing {len(documents):,} documents...")
        self._vectorizer = CountVectorizer(
            min_df=2, max_df=0.95, token_pattern=r"(?u)\b\w\w+\b"
        )
        tf_sparse = self._vectorizer.fit_transform(documents)
        self._vocab = self._vectorizer.get_feature_names_out()
        n_docs, V = tf_sparse.shape

        if self.verbose:
            print(f"  Vocab: {V:,} | Docs: {n_docs:,} | Topics: {K}")

        # Choose learning method
        method = self.learning_method
        if method == "auto":
            # Estimate memory: docs × vocab × 4 bytes × 3 (tf + theta + denom)
            est_gb = (n_docs * V * 4 * 3) / (1024**3)
            method = "batch" if est_gb < self.memory_limit_gb else "online"
            if self.verbose:
                print(f"  Memory estimate: {est_gb:.1f}GB → using '{method}' mode")

        if method == "online":
            return self._fit_online(tf_sparse, n_docs, V, K, t0)
        else:
            return self._fit_batch(tf_sparse, n_docs, V, K, t0)

    def _fit_batch(self, tf_sparse, n_docs, V, K, t0) -> "LDA":
        """Batch variational EM (all docs in GPU at once)."""
        # Convert to dense MLX
        tf = mx.array(tf_sparse.toarray().astype(np.float32))

        # Initialize beta (topic-word) randomly and normalize
        rng = np.random.default_rng(self.random_state)
        beta = rng.dirichlet(np.ones(V) * 0.1, size=K).astype(np.float32)
        beta_mx = mx.array(beta)  # (K, V)

        # Initialize theta (doc-topic) uniformly
        theta = mx.ones((n_docs, K)) / K  # (D, K)

        for it in range(self.n_iter):
            # E-step: update theta given beta
            denom = theta @ beta_mx + 1e-100  # (D, V)
            theta_new = self.alpha + theta * ((tf / denom) @ beta_mx.T)
            theta = theta_new / mx.sum(theta_new, axis=1, keepdims=True)
            mx.eval(theta)

            # M-step: update beta given theta
            denom = theta @ beta_mx + 1e-100
            beta_new = self.eta + beta_mx * (theta.T @ (tf / denom))
            beta_mx = beta_new / mx.sum(beta_new, axis=1, keepdims=True)
            mx.eval(beta_mx)

            if self.verbose and (it + 1) % 5 == 0:
                prob = theta @ beta_mx + 1e-100
                ll = float(mx.sum(tf * mx.log(prob)).item())
                total_w = float(mx.sum(tf).item())
                ppx = np.exp(-ll / total_w)
                elapsed = time.time() - t0
                print(f"  Iter {it+1}/{self.n_iter} | perplexity={ppx:.0f} | {elapsed:.1f}s")

        self._beta = beta_mx
        self._doc_topic = theta

        if self.verbose:
            print(f"  Done in {time.time()-t0:.1f}s")
        return self

    def _fit_online(self, tf_sparse, n_docs, V, K, t0) -> "LDA":
        """Online (mini-batch) variational inference.

        Based on Hoffman et al. (2010) "Online Learning for LDA".
        Memory usage: O(batch_size × V) instead of O(n_docs × V).
        """
        rng = np.random.default_rng(self.random_state)
        batch_size = self.batch_size

        # Initialize lambda (unnormalized topic-word sufficient statistics)
        # lambda_kv = expected count of word v in topic k
        lam = rng.gamma(100.0, 1.0 / 100.0, (K, V)).astype(np.float32)
        lam_mx = mx.array(lam)

        # Normalize to get beta
        beta_mx = lam_mx / mx.sum(lam_mx, axis=1, keepdims=True)
        mx.eval(beta_mx)

        n_batches = (n_docs + batch_size - 1) // batch_size
        total_batches = n_batches * self.n_iter

        # Online learning rate: rho_t = (tau + t)^{-kappa}
        tau = 64.0
        kappa = 0.7

        batch_count = 0
        for epoch in range(self.n_iter):
            # Shuffle document order each epoch
            perm = rng.permutation(n_docs)

            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_docs)
                idx = perm[start:end]
                mb_size = len(idx)

                # Get mini-batch term-frequency
                tf_batch = mx.array(tf_sparse[idx].toarray().astype(np.float32))

                # E-step: local variational inference on mini-batch
                theta_b = mx.ones((mb_size, K)) / K
                for _ in range(5):  # local iterations
                    denom = theta_b @ beta_mx + 1e-100
                    theta_b = self.alpha + theta_b * ((tf_batch / denom) @ beta_mx.T)
                    theta_b = theta_b / mx.sum(theta_b, axis=1, keepdims=True)
                mx.eval(theta_b)

                # M-step: compute sufficient statistics from mini-batch
                denom = theta_b @ beta_mx + 1e-100
                ss = self.eta + beta_mx * (theta_b.T @ (tf_batch / denom))
                # Scale to full corpus size
                ss = ss * (n_docs / mb_size)
                mx.eval(ss)

                # Online update with learning rate
                batch_count += 1
                rho = (tau + batch_count) ** (-kappa)
                lam_mx = (1 - rho) * lam_mx + rho * ss
                beta_mx = lam_mx / mx.sum(lam_mx, axis=1, keepdims=True)
                mx.eval(beta_mx)

            if self.verbose and (epoch + 1) % max(1, self.n_iter // 10) == 0:
                # Estimate perplexity on a sample
                sample_idx = rng.choice(n_docs, size=min(1000, n_docs), replace=False)
                tf_sample = mx.array(tf_sparse[sample_idx].toarray().astype(np.float32))
                theta_s = mx.ones((len(sample_idx), K)) / K
                for _ in range(10):
                    denom = theta_s @ beta_mx + 1e-100
                    theta_s = self.alpha + theta_s * ((tf_sample / denom) @ beta_mx.T)
                    theta_s = theta_s / mx.sum(theta_s, axis=1, keepdims=True)
                mx.eval(theta_s)
                prob = theta_s @ beta_mx + 1e-100
                ll = float(mx.sum(tf_sample * mx.log(prob)).item())
                total_w = float(mx.sum(tf_sample).item())
                ppx = np.exp(-ll / total_w)
                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1}/{self.n_iter} | perplexity≈{ppx:.0f} | {elapsed:.1f}s")

        self._beta = beta_mx
        self._doc_topic = None  # online mode doesn't keep full doc-topic

        if self.verbose:
            print(f"  Done in {time.time()-t0:.1f}s (online, {batch_count} batches)")
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        """Infer document-topic distribution for new documents."""
        if self._beta is None:
            raise ValueError("Model not fitted.")
        tf_sparse = self._vectorizer.transform(documents)
        tf = mx.array(tf_sparse.toarray().astype(np.float32))
        K = self.n_topics
        n_docs = tf.shape[0]

        theta = mx.ones((n_docs, K)) / K
        for _ in range(30):  # inference iterations
            denom = theta @ self._beta + 1e-100
            theta_new = self.alpha + theta * ((tf / denom) @ self._beta.T)
            theta = theta_new / mx.sum(theta_new, axis=1, keepdims=True)
            mx.eval(theta)

        return np.array(theta)

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        """Fit and return document-topic distribution.

        In online mode ``fit`` does not retain the full doc-topic matrix
        (``_doc_topic`` is None), so infer it here via ``transform`` instead of
        returning ``np.array(None)`` (which previously broke downstream code).
        """
        self.fit(documents)
        if self._doc_topic is None:
            return self.transform(documents)
        return np.array(self._doc_topic)

    def get_topics(self, n_words: int = 10) -> list[list[tuple[str, float]]]:
        """Return topics as list of (word, weight) tuples."""
        if self._beta is None:
            raise ValueError("Model not fitted.")
        beta = np.array(self._beta)
        topics = []
        for k in range(self.n_topics):
            top_idx = beta[k].argsort()[-n_words:][::-1]
            topics.append([(self._vocab[i], float(beta[k, i])) for i in top_idx])
        return topics

    def print_topics(self, n_words: int = 10):
        """Print top words per topic."""
        for k, topic in enumerate(self.get_topics(n_words)):
            words = " ".join(f"{w}({p:.4f})" for w, p in topic)
            print(f"Topic {k:2d}: {words}")

    def topic_word_matrix(self) -> np.ndarray:
        """Topic-word distribution (K, V)."""
        return np.array(self._beta)

    def doc_topic_matrix(self) -> np.ndarray:
        """Training doc-topic distribution (D, K)."""
        if self._doc_topic is None:
            raise ValueError("No training doc-topic available. Call fit() first.")
        return np.array(self._doc_topic)

    @property
    def components_(self) -> np.ndarray:
        """sklearn-compatible: topic-word matrix (unnormalized)."""
        return np.array(self._beta) * 1000  # scale for compatibility

    def save(self, path: str):
        np.savez(path, beta=np.array(self._beta), vocab=self._vocab,
                 n_topics=self.n_topics, alpha=self.alpha, eta=self.eta)

    @classmethod
    def load(cls, path: str) -> "LDA":
        data = np.load(path, allow_pickle=True)
        model = cls(n_topics=int(data["n_topics"]))
        model._beta = mx.array(data["beta"])
        model._vocab = data["vocab"]
        model.alpha = float(data["alpha"])
        model.eta = float(data["eta"])
        return model
