"""MlxCTFIDF — BERTopic ctfidf_model backend for Apple Silicon (MLX Metal GPU).

Wraps mlx-ctfidf to match BERTopic's ClassTfidfTransformer interface.
"""
import warnings

import numpy as np
import scipy.sparse as sp

from mlx_ctfidf import CTFIDFTransformer

# Maximum number of elements before falling back to CPU-only numpy.
# 500M elements × 4 bytes = ~2 GB float32 — beyond this, MLX may OOM.
_MAX_ELEMENTS_FOR_GPU = 500_000_000


class MlxCTFIDF:
    """BERTopic-compatible c-TF-IDF using MLX Metal GPU.

    Drop-in replacement for BERTopic's ClassTfidfTransformer.
    For matrices exceeding 500M elements (~2 GB float32), automatically
    falls back to a CPU-only numpy implementation to avoid GPU OOM.
    """

    def __init__(self, reduce_frequent_words: bool = False, bm25_weighting: bool = False):
        self.reduce_frequent_words = reduce_frequent_words
        self.bm25_weighting = bm25_weighting
        # BERTopic checks these attributes for seed-guided topic modeling
        self.seed_words = None
        self.seed_multiplier = 1
        self._transformer = CTFIDFTransformer(
            reduce_frequent_words=reduce_frequent_words,
            bm25_weighting=bm25_weighting,
        )
        self._use_fallback = False
        self._idf_np: np.ndarray | None = None
        self._multiplier: np.ndarray | None = None

    def _ctfidf_numpy_fallback(self, X_dense: np.ndarray) -> np.ndarray:
        """Pure-numpy c-TF-IDF matching BERTopic's ClassTfidfTransformer (no MLX).

        Same math as ``mlx_ctfidf.CTFIDFTransformer`` and BERTopic's
        ``ClassTfidfTransformer`` (``bertopic/vectorizers/_ctfidf.py``), used
        when the matrix is too large for GPU.

        Args:
            X_dense: (n_classes, vocab_size) term frequency matrix.

        Returns:
            (n_classes, vocab_size) c-TF-IDF matrix as float32 numpy array.
            NOT L2-normalized (matching BERTopic).
        """
        X = X_dense.astype(np.float64)

        # df = total term count per word across classes (NOT binary class count).
        df = X.sum(axis=0)
        # Average number of words per class (matches BERTopic, truncated to int).
        avg_nr_samples = int(X.sum(axis=1).mean())

        if self.bm25_weighting:
            idf = np.log(1.0 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))
        else:
            idf = np.log((avg_nr_samples / df) + 1.0)
        # All-zero columns (df == 0) yield inf idf -> dense tf(0)*idf(inf)=NaN.
        # BERTopic's sparse path leaves those columns at 0; mask idf to match.
        idf = np.where(df > 0, idf, 0.0)
        self._idf_np = idf.astype(np.float32)

        # TF: L1-normalize rows.
        row_sums = np.maximum(X.sum(axis=1, keepdims=True), 1e-12)
        tf = X / row_sums

        if self.reduce_frequent_words:
            # sqrt of L1-normalized TF, before IDF (BERTopic semantics).
            tf = np.sqrt(tf)

        # TF × IDF. No L2 normalization (BERTopic does not L2-normalize output).
        result = tf * idf
        return result.astype(np.float32)

    @property
    def _idf_diag(self):
        """BERTopic save compatibility: expose IDF as sparse diagonal matrix.

        BERTopic's save_ctfidf() accesses `ctfidf_model._idf_diag.data` to
        serialize the IDF vector. We construct a sparse diagonal on the fly
        from our internal `_idf_np` array.
        """
        if self._idf_np is not None:
            return sp.diags(self._idf_np.ravel())
        # Fallback: try to get from the MLX transformer
        if hasattr(self._transformer, '_idf'):
            idf = self._transformer._idf
            if hasattr(idf, 'tolist'):
                idf_np = np.array(idf.tolist(), dtype=np.float32)
            else:
                idf_np = np.asarray(idf, dtype=np.float32)
            return sp.diags(idf_np.ravel())
        # Empty fallback
        return sp.diags([0.0])

    def fit(self, X, multiplier=None):
        """Fit c-TF-IDF. Matches BERTopic's ClassTfidfTransformer.fit() signature.

        ``multiplier`` is a per-vocabulary-word idf scaling vector (used by
        BERTopic's seed-guided / supervised modeling). Following BERTopic, it is
        consumed at fit time and applied during transform.
        """
        # Store the multiplier (None or per-word array). BERTopic bakes the
        # multiplier into idf at fit time and calls transform() without it.
        self._multiplier = (
            None if multiplier is None else np.asarray(multiplier, dtype=np.float32)
        )

        if sp.issparse(X):
            n_elements = X.shape[0] * X.shape[1]
        else:
            n_elements = np.asarray(X).shape[0] * np.asarray(X).shape[1]

        self._use_fallback = n_elements > _MAX_ELEMENTS_FOR_GPU

        if self._use_fallback:
            shape = (X.shape[0], X.shape[1])
            warnings.warn(
                f"Matrix too large for GPU ({shape}), falling back to CPU",
                UserWarning,
                stacklevel=2,
            )
            # For fit-only, compute IDF via the fallback (full fit_transform deferred to transform)
            if sp.issparse(X):
                X_dense = X.toarray().astype(np.float32)
            else:
                X_dense = np.asarray(X, dtype=np.float32)
            # Compute IDF only (BERTopic formula; transform recomputes anyway).
            df = X_dense.sum(axis=0)
            avg_nr_samples = int(X_dense.sum(axis=1).mean())
            if self.bm25_weighting:
                self._idf_np = np.log(
                    1.0 + ((avg_nr_samples - df + 0.5) / (df + 0.5))
                ).astype(np.float32)
            else:
                self._idf_np = np.log((avg_nr_samples / df) + 1.0).astype(np.float32)
        else:
            if sp.issparse(X):
                X_dense = X.toarray().astype(np.float32)
            else:
                X_dense = np.asarray(X, dtype=np.float32)
            self._transformer.fit(X_dense)
            # Populate _idf_np for save compatibility
            if hasattr(self._transformer, '_idf'):
                idf = self._transformer._idf
                if hasattr(idf, 'tolist'):
                    self._idf_np = np.array(idf.tolist(), dtype=np.float32)
                else:
                    self._idf_np = np.asarray(idf, dtype=np.float32)

        return self

    def transform(self, X, multiplier=None):
        """Transform to c-TF-IDF. Returns scipy sparse (BERTopic expectation)."""
        if sp.issparse(X):
            X_dense = X.toarray().astype(np.float32)
        else:
            X_dense = np.asarray(X, dtype=np.float32)

        if self._use_fallback:
            result = self._ctfidf_numpy_fallback(X_dense)
        else:
            result = self._transformer.transform(X_dense)

        # Effective per-word idf multiplier: an explicit transform() argument
        # wins, else the one baked in at fit() time (BERTopic's contract).
        mult = multiplier if multiplier is not None else self._multiplier
        if mult is not None:
            # Equivalent to scaling idf: result * mult == tf * (idf * mult).
            result = result * np.asarray(mult, dtype=np.float32)

        # BERTopic expects scipy sparse CSR back
        return sp.csr_matrix(result)

    def fit_transform(self, X, multiplier=None):
        """Fit and transform in one step."""
        # Store multiplier so the fallback path (which doesn't call fit) still
        # applies it, and so a later transform() picks it up.
        self._multiplier = (
            None if multiplier is None else np.asarray(multiplier, dtype=np.float32)
        )
        if sp.issparse(X):
            n_elements = X.shape[0] * X.shape[1]
        else:
            n_elements = np.asarray(X).shape[0] * np.asarray(X).shape[1]

        self._use_fallback = n_elements > _MAX_ELEMENTS_FOR_GPU

        if self._use_fallback:
            shape = (X.shape[0], X.shape[1])
            warnings.warn(
                f"Matrix too large for GPU ({shape}), falling back to CPU",
                UserWarning,
                stacklevel=2,
            )
            if sp.issparse(X):
                X_dense = X.toarray().astype(np.float32)
            else:
                X_dense = np.asarray(X, dtype=np.float32)
            result = self._ctfidf_numpy_fallback(X_dense)
            if multiplier is not None:
                result = result * np.asarray(multiplier, dtype=np.float32)
            return sp.csr_matrix(result)
        else:
            return self.fit(X, multiplier).transform(X, multiplier)
