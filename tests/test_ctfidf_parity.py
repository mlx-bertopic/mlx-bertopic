"""Parity test: MlxCTFIDF vs BERTopic's ClassTfidfTransformer.

Verifies that our MLX implementation produces identical (within float tolerance)
results to the reference BERTopic implementation on the same input.
"""
import numpy as np
import scipy.sparse as sp
import pytest

from bertopic.vectorizers import ClassTfidfTransformer
from mlx_bertopic import MlxCTFIDF


def _make_class_term_matrix(n_classes: int, vocab_size: int, density: float = 0.3, seed: int = 42):
    """Generate a realistic class-term count matrix (sparse, integer counts)."""
    rng = np.random.default_rng(seed)
    # Sparse random counts (0-20 range, mostly zeros)
    X = rng.negative_binomial(1, 0.3, size=(n_classes, vocab_size)).astype(np.float64)
    mask = rng.random((n_classes, vocab_size)) > density
    X[mask] = 0
    return sp.csr_matrix(X)


class TestCTFIDFParity:
    """Compare MlxCTFIDF output against BERTopic's ClassTfidfTransformer."""

    def test_standard_mode_small(self):
        """Standard c-TF-IDF, small matrix (20 topics × 5K vocab)."""
        X = _make_class_term_matrix(20, 5000)
        self._assert_parity(X, reduce_frequent_words=False, bm25_weighting=False)

    def test_standard_mode_medium(self):
        """Standard c-TF-IDF, medium matrix (100 topics × 20K vocab)."""
        X = _make_class_term_matrix(100, 20000, seed=123)
        self._assert_parity(X, reduce_frequent_words=False, bm25_weighting=False)

    def test_reduce_frequent_words(self):
        """With reduce_frequent_words=True."""
        X = _make_class_term_matrix(30, 8000, seed=77)
        self._assert_parity(X, reduce_frequent_words=True, bm25_weighting=False)

    def test_bm25_weighting(self):
        """With bm25_weighting=True."""
        X = _make_class_term_matrix(25, 6000, seed=99)
        self._assert_parity(X, reduce_frequent_words=False, bm25_weighting=True)

    def test_with_empty_columns(self):
        """Matrix with some all-zero columns (words that appear in no class)."""
        X = _make_class_term_matrix(15, 3000, density=0.1, seed=55)
        # Force some columns to zero
        X_dense = X.toarray()
        X_dense[:, 100:200] = 0
        X = sp.csr_matrix(X_dense)
        self._assert_parity(X, reduce_frequent_words=False, bm25_weighting=False)

    def test_with_empty_rows(self):
        """Matrix with an all-zero row (empty class — edge case)."""
        X = _make_class_term_matrix(10, 2000, seed=33)
        X_dense = X.toarray()
        X_dense[3, :] = 0  # empty class
        X = sp.csr_matrix(X_dense)
        self._assert_parity(X, reduce_frequent_words=False, bm25_weighting=False)

    def test_with_multiplier(self):
        """With a per-word idf multiplier (seed-guided modeling)."""
        X = _make_class_term_matrix(20, 5000, seed=42)
        rng = np.random.default_rng(88)
        multiplier = rng.uniform(0.5, 2.0, size=5000).astype(np.float32)

        ref = ClassTfidfTransformer(reduce_frequent_words=False, bm25_weighting=False)
        ref_result = ref.fit(X.copy(), multiplier=multiplier).transform(X.copy())

        mlx = MlxCTFIDF(reduce_frequent_words=False, bm25_weighting=False)
        mlx_result = mlx.fit(X.copy(), multiplier=multiplier).transform(X.copy())

        ref_dense = ref_result.toarray().astype(np.float32)
        mlx_dense = mlx_result.toarray().astype(np.float32)

        # Zero out NaN/inf for comparison (should not exist but defensive)
        ref_dense = np.nan_to_num(ref_dense, nan=0.0, posinf=0.0, neginf=0.0)
        mlx_dense = np.nan_to_num(mlx_dense, nan=0.0, posinf=0.0, neginf=0.0)

        np.testing.assert_allclose(mlx_dense, ref_dense, rtol=1e-3, atol=1e-5,
                                   err_msg="Multiplier mode: MlxCTFIDF != ClassTfidfTransformer")

    def _assert_parity(self, X, reduce_frequent_words: bool, bm25_weighting: bool):
        """Run both implementations and assert outputs match."""
        # BERTopic reference (NOTE: ClassTfidfTransformer mutates X in-place
        # via normalize(..., copy=False), so pass a copy to each)
        ref = ClassTfidfTransformer(
            reduce_frequent_words=reduce_frequent_words,
            bm25_weighting=bm25_weighting,
        )
        ref_result = ref.fit_transform(X.copy())

        # MLX implementation
        mlx = MlxCTFIDF(
            reduce_frequent_words=reduce_frequent_words,
            bm25_weighting=bm25_weighting,
        )
        mlx_result = mlx.fit_transform(X.copy())

        # Both should be sparse
        assert sp.issparse(ref_result), "Reference should return sparse"
        assert sp.issparse(mlx_result), "MLX should return sparse"
        assert ref_result.shape == mlx_result.shape, f"Shape mismatch: {ref_result.shape} vs {mlx_result.shape}"

        # Convert to dense for comparison
        ref_dense = ref_result.toarray().astype(np.float32)
        mlx_dense = mlx_result.toarray().astype(np.float32)

        # Zero out NaN/inf for comparison
        ref_dense = np.nan_to_num(ref_dense, nan=0.0, posinf=0.0, neginf=0.0)
        mlx_dense = np.nan_to_num(mlx_dense, nan=0.0, posinf=0.0, neginf=0.0)

        # rtol=1e-3 because float32 vs float64 paths may differ slightly
        np.testing.assert_allclose(
            mlx_dense, ref_dense, rtol=1e-3, atol=1e-5,
            err_msg=f"MlxCTFIDF != ClassTfidfTransformer "
                    f"(reduce={reduce_frequent_words}, bm25={bm25_weighting})"
        )
