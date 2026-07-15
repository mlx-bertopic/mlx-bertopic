"""Tests for mlx-ctfidf: verify correctness against the REAL BERTopic
ClassTfidfTransformer (golden/upstream-aligned), not a self-derived reference.
"""
import numpy as np
import pytest
import scipy.sparse as sp
from bertopic.vectorizers import ClassTfidfTransformer as BTClassTfidf

from mlx_ctfidf import CTFIDFTransformer


def _class_term(n_classes=5, vocab=200, seed=0):
    """Random class-term count matrix (Poisson bag-of-words)."""
    rng = np.random.default_rng(seed)
    return rng.poisson(2.0, size=(n_classes, vocab)).astype(np.float32)


def _bt_transform(transformer, X):
    """Run real BERTopic transformer (expects sparse input) and return dense numpy."""
    Xs = sp.csr_matrix(X.copy())
    out = transformer.fit(Xs).transform(Xs)
    if hasattr(out, "todense"):
        out = np.asarray(out.todense())
    return np.asarray(out, dtype=np.float64)


# --- Golden alignment vs real BERTopic --------------------------------------

def test_values_close_to_bertopic():
    X = _class_term(5, 200, 7)
    bt = _bt_transform(BTClassTfidf(), X)
    mlx = CTFIDFTransformer().fit_transform(X)
    np.testing.assert_allclose(mlx, bt, rtol=1e-3, atol=1e-4)


def test_topk_words_match_bertopic():
    """Top-10 words per class must be identical to real BERTopic."""
    for seed in (0, 1, 2):
        X = _class_term(6, 300, seed)
        bt = _bt_transform(BTClassTfidf(), X)
        mlx = CTFIDFTransformer().fit_transform(X)
        for c in range(X.shape[0]):
            bt_top = set(np.argsort(-bt[c])[:10])
            mlx_top = set(np.argsort(-mlx[c])[:10])
            assert bt_top == mlx_top, f"seed={seed} class={c}: bt={bt_top} mlx={mlx_top}"


def test_bm25_matches_bertopic():
    X = _class_term(5, 200, 3)
    bt = _bt_transform(BTClassTfidf(bm25_weighting=True), X)
    mlx = CTFIDFTransformer(bm25_weighting=True).fit_transform(X)
    np.testing.assert_allclose(mlx, bt, rtol=1e-3, atol=1e-4)
    for c in range(X.shape[0]):
        assert set(np.argsort(-bt[c])[:10]) == set(np.argsort(-mlx[c])[:10])


def test_reduce_frequent_matches_bertopic():
    X = _class_term(5, 200, 5)
    bt = _bt_transform(BTClassTfidf(reduce_frequent_words=True), X)
    mlx = CTFIDFTransformer(reduce_frequent_words=True).fit_transform(X)
    np.testing.assert_allclose(mlx, bt, rtol=1e-3, atol=1e-4)


def test_no_l2_normalization():
    """Output rows must NOT be unit-norm (BERTopic only L1-normalizes TF)."""
    X = _class_term(4, 100, 1)
    out = CTFIDFTransformer().fit_transform(X)
    norms = np.linalg.norm(out, axis=1)
    nonzero = norms > 1e-6
    # If L2-normalized, all norms would be ~1.0; BERTopic output is not.
    assert not np.allclose(norms[nonzero], 1.0, atol=1e-3), "output is L2-normalized (wrong)"


# --- Basic contract ---------------------------------------------------------

def test_shape():
    X = _class_term(20, 500, 9)
    out = CTFIDFTransformer().fit_transform(X)
    assert out.shape == X.shape


def test_fit_then_transform_equals_fit_transform():
    X = _class_term(25, 800, 4)
    t = CTFIDFTransformer()
    t.fit(X)
    a = t.transform(X)
    b = CTFIDFTransformer().fit_transform(X)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        CTFIDFTransformer().transform(_class_term(3, 50, 0))


def test_output_finite():
    X = _class_term(5, 100, 2)
    out = CTFIDFTransformer().fit_transform(X)
    assert np.all(np.isfinite(out))


def test_all_zero_columns_match_bertopic_and_stay_finite():
    """Columns with no term in any class (df==0) must not become NaN and must
    match BERTopic (which leaves them at 0 via its sparse path)."""
    X = _class_term(5, 100, 2)
    X[:, [10, 33, 70]] = 0  # force all-zero columns
    assert np.all(np.isfinite(CTFIDFTransformer().fit_transform(X)))

    bt = _bt_transform(BTClassTfidf(), X)
    mlx = CTFIDFTransformer().fit_transform(X)
    # The all-zero columns must be 0 in both; everything finite-matchable.
    np.testing.assert_allclose(mlx, bt, rtol=1e-3, atol=1e-4)
    assert np.all(mlx[:, [10, 33, 70]] == 0.0)
