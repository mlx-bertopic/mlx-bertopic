"""Golden tests: MlxCTFIDF (GPU path AND numpy fallback) must match the real
BERTopic ClassTfidfTransformer, for sparse and dense input.
"""
import numpy as np
import pytest
import scipy.sparse as sp
from bertopic.vectorizers import ClassTfidfTransformer as BTClassTfidf

from mlx_bertopic import MlxCTFIDF
from mlx_bertopic import ctfidf as ctfidf_mod


def _class_term(n_classes=5, vocab=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.poisson(2.0, size=(n_classes, vocab)).astype(np.float32)


def _bt_dense(transformer, X):
    Xs = sp.csr_matrix(X.copy())
    out = transformer.fit(Xs).transform(Xs)
    if hasattr(out, "todense"):
        out = np.asarray(out.todense())
    return np.asarray(out, dtype=np.float64)


def _topk(vec, k=10):
    return set(np.argsort(-vec)[:k])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gpu_path_matches_bertopic_topk(seed):
    X = _class_term(6, 300, seed)
    bt = _bt_dense(BTClassTfidf(), X)
    mlx = MlxCTFIDF().fit_transform(X).toarray()
    for c in range(X.shape[0]):
        assert _topk(bt[c]) == _topk(mlx[c]), f"seed={seed} class={c}"


def test_gpu_path_values_close():
    X = _class_term(5, 200, 7)
    bt = _bt_dense(BTClassTfidf(), X)
    mlx = MlxCTFIDF().fit_transform(X).toarray()
    np.testing.assert_allclose(mlx, bt, rtol=1e-3, atol=1e-4)


def test_gpu_bm25_matches():
    X = _class_term(5, 200, 3)
    bt = _bt_dense(BTClassTfidf(bm25_weighting=True), X)
    mlx = MlxCTFIDF(bm25_weighting=True).fit_transform(X).toarray()
    for c in range(X.shape[0]):
        assert _topk(bt[c]) == _topk(mlx[c])


def test_gpu_reduce_frequent_matches():
    X = _class_term(5, 200, 5)
    bt = _bt_dense(BTClassTfidf(reduce_frequent_words=True), X)
    mlx = MlxCTFIDF(reduce_frequent_words=True).fit_transform(X).toarray()
    np.testing.assert_allclose(mlx, bt, rtol=1e-3, atol=1e-4)


def test_numpy_fallback_matches_bertopic():
    """Force the CPU fallback path (small _MAX_ELEMENTS_FOR_GPU) and verify."""
    X = _class_term(5, 200, 11)
    bt = _bt_dense(BTClassTfidf(), X)
    orig = ctfidf_mod._MAX_ELEMENTS_FOR_GPU
    ctfidf_mod._MAX_ELEMENTS_FOR_GPU = 4  # force fallback
    try:
        mlx = MlxCTFIDF().fit_transform(X).toarray()
    finally:
        ctfidf_mod._MAX_ELEMENTS_FOR_GPU = orig
    np.testing.assert_allclose(mlx, bt, rtol=1e-3, atol=1e-4)
    for c in range(X.shape[0]):
        assert _topk(bt[c]) == _topk(mlx[c])


def test_numpy_fallback_reduce_matches():
    X = _class_term(5, 200, 13)
    bt = _bt_dense(BTClassTfidf(reduce_frequent_words=True), X)
    orig = ctfidf_mod._MAX_ELEMENTS_FOR_GPU
    ctfidf_mod._MAX_ELEMENTS_FOR_GPU = 4
    try:
        mlx = MlxCTFIDF(reduce_frequent_words=True).fit_transform(X).toarray()
    finally:
        ctfidf_mod._MAX_ELEMENTS_FOR_GPU = orig
    np.testing.assert_allclose(mlx, bt, rtol=1e-3, atol=1e-4)


def test_sparse_and_dense_both_match_bertopic():
    X = _class_term(5, 150, 21)
    bt = _bt_dense(BTClassTfidf(), X)
    dense_out = MlxCTFIDF().fit_transform(X).toarray()
    sparse_out = MlxCTFIDF().fit_transform(sp.csr_matrix(X)).toarray()
    np.testing.assert_allclose(dense_out, sparse_out, atol=1e-5)
    for c in range(X.shape[0]):
        assert _topk(bt[c]) == _topk(dense_out[c])


def test_multiplier_boosts_seed_words_fit_then_transform():
    """BERTopic passes multiplier to fit() (baked into idf), not transform()."""
    X = _class_term(5, 200, 7)
    base = MlxCTFIDF().fit_transform(X).toarray()

    mult = np.ones(X.shape[1], dtype=np.float32)
    mult[0] = 100.0  # boost word 0

    m = MlxCTFIDF()
    m.fit(X, multiplier=mult)
    boosted = m.transform(X).toarray()  # transform WITHOUT multiplier arg

    ratio0 = boosted[:, 0].mean() / base[:, 0].mean()
    ratio_other = boosted[:, 1:].mean() / base[:, 1:].mean()
    assert ratio0 > 50.0, f"word 0 not boosted: {ratio0}"
    assert 0.9 < ratio_other < 1.1, f"other words changed: {ratio_other}"


def test_multiplier_via_numpy_fallback():
    X = _class_term(5, 200, 9)
    mult = np.ones(X.shape[1], dtype=np.float32)
    mult[3] = 50.0
    orig = ctfidf_mod._MAX_ELEMENTS_FOR_GPU
    ctfidf_mod._MAX_ELEMENTS_FOR_GPU = 4  # force fallback path
    try:
        base = MlxCTFIDF().fit_transform(X).toarray()
        boosted = MlxCTFIDF().fit_transform(X, multiplier=mult).toarray()
    finally:
        ctfidf_mod._MAX_ELEMENTS_FOR_GPU = orig
    ratio3 = boosted[:, 3].mean() / base[:, 3].mean()
    assert ratio3 > 25.0, f"word 3 not boosted on fallback path: {ratio3}"
