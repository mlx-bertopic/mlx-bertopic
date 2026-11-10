"""MlxUMAPWrapper.transform() contract: cached embedding for fitted data, raise for new."""

import numpy as np
import pytest

from mlx_bertopic.umap import MlxUMAPWrapper


@pytest.fixture(scope="module")
def fitted():
    X = np.random.default_rng(0).standard_normal((20, 8)).astype(np.float32)
    wrapper = MlxUMAPWrapper(n_components=2, n_neighbors=5)
    embedding = wrapper.fit_transform(X)
    return wrapper, X, embedding


def test_fit_transform_shape(fitted):
    _, X, embedding = fitted
    assert embedding.shape == (X.shape[0], 2)


def test_transform_returns_cached_embedding_for_fitted_data(fitted):
    """BERTopic >=0.17 does fit(X) then transform(X) on the same array."""
    wrapper, X, embedding = fitted
    assert np.array_equal(wrapper.transform(X), embedding)
    # An equal copy must be recognised too, not only the identical object.
    assert np.array_equal(wrapper.transform(X.copy()), embedding)


def test_transform_raises_for_different_data_of_same_length(fitted):
    """Same row count, different rows: returning the cache would be wrong."""
    wrapper, X, _ = fitted
    other = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="incremental transform"):
        wrapper.transform(other)


def test_transform_raises_for_new_data(fitted):
    wrapper, X, _ = fitted
    new_data = np.zeros((X.shape[0] + 5, X.shape[1]), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="incremental transform"):
        wrapper.transform(new_data)


def test_transform_before_fit_raises():
    wrapper = MlxUMAPWrapper(n_components=2)
    with pytest.raises(NotImplementedError, match="not been fitted"):
        wrapper.transform(np.zeros((4, 4), dtype=np.float32))
