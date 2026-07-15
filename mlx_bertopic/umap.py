"""MlxUMAPWrapper — BERTopic UMAP backend for Apple Silicon (MLX Metal GPU).

Wraps mlx-vis UMAP to match the interface BERTopic expects.
"""


class MlxUMAPWrapper:
    """Wraps mlx-vis UMAP to accept the `y=` kwarg that BERTopic passes.

    BERTopic calls `umap_model.fit_transform(embeddings, y=y)` for
    semi-supervised topic modeling. mlx-vis UMAP doesn't support `y`,
    so we silently ignore it (unsupervised mode only).
    """

    def __init__(self, **kwargs):
        from mlx_vis import UMAP as MlxUMAP
        self._umap = MlxUMAP(**kwargs)

    def fit(self, X, y=None):
        self._umap.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        # y is ignored — mlx-vis UMAP is unsupervised only
        return self._umap.fit_transform(X)

    def transform(self, X):
        """Transform is not supported.

        mlx-vis UMAP does not implement an incremental ``transform`` for new
        data. Previously this silently called ``fit_transform`` (i.e. re-fit on
        the new data), which produces coordinates inconsistent with the
        original embedding. Raise explicitly instead of returning wrong results.

        To embed new points together with existing ones, call ``fit_transform``
        on the combined dataset.
        """
        raise NotImplementedError(
            "mlx-vis UMAP does not support incremental transform on new data; "
            "use fit_transform on the combined dataset instead."
        )
