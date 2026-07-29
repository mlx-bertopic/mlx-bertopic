"""MlxUMAPWrapper — BERTopic UMAP backend for Apple Silicon (MLX Metal GPU).

Wraps mlx-vis UMAP to match the interface BERTopic expects.

NOTE: mlx-vis UMAP has a known embedding divergence bug on datasets >50K
with high-dimensional (>1000d) inputs. Hub nodes accumulate excessive
gradient from batch scatter-add, causing coordinates to fly to ±hundreds.

Workaround applied here:
  1. Post-hoc L2 norm clipping (max_norm=15) on the output
  2. Hard coordinate clipping to ±20

For the proper fix, patch mlx_vis/_umap/umap.py:
  - _sgd_step: change `clip(grad, -4, 4) * alpha` → `clip(grad * alpha, -4, 4)`
  - _optimize: add per-epoch `Y = mx.clip(Y, -20.0, 20.0)` after _sgd_step

See: patches/mlx_vis_umap_stability.md
"""

import numpy as np


class MlxUMAPWrapper:
    """Wraps mlx-vis UMAP to accept the `y=` kwarg that BERTopic passes.

    BERTopic calls `umap_model.fit_transform(embeddings, y=y)` for
    semi-supervised topic modeling. mlx-vis UMAP doesn't support `y`,
    so we silently ignore it (unsupervised mode only).

    Includes a post-hoc stability fix for large datasets (>50K points).
    """

    # Embedding bound for divergence protection
    MAX_NORM = 15.0
    HARD_CLIP = 20.0

    def __init__(self, **kwargs):
        from mlx_vis import UMAP as MlxUMAP
        self._umap = MlxUMAP(**kwargs)
        self._fitted_embedding: np.ndarray | None = None

    def _stabilize(self, embedding: np.ndarray) -> np.ndarray:
        """Clip divergent points back to safe range.

        Applies two-layer protection:
          1. Per-point L2 norm clipping (smooth, preserves direction)
          2. Hard coordinate clipping (final safety net)

        This is a post-hoc workaround. The proper fix is in the SGD loop
        (see module docstring). Points within normal range are unaffected.
        """
        # L2 norm clip
        norms = np.sqrt((embedding ** 2).sum(axis=1, keepdims=True) + 1e-8)
        scale = np.minimum(self.MAX_NORM / norms, 1.0)
        embedding = embedding * scale
        # Hard clip
        embedding = np.clip(embedding, -self.HARD_CLIP, self.HARD_CLIP)
        return embedding

    def fit(self, X, y=None):
        # y is ignored — mlx-vis UMAP is unsupervised only
        self._fitted_embedding = self._umap.fit_transform(X)
        self._fitted_embedding = self._stabilize(self._fitted_embedding)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self._fitted_embedding

    def transform(self, X):
        """Return the fitted embedding if the same data is passed.

        mlx-vis UMAP does not implement an incremental ``transform`` for new
        data. BERTopic's ``fit_transform`` calls ``fit`` followed by
        ``transform`` on the same training set, so we return the cached
        ``_fitted_embedding``. For genuinely new data, use ``fit_transform`` on
        the combined dataset.
        """
        if self._fitted_embedding is None:
            raise NotImplementedError(
                "MlxUMAPWrapper has not been fitted yet; call fit_transform first."
            )
        if X.shape[0] != self._fitted_embedding.shape[0]:
            raise NotImplementedError(
                "mlx-vis UMAP does not support incremental transform on new data; "
                "use fit_transform on the combined dataset instead."
            )
        return self._fitted_embedding
