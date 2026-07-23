"""mlx-bertopic — BERTopic backends for Apple Silicon (MLX Metal GPU).

Provides drop-in replacements for BERTopic's components, accelerated on Metal:

  - MlxEmbedder: embedding_model (requires mlx-embeddings)
  - MlxUMAPWrapper: umap_model (requires mlx-vis)
  - MlxCTFIDF: ctfidf_model (requires mlx-ctfidf)
  - MlxKeyBERT / MlxMMR: representation_model (requires mlx-keybert)
  - reduce_outliers_embeddings / reduce_outliers_ctfidf (requires mlx-outlier)

For HDBSCAN, use the dedicated package directly:
    from mlx_hdbscan import HDBSCAN  # (mlx-hdbscan is a separate package)

Backends are imported lazily, so importing this package (or just one backend,
e.g. ``from mlx_bertopic import MlxCTFIDF``) does not require every optional
dependency to be installed — only the backend you actually use.

Usage:
    from mlx_bertopic import (
        MlxEmbedder, MlxUMAPWrapper,
        MlxCTFIDF, MlxKeyBERT, MlxMMR,
        reduce_outliers_embeddings, reduce_outliers_ctfidf,
    )
    from bertopic import BERTopic

    topic_model = BERTopic(
        embedding_model=MlxEmbedder("mlx-community/bge-small-en-v1.5-bf16"),
        umap_model=MlxUMAPWrapper(n_components=5, n_neighbors=15),
        ctfidf_model=MlxCTFIDF(),
        representation_model=MlxKeyBERT(top_n_words=10),
    )
    topics, probs = topic_model.fit_transform(documents)
"""

__version__ = "0.6.0"

# Map public name -> module that defines it. Imported lazily so that importing
# one backend does not require every optional dependency (mlx-embeddings,
# mlx-vis, etc.) to be installed.
_LAZY = {
    "MlxEmbedder": "mlx_bertopic.embedder",
    "MlxUMAPWrapper": "mlx_bertopic.umap",
    "MlxCTFIDF": "mlx_bertopic.ctfidf",
    "MlxKeyBERT": "mlx_bertopic.representation",
    "MlxMMR": "mlx_bertopic.representation",
    "reduce_outliers_embeddings": "mlx_bertopic.outlier",
    "reduce_outliers_ctfidf": "mlx_bertopic.outlier",
}

__all__ = list(_LAZY) + ["__version__"]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        try:
            return getattr(module, name)
        except AttributeError:  # pragma: no cover - defensive
            raise ImportError(f"{name} cannot be imported from {_LAZY[name]}") from None
    raise AttributeError(f"module 'mlx_bertopic' has no attribute {name!r}")
