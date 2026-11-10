"""MlxEmbedder — BERTopic embedding backend for Apple Silicon (MLX Metal GPU).

Wraps mlx-embeddings to match BERTopic's BaseEmbedder interface.

mlx-embeddings is an optional dependency (GPL-3.0): install it with
``pip install "mlx-bertopic[embed]"``. Any object exposing
``embed(documents) -> np.ndarray`` can be passed to BERTopic instead.
"""
import numpy as np
from bertopic.backend import BaseEmbedder

_INSTALL_HINT = (
    "MlxEmbedder needs the optional dependency mlx-embeddings (GPL-3.0), which "
    'is not installed by default: pip install "mlx-bertopic[embed]" — or pass '
    "any object with embed(documents) -> np.ndarray as embedding_model."
)


def _mlx_embeddings():
    """Import mlx_embeddings, or explain how to install it."""
    try:
        import mlx_embeddings
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return mlx_embeddings


class MlxEmbedder(BaseEmbedder):
    """Apple Silicon MLX embedding backend for BERTopic."""

    def __init__(self, model_name: str = "mlx-community/bge-small-en-v1.5-bf16"):
        super().__init__()
        self._model_name = model_name
        self.model, self.tokenizer = _mlx_embeddings().load(model_name)
        self._embedding_dim = None

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
        generate = _mlx_embeddings().generate

        batch_size = 64
        all_embeddings = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            result = generate(
                self.model, self.tokenizer, batch,
                max_length=512, padding=True, truncation=True,
            )

            if hasattr(result, 'text_embeds') and result.text_embeds is not None:
                emb = np.array(result.text_embeds)
            elif hasattr(result, 'pooler_output') and result.pooler_output is not None:
                emb = np.array(result.pooler_output)
            else:
                hidden = np.array(result.last_hidden_state)
                emb = hidden.mean(axis=1)

            all_embeddings.append(emb.astype(np.float32))

        embeddings = np.vstack(all_embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-9, None)

        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]

        return embeddings
