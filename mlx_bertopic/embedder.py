"""MlxEmbedder — BERTopic embedding backend for Apple Silicon (MLX Metal GPU).

Wraps mlx-embeddings to match BERTopic's BaseEmbedder interface.
"""
import numpy as np
from bertopic.backend import BaseEmbedder


class MlxEmbedder(BaseEmbedder):
    """Apple Silicon MLX embedding backend for BERTopic."""

    def __init__(self, model_name: str = "mlx-community/bge-small-en-v1.5-bf16"):
        super().__init__()
        from mlx_embeddings import load
        self._model_name = model_name
        self.model, self.tokenizer = load(model_name)
        self._embedding_dim = None

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
        from mlx_embeddings import generate

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
