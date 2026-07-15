"""mlx-keybert — KeyBERT-inspired + MMR topic representation via MLX (Metal GPU).

BERTopic representation_model backends accelerated on Apple Silicon.
"""

import mlx.core as mx
import numpy as np

__version__ = "0.1.0"


class KeyBERTInspired:
    """KeyBERT-inspired representation: select keywords most similar to topic embedding.

    For each topic, compute cosine similarity between candidate word embeddings
    and the topic centroid, then pick the top-N most similar.

    NOTE: this is a standalone API (``extract_topics(topic_ids, doc_embeddings,
    topic_assignments, word_embeddings, vocab)``) and is NOT directly callable
    as a BERTopic ``representation_model``. Use the ``mlx_bertopic.MlxKeyBERT``
    wrapper, which adapts this backend to BERTopic's representation interface.
    """

    def __init__(self, top_n_words: int = 10):
        self.top_n_words = top_n_words

    def extract_topics(
        self,
        topic_ids: list[int],
        doc_embeddings: np.ndarray,
        topic_assignments: np.ndarray,
        word_embeddings: np.ndarray,
        vocab: list[str],
    ) -> dict[int, list[tuple[str, float]]]:
        """Extract top keywords per topic using embedding similarity.

        Args:
            topic_ids: list of topic IDs to process.
            doc_embeddings: (N, D) document embeddings.
            topic_assignments: (N,) topic per document.
            word_embeddings: (V, D) word/token embeddings.
            vocab: list of V words matching word_embeddings rows.

        Returns:
            {topic_id: [(word, score), ...]} for each topic.
        """
        doc_emb_f32 = doc_embeddings.astype(np.float32)
        word_emb_mx = mx.array(word_embeddings.astype(np.float32))

        # L2-normalize words
        w_norms = mx.maximum(mx.sqrt(mx.sum(word_emb_mx * word_emb_mx, axis=1, keepdims=True)), 1e-9)
        word_emb_mx = word_emb_mx / w_norms
        mx.eval(word_emb_mx)

        results = {}
        for tid in topic_ids:
            idx = np.where(topic_assignments == tid)[0]
            if len(idx) == 0:
                results[tid] = []
                continue

            # Topic centroid (index with numpy, compute with MLX)
            centroid = mx.mean(mx.array(doc_emb_f32[idx]), axis=0, keepdims=True)
            c_norm = mx.maximum(mx.sqrt(mx.sum(centroid * centroid, axis=1, keepdims=True)), 1e-9)
            centroid = centroid / c_norm

            # Cosine similarity to all words
            sims = (centroid @ word_emb_mx.T).squeeze(0)
            mx.eval(sims)

            sims_np = np.array(sims)
            top_idx = np.argsort(sims_np)[::-1][:self.top_n_words]
            results[tid] = [(vocab[i], float(sims_np[i])) for i in top_idx]

        return results


class MaximalMarginalRelevance:
    """MMR-based keyword selection: balance relevance and diversity.

    Iteratively selects words that are similar to the topic centroid
    but dissimilar to already-selected words.

    NOTE: standalone API (see ``KeyBERTInspired``); use the
    ``mlx_bertopic.MlxMMR`` wrapper to plug into BERTopic's
    ``representation_model``.
    """

    def __init__(self, top_n_words: int = 10, diversity: float = 0.3):
        self.top_n_words = top_n_words
        self.diversity = diversity

    def extract_topics(
        self,
        topic_ids: list[int],
        doc_embeddings: np.ndarray,
        topic_assignments: np.ndarray,
        word_embeddings: np.ndarray,
        vocab: list[str],
        candidates_per_topic: int = 50,
    ) -> dict[int, list[tuple[str, float]]]:
        """Extract diverse keywords per topic using MMR.

        Args:
            topic_ids: list of topic IDs.
            doc_embeddings: (N, D) document embeddings.
            topic_assignments: (N,) topic per document.
            word_embeddings: (V, D) word embeddings.
            vocab: vocabulary list.
            candidates_per_topic: pre-filter top candidates before MMR.

        Returns:
            {topic_id: [(word, score), ...]}
        """
        doc_emb_f32 = doc_embeddings.astype(np.float32)
        word_emb_mx = mx.array(word_embeddings.astype(np.float32))

        # L2-normalize
        w_norms = mx.maximum(mx.sqrt(mx.sum(word_emb_mx * word_emb_mx, axis=1, keepdims=True)), 1e-9)
        word_emb_mx = word_emb_mx / w_norms
        mx.eval(word_emb_mx)

        # Convert normalized word embeddings to numpy once for MMR
        word_emb_np = np.array(word_emb_mx)

        results = {}
        for tid in topic_ids:
            idx = np.where(topic_assignments == tid)[0]
            if len(idx) == 0:
                results[tid] = []
                continue

            # Topic centroid
            centroid = mx.mean(mx.array(doc_emb_f32[idx]), axis=0, keepdims=True)
            c_norm = mx.maximum(mx.sqrt(mx.sum(centroid * centroid, axis=1, keepdims=True)), 1e-9)
            centroid = centroid / c_norm

            # Pre-filter top candidates
            sims_to_topic = (centroid @ word_emb_mx.T).squeeze(0)
            mx.eval(sims_to_topic)
            sims_np = np.array(sims_to_topic)
            candidate_idx = np.argsort(sims_np)[::-1][:candidates_per_topic]

            # MMR selection
            selected = self._mmr(
                word_emb_np, candidate_idx, sims_np, self.top_n_words, self.diversity
            )
            results[tid] = [(vocab[i], float(sims_np[i])) for i in selected]

        return results

    def _mmr(
        self,
        word_emb_np: np.ndarray,
        candidate_idx: np.ndarray,
        topic_sims: np.ndarray,
        top_n: int,
        diversity: float,
    ) -> list[int]:
        """Maximal Marginal Relevance greedy selection.

        Pre-computes the full pairwise similarity matrix on GPU in one
        matmul call, then does greedy selection entirely in numpy.
        """
        if len(candidate_idx) == 0:
            return []

        # 1. Extract candidate embeddings once (numpy indexing)
        cand_emb = word_emb_np[candidate_idx]  # (C, D)

        # 2. Compute full pairwise similarity matrix in one GPU matmul
        cand_emb_mx = mx.array(cand_emb)
        pairwise_sim_mx = cand_emb_mx @ cand_emb_mx.T  # (C, C)
        mx.eval(pairwise_sim_mx)
        pairwise_sim = np.array(pairwise_sim_mx)  # back to numpy once

        # 3. Candidate-to-topic similarities (already in topic_sims)
        cand_topic_sims = topic_sims[candidate_idx]  # (C,)

        # 4. Greedy MMR selection entirely in numpy
        n_candidates = len(candidate_idx)
        selected_mask = np.zeros(n_candidates, dtype=bool)
        selected_local = [0]  # first candidate (highest topic sim)
        selected_mask[0] = True

        for _ in range(top_n - 1):
            if np.sum(~selected_mask) == 0:
                break

            # Max similarity of each remaining candidate to any selected candidate
            remaining_mask = ~selected_mask
            # pairwise_sim[remaining, :][:, selected] → max over selected axis
            sim_to_selected = pairwise_sim[remaining_mask][:, selected_mask].max(axis=1)

            # MMR score
            relevance = cand_topic_sims[remaining_mask]
            mmr_scores = (1 - diversity) * relevance - diversity * sim_to_selected

            # Map back to local candidate index
            remaining_indices = np.where(remaining_mask)[0]
            best_local = remaining_indices[np.argmax(mmr_scores)]

            selected_local.append(best_local)
            selected_mask[best_local] = True

        # Map local indices back to global vocab indices
        return [int(candidate_idx[i]) for i in selected_local]


__all__ = ["KeyBERTInspired", "MaximalMarginalRelevance"]
