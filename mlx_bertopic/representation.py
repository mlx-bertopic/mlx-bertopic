"""MlxKeyBERT / MlxMMR — BERTopic representation_model backends (MLX Metal GPU).

Wraps mlx-keybert to match BERTopic's BaseRepresentation interface.
"""
import warnings

import numpy as np
from mlx_keybert import KeyBERTInspired, MaximalMarginalRelevance


def _get_embeddings(topic_model):
    """Try multiple attribute names to retrieve document embeddings.

    BERTopic has used different attribute names across versions.
    Returns the embeddings array or None if unavailable.
    """
    # Try the common attribute names
    for attr in ('_embeddings', 'embeddings_'):
        emb = getattr(topic_model, attr, None)
        if emb is not None:
            return emb

    # Check if embedding_model has cached embeddings
    embedding_model = getattr(topic_model, 'embedding_model', None)
    if embedding_model is not None:
        cached = getattr(embedding_model, 'embeddings_', None)
        if cached is not None:
            return cached

    return None


def _extract_topics_impl(backend, topic_model, documents, c_tf_idf, topics):
    """Shared implementation for MlxKeyBERT and MlxMMR extract_topics.

    Args:
        backend: KeyBERTInspired or MaximalMarginalRelevance instance.
        topic_model: the BERTopic model instance.
        documents: DataFrame with 'Document', 'Topic', 'ID' columns.
        c_tf_idf: scipy sparse c-TF-IDF matrix.
        topics: dict {topic_id: [(word, score), ...]}.

    Returns:
        Updated topics dict with MLX-derived keywords, or original topics on failure.
    """
    embeddings = _get_embeddings(topic_model)
    if embeddings is None:
        warnings.warn(
            'MlxKeyBERT: no embeddings available, skipping GPU representation',
            UserWarning,
            stacklevel=3,
        )
        return topics

    if documents is not None and hasattr(documents, '__getitem__'):
        try:
            topic_assignments = np.array(documents['Topic'].values)
        except (KeyError, AttributeError, TypeError):
            topic_assignments = np.array(topic_model.topics_)
    else:
        topic_assignments = np.array(topic_model.topics_)

    if not hasattr(topic_model, 'vectorizer_model') or topic_model.vectorizer_model is None:
        warnings.warn(
            'MlxKeyBERT: no vectorizer_model, skipping GPU representation',
            UserWarning,
            stacklevel=3,
        )
        return topics

    vocab = topic_model.vectorizer_model.get_feature_names_out().tolist()

    embedding_model = getattr(topic_model, 'embedding_model', None)
    if embedding_model is None or not hasattr(embedding_model, 'embed'):
        warnings.warn(
            'MlxKeyBERT: embedding_model has no embed() method, skipping',
            UserWarning,
            stacklevel=3,
        )
        return topics

    word_embeddings = embedding_model.embed(vocab)

    topic_ids = [t for t in set(topic_assignments) if t >= 0]

    result = backend.extract_topics(
        topic_ids=topic_ids,
        doc_embeddings=embeddings,
        topic_assignments=topic_assignments,
        word_embeddings=word_embeddings,
        vocab=vocab,
    )

    updated_topics = {}
    for tid, words_scores in result.items():
        updated_topics[tid] = words_scores

    return updated_topics


class MlxKeyBERT:
    """BERTopic representation_model using KeyBERT-inspired MLX similarity.

    Usage:
        topic_model = BERTopic(representation_model=MlxKeyBERT(top_n_words=10))
    """

    def __init__(self, top_n_words: int = 10):
        self._backend = KeyBERTInspired(top_n_words=top_n_words)
        self.top_n_words = top_n_words

    def extract_topics(self, topic_model, documents, c_tf_idf, topics):
        """BERTopic representation_model interface.

        Args:
            topic_model: the BERTopic model instance.
            documents: DataFrame with 'Document', 'Topic', 'ID' columns.
            c_tf_idf: scipy sparse c-TF-IDF matrix.
            topics: dict {topic_id: [(word, score), ...]}.

        Returns:
            Updated topics dict with MLX-derived keywords.
        """
        return _extract_topics_impl(
            self._backend, topic_model, documents, c_tf_idf, topics
        )


class MlxMMR:
    """BERTopic representation_model using MMR diversity selection (MLX Metal GPU).

    Usage:
        topic_model = BERTopic(representation_model=MlxMMR(diversity=0.3))
    """

    def __init__(self, top_n_words: int = 10, diversity: float = 0.3):
        self._backend = MaximalMarginalRelevance(
            top_n_words=top_n_words, diversity=diversity
        )
        self.top_n_words = top_n_words
        self.diversity = diversity

    def extract_topics(self, topic_model, documents, c_tf_idf, topics):
        """BERTopic representation_model interface."""
        return _extract_topics_impl(
            self._backend, topic_model, documents, c_tf_idf, topics
        )
