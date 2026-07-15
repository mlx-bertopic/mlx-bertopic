"""Outlier reduction utilities — re-exports from mlx-outlier.

Usage:
    from mlx_bertopic import reduce_outliers_embeddings, reduce_outliers_ctfidf

    # After fitting BERTopic:
    new_topics = reduce_outliers_embeddings(topics, embeddings, threshold=0.5)
    topic_model.update_topics(docs, topics=new_topics)
"""

from mlx_outlier import reduce_outliers_embeddings, reduce_outliers_ctfidf

__all__ = ["reduce_outliers_embeddings", "reduce_outliers_ctfidf"]
