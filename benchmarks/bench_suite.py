#!/usr/bin/env python3.11
"""Benchmark: MLX vs CPU for BERTopic sub-components.

Measures wall-clock time for c-TF-IDF, outlier reduction, and KeyBERT/MMR
at various scales. Reports speedup factors.
"""
import time
import numpy as np
import scipy.sparse as sp


def bench_ctfidf():
    """c-TF-IDF: MlxCTFIDF (GPU) vs ClassTfidfTransformer (CPU sparse)."""
    from bertopic.vectorizers import ClassTfidfTransformer
    from mlx_ctfidf import CTFIDFTransformer

    print("═══ c-TF-IDF ═══")
    print(f"{'Scale':<30} {'BERTopic':>10} {'MLX':>10} {'Speedup':>10}")
    print("─" * 65)

    for n_topics, vocab in [(20, 5000), (50, 20000), (100, 50000), (200, 100000), (500, 200000)]:
        rng = np.random.default_rng(42)
        X_dense = rng.negative_binomial(1, 0.2, size=(n_topics, vocab)).astype(np.float64)
        X_dense[rng.random((n_topics, vocab)) > 0.3] = 0
        X_sp = sp.csr_matrix(X_dense)

        # BERTopic (sparse)
        times_ref = []
        for _ in range(3):
            X_copy = X_sp.copy()
            t0 = time.perf_counter()
            ref = ClassTfidfTransformer()
            ref.fit_transform(X_copy)
            times_ref.append(time.perf_counter() - t0)
        t_ref = min(times_ref)

        # MLX (dense GPU)
        times_mlx = []
        for _ in range(3):
            t0 = time.perf_counter()
            mlx_t = CTFIDFTransformer()
            mlx_t.fit_transform(X_dense.astype(np.float32))
            times_mlx.append(time.perf_counter() - t0)
        t_mlx = min(times_mlx)

        sp_str = f"{t_ref/t_mlx:.1f}x" if t_mlx > 0.001 else "N/A"
        print(f"  {n_topics:>4} topics × {vocab:>6} vocab  {t_ref*1000:>7.1f}ms {t_mlx*1000:>7.1f}ms {sp_str:>10}")

    print()


def bench_outlier():
    """Outlier reduction: MLX GPU vs naive numpy."""
    from mlx_outlier import reduce_outliers_embeddings

    print("═══ Outlier Reduction (embedding-based) ═══")
    print(f"{'Scale':<30} {'numpy':>10} {'MLX':>10} {'Speedup':>10}")
    print("─" * 65)

    for n_docs, n_topics, dim in [(5000, 10, 384), (20000, 30, 384), (50000, 50, 768), (100000, 100, 768)]:
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((n_docs, dim)).astype(np.float32)
        # 30% outliers
        topics = rng.integers(0, n_topics, size=n_docs)
        outlier_mask = rng.random(n_docs) < 0.3
        topics[outlier_mask] = -1

        # Numpy baseline (same logic but no MLX)
        def numpy_outlier(topics, embeddings):
            topics = np.array(topics, dtype=np.int32)
            new_topics = topics.copy()
            unique_topics = np.unique(topics[topics >= 0])
            centroids = []
            for t in unique_topics:
                idx = np.where(topics == t)[0]
                centroids.append(embeddings[idx].mean(axis=0))
            centroids = np.array(centroids)
            centroids = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
            outlier_idx = np.where(topics == -1)[0]
            out_emb = embeddings[outlier_idx]
            out_emb = out_emb / np.maximum(np.linalg.norm(out_emb, axis=1, keepdims=True), 1e-9)
            sims = out_emb @ centroids.T
            best = sims.argmax(axis=1)
            for j, idx in enumerate(outlier_idx):
                new_topics[idx] = unique_topics[best[j]]
            return new_topics

        # Numpy
        t0 = time.perf_counter()
        numpy_outlier(topics, embeddings)
        t_np = time.perf_counter() - t0

        # MLX
        t0 = time.perf_counter()
        reduce_outliers_embeddings(topics, embeddings, threshold=0.0)
        t_mlx = time.perf_counter() - t0

        sp_str = f"{t_np/t_mlx:.1f}x" if t_mlx > 0.001 else "N/A"
        n_outliers = outlier_mask.sum()
        print(f"  {n_docs:>6} docs, {n_outliers:>5} outliers  {t_np*1000:>7.1f}ms {t_mlx*1000:>7.1f}ms {sp_str:>10}")

    print()


def bench_keybert():
    """KeyBERT: MLX GPU vs numpy cosine similarity."""
    from mlx_keybert import KeyBERTInspired

    print("═══ KeyBERT (top-N by cosine sim) ═══")
    print(f"{'Scale':<30} {'numpy':>10} {'MLX':>10} {'Speedup':>10}")
    print("─" * 65)

    for n_docs, n_words, dim, n_topics in [(5000, 10000, 384, 10), (20000, 30000, 384, 30), (50000, 50000, 768, 50)]:
        rng = np.random.default_rng(42)
        doc_emb = rng.standard_normal((n_docs, dim)).astype(np.float32)
        word_emb = rng.standard_normal((n_words, dim)).astype(np.float32)
        topics = np.array([i % n_topics for i in range(n_docs)])
        vocab = [f"w{i}" for i in range(n_words)]
        topic_ids = list(range(n_topics))

        # Numpy baseline
        def numpy_keybert(doc_emb, word_emb, topics, vocab, topic_ids, top_n=10):
            word_norms = np.maximum(np.linalg.norm(word_emb, axis=1, keepdims=True), 1e-9)
            word_emb_n = word_emb / word_norms
            results = {}
            for tid in topic_ids:
                idx = np.where(topics == tid)[0]
                centroid = doc_emb[idx].mean(axis=0, keepdims=True)
                c_norm = np.maximum(np.linalg.norm(centroid), 1e-9)
                centroid = centroid / c_norm
                sims = (centroid @ word_emb_n.T).squeeze()
                top_idx = np.argsort(sims)[::-1][:top_n]
                results[tid] = [(vocab[i], float(sims[i])) for i in top_idx]
            return results

        # Numpy
        t0 = time.perf_counter()
        numpy_keybert(doc_emb, word_emb, topics, vocab, topic_ids)
        t_np = time.perf_counter() - t0

        # MLX
        kb = KeyBERTInspired(top_n_words=10)
        t0 = time.perf_counter()
        kb.extract_topics(topic_ids, doc_emb, topics, word_emb, vocab)
        t_mlx = time.perf_counter() - t0

        sp_str = f"{t_np/t_mlx:.1f}x" if t_mlx > 0.001 else "N/A"
        print(f"  {n_docs:>5}d × {n_words:>5}w × {dim}D   {t_np*1000:>7.1f}ms {t_mlx*1000:>7.1f}ms {sp_str:>10}")

    print()


if __name__ == "__main__":
    print("MLX BERTopic Component Benchmarks")
    print("=" * 65)
    print()
    bench_ctfidf()
    bench_outlier()
    bench_keybert()
