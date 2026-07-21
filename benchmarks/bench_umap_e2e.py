#!/usr/bin/env python3
"""End-to-end BERTopic pipeline benchmark: MLX (Metal GPU) vs CPU.

Dataset: 214,515 pre-computed embeddings (4096-d) from PTT forum data.
Hardware: Apple M4 Max, 128 GB unified memory.

Stages timed independently:
  1. load        — numpy .npy + parquet metadata
  2. umap_5d     — dimensionality reduction for clustering
  3. hdbscan     — density-based clustering
  4. bertopic_fit — c-TF-IDF topic extraction
  5. umap_2d     — dimensionality reduction for visualization

Usage:
    python bench_umap_e2e.py --data-dir /path/to/data [--backend cpu|mlx]

Requires:
    - ptt_khs_embeddings.npy  (214515, 4096) float32
    - ptt_khs_meta.parquet    (214515 rows, 'sentence' column)
"""

import argparse
import json
import os
import platform
import resource
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def get_peak_rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def timer(func, label):
    print(f"  [{label}] starting...", flush=True)
    t0 = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] done: {elapsed:.2f}s", flush=True)
    return result, elapsed


def run_benchmark(data_dir: str, backend: str = "cpu", multicore: bool = False):
    hostname = socket.gethostname()
    mode_label = f"{backend}" + ("_multicore" if multicore else "_singlecore")

    print(f"=== BERTopic E2E Benchmark ===")
    print(f"Host: {hostname} | Backend: {backend} | Multicore: {multicore}")
    print(f"Python: {sys.version.split()[0]} | Cores: {os.cpu_count()}")
    print()

    stages = {}

    # ─── Load ────────────────────────────────────────────────────────
    def load_data():
        emb = np.load(os.path.join(data_dir, "ptt_khs_embeddings.npy"), mmap_mode="r")
        meta = pd.read_parquet(os.path.join(data_dir, "ptt_khs_meta.parquet"))
        docs = meta["sentence"].fillna("").tolist()
        emb_array = np.array(emb, dtype=np.float32)
        return emb_array, docs

    (embeddings, docs), t_load = timer(load_data, "load")
    stages["load"] = round(t_load, 3)
    print(f"  shape: {embeddings.shape}, docs: {len(docs):,}\n")

    # ─── UMAP 5D ────────────────────────────────────────────────────
    if backend == "mlx":
        def umap_5d():
            from mlx_bertopic.umap import MlxUMAPWrapper
            model = MlxUMAPWrapper(
                n_neighbors=15, n_components=5, min_dist=0.0,
                random_state=42, normalize=True,
            )
            return model.fit_transform(embeddings)
    else:
        def umap_5d():
            from umap import UMAP
            kw = dict(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", low_memory=True)
            if multicore:
                kw["n_jobs"] = -1
            else:
                kw["random_state"] = 42
            return UMAP(**kw).fit_transform(embeddings)

    umap_5d_result, t_umap5d = timer(umap_5d, "umap_5d")
    stages["umap_5d"] = round(t_umap5d, 3)

    # ─── HDBSCAN ────────────────────────────────────────────────────
    def run_hdbscan():
        from hdbscan import HDBSCAN
        model = HDBSCAN(min_cluster_size=150, min_samples=10,
                        metric="euclidean", cluster_selection_method="eom",
                        prediction_data=True)
        return model.fit_predict(umap_5d_result)

    labels, t_hdbscan = timer(run_hdbscan, "hdbscan")
    stages["hdbscan"] = round(t_hdbscan, 3)
    n_topics = len(set(labels)) - (1 if -1 in labels else 0)
    outlier_pct = round((labels == -1).sum() / len(labels) * 100, 2)
    print(f"  topics: {n_topics}, outliers: {outlier_pct}%\n")

    # ─── BERTopic fit ───────────────────────────────────────────────
    def run_bertopic_fit():
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer

        class PrecomputedUMAP:
            def fit(self, X, **kw): return self
            def transform(self, X, **kw): return umap_5d_result
            def fit_transform(self, X, **kw): return umap_5d_result

        class PrecomputedHDBSCAN:
            def __init__(self, lbl): self.labels_ = lbl
            def fit(self, X, **kw): return self
            def fit_predict(self, X, **kw): return self.labels_
            def generate_prediction_data(self): pass

        topic_model = BERTopic(
            embedding_model=None,
            umap_model=PrecomputedUMAP(),
            hdbscan_model=PrecomputedHDBSCAN(labels),
            vectorizer_model=CountVectorizer(
                min_df=5, ngram_range=(1, 2),
                token_pattern=r"(?u)\b\w[\w\-]+\b"),
            top_n_words=10, verbose=False,
        )
        return topic_model.fit_transform(docs, embeddings)

    _, t_fit = timer(run_bertopic_fit, "bertopic_fit")
    stages["bertopic_fit"] = round(t_fit, 3)

    # ─── UMAP 2D ───────────────────────────────────────────────────
    if backend == "mlx":
        def umap_2d():
            from mlx_bertopic.umap import MlxUMAPWrapper
            model = MlxUMAPWrapper(
                n_neighbors=10, n_components=2, min_dist=0.0,
                random_state=42, normalize=True,
            )
            return model.fit_transform(embeddings)
    else:
        def umap_2d():
            from umap import UMAP
            kw = dict(n_neighbors=10, n_components=2, min_dist=0.0,
                      metric="cosine", low_memory=True)
            if multicore:
                kw["n_jobs"] = -1
            else:
                kw["random_state"] = 42
            return UMAP(**kw).fit_transform(embeddings)

    _, t_umap2d = timer(umap_2d, "umap_2d")
    stages["umap_2d"] = round(t_umap2d, 3)

    # ─── Summary ───────────────────────────────────────────────────
    total = sum(stages.values())
    peak_rss = round(get_peak_rss_mb(), 1)

    print(f"\n{'='*55}")
    print(f"TOTAL: {total:.1f}s | Peak RSS: {peak_rss:.0f} MB | Topics: {n_topics}")
    print(f"{'='*55}")

    result = {
        "hostname": hostname,
        "backend": backend,
        "multicore": multicore,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": f"ptt_khs ({embeddings.shape[0]:,} × {embeddings.shape[1]})",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "stages_seconds": stages,
        "total_seconds": round(total, 3),
        "peak_rss_mb": peak_rss,
        "n_topics": n_topics,
        "outlier_pct": outlier_pct,
    }

    out_file = f"result_{hostname}_{mode_label}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_file}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--backend", choices=["cpu", "mlx"], default="cpu")
    parser.add_argument("--multicore", action="store_true")
    args = parser.parse_args()
    run_benchmark(args.data_dir, args.backend, args.multicore)
