#!/usr/bin/env python3.11
"""Benchmark c-TF-IDF at extreme scale: numpy vs MLX."""
import numpy as np
import time
import mlx.core as mx

mx.eval(mx.array([1.0]))  # warmup

def ctfidf_numpy(X):
    """c-TF-IDF on numpy. X is dense (n_topics × vocab)."""
    row_sums = np.maximum(X.sum(axis=1, keepdims=True), 1e-9)
    X_norm = X / row_sums
    df = np.maximum(X.sum(axis=0), 1e-9)
    avg_nr = X.sum(axis=1).mean()
    idf = np.log((avg_nr / df) + 1)
    result = X_norm * idf
    norms = np.maximum(np.sqrt((result**2).sum(axis=1, keepdims=True)), 1e-9)
    return result / norms

def ctfidf_mlx(X_np):
    """c-TF-IDF on MLX (Metal GPU)."""
    X = mx.array(X_np.astype(np.float32))
    row_sums = mx.maximum(mx.sum(X, axis=1, keepdims=True), 1e-9)
    X_norm = X / row_sums
    df = mx.maximum(mx.sum(X, axis=0), 1e-9)
    avg_nr = mx.mean(mx.sum(X, axis=1))
    idf = mx.log((avg_nr / df) + 1)
    result = X_norm * idf
    norms = mx.maximum(mx.sqrt(mx.sum(result * result, axis=1, keepdims=True)), 1e-9)
    res = result / norms
    mx.eval(res)
    return res

print("c-TF-IDF extreme scale: numpy vs MLX\n")
print(f"{'n_topics':>10} {'vocab':>10} {'matrix':>10} {'numpy':>12} {'MLX':>12} {'speedup':>10}")
print("-" * 68)

for n_topics, vocab in [
    (12, 10000),
    (50, 30000),
    (100, 50000),
    (500, 100000),
    (1000, 200000),
    (3000, 300000),
    (5000, 500000),
    (10000, 500000),
    (20000, 500000),
    (50000, 500000),
]:
    matrix_gb = n_topics * vocab * 4 / 1e9
    if matrix_gb > 200:  # skip absurd sizes
        print(f"{n_topics:>10} {vocab:>10} {'SKIP':>10} (matrix = {matrix_gb:.0f}GB)")
        continue

    X = np.random.negative_binomial(1, 0.1, size=(n_topics, vocab)).astype(np.float32)
    X[X > 5] = 0

    # numpy
    t0 = time.time()
    ctfidf_numpy(X)
    t_np = time.time() - t0

    # MLX
    try:
        t0 = time.time()
        ctfidf_mlx(X)
        t_mx = time.time() - t0
        sp = t_np / t_mx if t_mx > 0.001 else 0
        mlx_str = f"{t_mx*1000:.0f}ms"
        sp_str = f"{sp:.1f}x"
    except Exception as e:
        mlx_str = f"OOM"
        sp_str = "—"

    print(f"{n_topics:>10} {vocab:>10} {matrix_gb:>8.1f}GB {t_np*1000:>10.0f}ms {mlx_str:>12} {sp_str:>10}", flush=True)
