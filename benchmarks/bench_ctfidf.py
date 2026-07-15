#!/usr/bin/env python3.11
"""Benchmark c-TF-IDF: scipy/sklearn vs MLX (Metal GPU)."""
import mlx.core as mx
import numpy as np
import time

def ctfidf_mlx(X_np):
    """c-TF-IDF using MLX arrays on Metal GPU."""
    X = mx.array(X_np.astype(np.float32))
    
    # L1 normalize rows
    row_sums = mx.sum(X, axis=1, keepdims=True)
    row_sums = mx.maximum(row_sums, 1e-9)
    X_norm = X / row_sums
    
    # IDF
    df = mx.sum(X, axis=0)
    avg_nr = mx.mean(mx.sum(X, axis=1))
    # avoid divide by zero
    df_safe = mx.maximum(df, 1e-9)
    idf = mx.log((avg_nr / df_safe) + 1)
    
    # TF × IDF
    result = X_norm * idf
    
    # L2 normalize
    norms = mx.sqrt(mx.sum(result * result, axis=1, keepdims=True))
    norms = mx.maximum(norms, 1e-9)
    result = result / norms
    mx.eval(result)
    return result

def ctfidf_numpy(X_np):
    """c-TF-IDF using pure numpy."""
    X = X_np.astype(np.float32)
    
    row_sums = np.maximum(X.sum(axis=1, keepdims=True), 1e-9)
    X_norm = X / row_sums
    
    df = np.maximum(X.sum(axis=0), 1e-9)
    avg_nr = X.sum(axis=1).mean()
    idf = np.log((avg_nr / df) + 1)
    
    result = X_norm * idf
    
    norms = np.maximum(np.sqrt((result ** 2).sum(axis=1, keepdims=True)), 1e-9)
    result = result / norms
    return result

# Warm up MLX
mx.eval(mx.array([1.0]))

print(f"{'Scale':<35} {'numpy':>10} {'MLX':>10} {'Speedup':>10}")
print("-"*68)

for n_topics, vocab in [
    (12, 10000),
    (50, 30000),
    (100, 50000),
    (500, 100000),
    (1000, 200000),
    (3000, 300000),
    (5000, 500000),
]:
    X = np.random.negative_binomial(1, 0.1, size=(n_topics, vocab)).astype(np.float32)
    X[X > 5] = 0
    
    # numpy
    t0 = time.time()
    ctfidf_numpy(X)
    t_np = time.time() - t0
    
    # MLX
    t0 = time.time()
    ctfidf_mlx(X)
    t_mx = time.time() - t0
    
    sp = t_np / t_mx if t_mx > 0.001 else 0
    print(f"  {n_topics:>5} topics × {vocab:>6} vocab  {t_np*1000:>7.1f}ms {t_mx*1000:>7.1f}ms {sp:>7.1f}x")
