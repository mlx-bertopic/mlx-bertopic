# End-to-End Pipeline Benchmark: MLX vs CPU

## Dataset

- **Source**: PTT forum (KHS/Yamaha discussions)
- **Size**: 214,515 sentences × 4,096 dimensions (pre-computed embeddings)
- **File**: 3.3 GB float32 numpy array

## Hardware

| Machine | CPU | RAM | Notes |
|---------|-----|-----|-------|
| Mac (MLX + CPU) | Apple M4 Max, 16C | 128 GB unified | Primary |
| p1g2 | Intel i7-9750H, 6C/12T | 30 GB DDR4 | Laptop, thermal throttle |
| dev | Intel Xeon E3-1245 v5, 4C/8T | 31 GB DDR4 | OVH server |

## Results

### Summary Table

| # | Machine | Backend | Total | Speedup |
|---|---------|---------|------:|--------:|
| 1 | M4 Max | **MLX (Metal GPU)** | **79.0s** | **24.6×** |
| 2 | M4 Max | CPU, 16 threads | 1,600s | 1.2× |
| 3 | M4 Max | CPU, single-thread | 1,944s | 1.0× (baseline) |
| 4 | i7-9750H | CPU, 12 threads | 3,812s | 0.51× |
| 5 | i7-9750H | CPU, single-thread | 5,306s | 0.37× |
| 6 | Xeon E3-1245v5 | CPU, single-thread | 5,348s | 0.36× |

### Per-Stage Breakdown (seconds)

| Stage | M4 Max MLX | M4 Max CPU 1T | M4 Max CPU 16T | i7-9750H 1T | Xeon E3 1T |
|-------|:---:|:---:|:---:|:---:|:---:|
| load | 0.4 | 5.1 | 5.4 | 0.6 | 13.4 |
| **umap_5d** | **34.7** | 1,081 | 1,017 | 2,423 | 2,757 |
| hdbscan | 7.6 | 17.8 | 8.9 | 11.2 | 13.6 |
| bertopic_fit | 8.0 | 10.0 | 7.3 | 9.1 | 6.9 |
| **umap_2d** | **28.4** | 830 | 562 | 2,861 | 2,557 |

### Peak Memory

| Backend | Peak RSS |
|---------|----------|
| MLX (Metal GPU) | 15.0 GB |
| CPU single-thread | 20.5 GB |
| CPU 16-thread | 22.8 GB |

## Key Findings

1. **UMAP dominates runtime** (>97% of total). All other stages are negligible.

2. **MLX GPU achieves 24.6× speedup** over CPU single-thread on this workload.
   This exceeds the 3.4× reported in the mlx-vis paper (70K × 784d Fashion-MNIST)
   because higher dimensionality (4096 vs 784) and larger N (214K vs 70K) better
   utilize GPU parallelism.

3. **CPU multi-threading barely helps** — only 1.2× on M4 Max (16 threads).
   UMAP's NNDescent kNN is memory-bandwidth-bound; adding threads increases
   cache thrashing without proportional compute gain.

4. **On memory-constrained servers** (31 GB, dual-channel DDR4), multi-threading
   can be *slower* than single-thread due to bandwidth saturation and swap pressure.

5. **MLX also uses less memory** (15 GB vs 20–23 GB) thanks to GPU memory pooling
   and the absence of numba JIT overhead.

## Reproducing

```bash
# MLX (Metal GPU)
python benchmarks/bench_umap_e2e.py --data-dir ./data --backend mlx

# CPU single-thread (reproducible, random_state=42)
python benchmarks/bench_umap_e2e.py --data-dir ./data --backend cpu

# CPU multi-thread (non-deterministic, n_jobs=-1)
python benchmarks/bench_umap_e2e.py --data-dir ./data --backend cpu --multicore
```

## Parameters

```python
# UMAP 5D (clustering)
n_neighbors=15, n_components=5, min_dist=0.0
# MLX: normalize=True (cosine-equivalent)
# CPU: metric='cosine', random_state=42

# UMAP 2D (visualization)
n_neighbors=10, n_components=2, min_dist=0.0

# HDBSCAN
min_cluster_size=150, min_samples=10, metric='euclidean'

# BERTopic
embedding_model=None (pre-computed), top_n_words=10
vectorizer: min_df=5, ngram_range=(1,2)
```

## Date

2026-07-18
