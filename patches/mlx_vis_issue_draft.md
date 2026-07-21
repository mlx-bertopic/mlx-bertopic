# GitHub Issue: Embedding divergence on large (>100K) high-dimensional datasets

## Title

**[BUG] UMAP embedding divergence on >100K high-dimensional inputs — batch scatter-add gradient accumulation**

## Body

---

### Description

When running UMAP on large datasets (>100K points) with high-dimensional embeddings (4096d), a small number of hub nodes diverge to extreme coordinate values (observed: ±665), effectively collapsing the entire embedding into a single dense blob + a few far-flung outliers.

This makes downstream clustering (HDBSCAN) fail — it sees one giant cluster instead of meaningful structure.

### Environment

- mlx-vis 0.2.x (latest as of 2026-07-20)
- Apple M4 Max, 128GB
- macOS, Python 3.14
- mlx 0.24.x

### Reproduce

```python
import numpy as np
from mlx_vis import UMAP

# Load any 100K+ high-dim embeddings (e.g. text embeddings from qwen3-embedding)
# For reproduction: random data with realistic structure also triggers it
rng = np.random.default_rng(42)
# Simulate clustered high-dim data
centers = rng.standard_normal((50, 4096)).astype(np.float32) * 0.05
labels = rng.integers(0, 50, size=150_000)
X = centers[labels] + rng.standard_normal((150_000, 4096)).astype(np.float32) * 0.01

result = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, n_epochs=200, random_state=42, verbose=True).fit_transform(X)

print(f"Range: [{result.min():.1f}, {result.max():.1f}]")
print(f"Max abs: {np.abs(result).max():.1f}")
# Expected: range ±10-15
# Actual: range ±200-700 (diverged)
```

### Observed Behavior

| Dataset | Size | UMAP range | HDBSCAN result |
|---------|------|-----------|----------------|
| Text embeddings (4096d) | 164K | **[-665, +744]** | 21 topics, 149K in one cluster (91%) |
| Text embeddings (4096d) | 100K | [-50, +70] | 83 topics, largest cluster 20K (20%) |
| Fashion-MNIST (784d) | 70K | [-12, +15] | Normal (no issue) |

The bug scales with dataset size and input dimensionality. At 70K/784d it doesn't manifest. At 100K/4096d it's mild. At 164K/4096d it completely breaks.

### Root Cause

In `_sgd_step`, gradients are applied via batch scatter-add:

```python
Y = Y.at[ef].add(pos_grad)
Y = Y.at[et].add(-pos_grad)
Y = Y.at[neg_from_idx].add(neg_grad)
```

A hub node appearing in N active edges receives N gradient contributions **simultaneously**. With `negative_sample_rate=5`, a well-connected node can receive 50+ updates in a single step. Each is clipped to ±4, but they accumulate to ±200+.

The reference CPU UMAP (lmcinnes/umap) doesn't have this issue because it updates sequentially — each edge sees the node's *current* position after previous updates.

Additionally, the clip ordering amplifies the issue:
```python
# Current: clips intermediate value, then scales by alpha
pos_grad = mx.clip(grad_coeff * diff, -4.0, 4.0) * alpha_epoch
```

The ±4 clip should bound the *final displacement*, not the pre-alpha value.

### Suggested Fix

**Two changes in `mlx_vis/_umap/umap.py`:**

**1. Clip the final displacement (in `_sgd_step`):**

```diff
- pos_grad = mx.clip(grad_coeff * diff, -4.0, 4.0) * alpha_epoch
+ pos_grad = mx.clip(grad_coeff * diff * alpha_epoch, -4.0, 4.0)

- neg_grad = mx.clip(neg_grad_coeff * neg_diff, -4.0, 4.0) * alpha_epoch
+ neg_grad = mx.clip(neg_grad_coeff * neg_diff * alpha_epoch, -4.0, 4.0)
```

**2. Per-epoch embedding projection (in `_optimize`, after `_sgd_step`):**

```python
Y = self._sgd_step(Y, ef, et, neg_from_idx, neg_to_idx, alpha_epoch, a_mx, b_mx)

# Projected gradient descent: prevent divergence from scatter-add accumulation
norms = mx.sqrt(mx.sum(Y * Y, axis=1, keepdims=True) + 1e-8)
max_norm = mx.array(15.0)
scale = mx.minimum(max_norm / norms, mx.array(1.0))
Y = Y * scale
Y = mx.clip(Y, -20.0, 20.0)
```

**Rationale for bounds:**
- Spectral initialization places points in [0, 10]
- CPU UMAP outputs range ±12–22 across all tested datasets
- `max_norm=15` covers >99.9% of legitimate points
- `hard_clip=20` is a final safety net that never activates under normal operation

### Results After Fix

| Dataset | Before | After | CPU UMAP (reference) |
|---------|--------|-------|---------------------|
| 164K×4096 | 21 topics, biggest=149K | **133 topics, biggest=6.5K** | 119 topics, biggest=4K |
| 100K×4096 | 83 topics, biggest=20K | **132 topics, biggest=2.5K** | 99 topics |
| 32K×4096 | OK (mild distortion) | **79 topics** (improved) | 50 topics |
| 11K×4096 | OK | **48 topics** (unchanged) | 48 topics |

Performance is unchanged (still 10-20s for 164K on M4 Max).

### Notes

- The fix is a standard technique: **Projected Gradient Descent** — project the solution back onto a feasible set after each step. It's the GPU-parallel equivalent of the implicit bounding that CPU UMAP gets from sequential updates.
- This only affects large datasets with high edge-density in the KNN graph. Fashion-MNIST (70K) benchmarks won't catch it because the graph is sparser at 784d.
- Happy to submit a PR if you'd like. The change is minimal and doesn't affect existing benchmarks.

---

### Labels suggestion

`bug`, `numerical stability`
