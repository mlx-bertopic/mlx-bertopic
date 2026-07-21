# mlx-vis UMAP Stability Patch

## Problem

`mlx_vis.UMAP` exhibits embedding divergence on datasets >50K points with
high-dimensional inputs (e.g. 4096d text embeddings). A small number of hub
nodes accumulate excessive gradient updates from batch `scatter_add`, causing
their coordinates to fly to extreme values (observed: ±665 on a 164K dataset).

This makes downstream HDBSCAN clustering fail — nearly all points collapse
into a single giant cluster because the few divergent outliers distort the
density estimation.

## Root Cause

In `mlx_vis/_umap/umap.py`, the `_sgd_step` method uses parallel scatter-add:

```python
Y = Y.at[ef].add(pos_grad)
Y = Y.at[et].add(-pos_grad)
Y = Y.at[neg_from_idx].add(neg_grad)
```

If a node appears in 50+ edges in a single epoch, all gradients are summed
simultaneously. Combined with the clip ordering issue (clip before alpha
multiplication), a single step can displace a node by hundreds of units.

The reference CPU UMAP (lmcinnes/umap) avoids this by:
- Sequential per-edge updates (each edge sees the node's current position)
- `move_other=False` by default (only head moves)
- Implicit bound from sequential nature: node never drifts far in one epoch

## Fix (apply to `mlx_vis/_umap/umap.py`)

### Change 1: Clip final displacement, not intermediate gradient

```diff
-        pos_grad = mx.clip(grad_coeff * diff, -4.0, 4.0) * alpha_epoch
+        pos_grad = mx.clip(grad_coeff * diff * alpha_epoch, -4.0, 4.0)
```

```diff
-        neg_grad = mx.clip(neg_grad_coeff * neg_diff, -4.0, 4.0) * alpha_epoch
+        neg_grad = mx.clip(neg_grad_coeff * neg_diff * alpha_epoch, -4.0, 4.0)
```

**Why**: The ±4 clip should bound the actual per-edge displacement applied to
the embedding, not an intermediate value that gets further scaled.

### Change 2: Per-epoch embedding projection

After `self._sgd_step(...)` in `_optimize`, add:

```python
# Prevent embedding divergence (projected gradient descent)
# L2 norm clip: smooth, preserves relative direction
norms = mx.sqrt(mx.sum(Y * Y, axis=1, keepdims=True) + 1e-8)
max_norm = mx.array(15.0)
scale = mx.minimum(max_norm / norms, mx.array(1.0))
Y = Y * scale
# Hard bound as final safety net
Y = mx.clip(Y, -20.0, 20.0)
```

**Why**: Even with per-edge clip at ±4, a hub node with 50 edges can
accumulate ±200 per epoch. The projection ensures no point ever leaves the
feasible embedding region [−20, 20]^d. Normal points (within ±12) are
unaffected.

## Bounds Rationale

- UMAP spectral initialization places points in [0, 10]
- Normal optimized embeddings range ±12–15 (observed across CPU UMAP runs)
- `max_norm=15`: L2 ball covers >99.9% of legitimate points
- `hard_clip=20`: Only catches catastrophic divergence

## Validation

| Dataset | Before Fix | After Fix | CPU UMAP (reference) |
|---------|-----------|-----------|---------------------|
| KHS 164K×4096 | 21 topics, 149K monster | 133 topics, biggest 6.5K | 119 topics, biggest 4K |
| Pan Piano 100K×4096 | 83 topics, biggest 20K | 132 topics, biggest 2.5K | 99 topics |
| PTT Decathlon 11K | OK (below threshold) | 48 topics (same) | 48 topics |
| YouTube Decathlon 32K | OK (mild) | 79 topics | 50 topics |

## Status

- Applied as post-hoc `_stabilize()` in `MlxUMAPWrapper` (this repo)
- Applied directly to local `mlx_vis/_umap/umap.py` for full fix
- Upstream PR to `hanxiao/mlx-vis`: pending (will file with reproduce script)
