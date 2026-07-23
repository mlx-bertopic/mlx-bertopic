# mlx-bertopic

**BERTopic on Apple Silicon — MLX (Metal GPU) backends.**

`mlx-bertopic` provides independent MLX re-implementations of BERTopic's
components, accelerated on Apple Silicon via the Metal GPU. Think of it as the
Apple-Silicon counterpart to [RAPIDS cuML](https://rapids.ai) GPU backends for
BERTopic: backends for `embedding_model`, `umap_model`, `hdbscan_model`,
`ctfidf_model`, and `representation_model`.

> Each component is an **independent MLX reimplementation, not a port** of the
> upstream source. Behavior is aligned to the upstream reference and verified by
> golden tests (see `NOTICES.md`).

## Packages

This is a monorepo; `pip install mlx-bertopic` installs all of them.

| Module | Role | Aligned to |
|---|---|---|
| `mlx_bertopic` | Umbrella: BERTopic backends (wraps the modules below) | BERTopic 0.17.4 |
| `mlx_ctfidf` | Class-based TF-IDF (c-TF-IDF) | BERTopic `ClassTfidfTransformer` |
| `mlx_hdbscan` | HDBSCAN clustering (Metal distance + MST) | hdbscan 0.8.x |
| `mlx_keybert` | KeyBERT-inspired + MMR keyword selection | KeyBERT |
| `mlx_lda` | Latent Dirichlet Allocation (variational EM) | scikit-learn LDA / Blei 2003 |
| `mlx_dtm` | Dynamic topic modeling (LSTM smoother + NeuralDTM) | BERTopic topics-over-time (concept) |
| `mlx_outlier` | BERTopic outlier reduction | BERTopic outlier strategies |

## Where MLX (Metal) wins

Benchmarked on M4 Max (128 GB unified memory), 214,515 documents × 4,096
dimensions. Full methodology and raw data in `benchmarks/`.

| Stage | MLX (Metal) | CPU | Speedup | Notes |
|---|---|---|---|---|
| UMAP (5D + 2D) | 63s | 1,911s | **~30×** | NNDescent on GPU; bandwidth-bound on CPU |
| HDBSCAN (distance + MST) | — | — | 45–135× at 5K–20K | O(n²) pairwise on GPU |
| HDBSCAN (tree extraction) | — | — | no benefit | O(n) pointer-chasing; keep on CPU |
| c-TF-IDF | — | — | scale-dependent | crossover ≈ 100 topics × 50K vocab |
| **Full pipeline** | **79s** | **1,944s** | **24.6×** | end-to-end, single machine |

On the same data, Intel Xeon servers (4–6 core, DDR4) take **89 minutes**
single-threaded — 67× slower than MLX. Multi-threading on low-bandwidth
hardware can be *counterproductive* (see `benchmarks/RESULTS.md`).

## Known Issues & Workarounds

### mlx-vis UMAP embedding divergence (>100K high-dim inputs)

`mlx-vis` UMAP has a numerical stability issue on large datasets (>100K points)
with high-dimensional embeddings (>1000d). Hub nodes can diverge due to
gradient accumulation in batch scatter-add operations.

**Symptoms**: HDBSCAN finds only 1 giant cluster; UMAP coordinate range expands
to ±hundreds instead of the normal ±15.

**This repo includes a workaround** in `MlxUMAPWrapper`: post-hoc L2 norm
clipping (max_norm=15) + hard coordinate bound (±20). This is transparent —
no user action required.

For the full upstream fix (clip ordering + per-epoch projection), see
`patches/mlx_vis_umap_stability.md`. A PR to `hanxiao/mlx-vis` is pending.

**Verified on**:
- 164K × 4096d: 133 topics (was 21 pre-fix), 24x faster than CPU
- 100K × 4096d: 132 topics (was 83 pre-fix)
- 32K and below: unaffected

## Install

```bash
pip install "git+https://github.com/mlx-bertopic/mlx-bertopic.git"
# optional: viz extras for dynamic-topic plotting
pip install "git+https://github.com/mlx-bertopic/mlx-bertopic.git[viz]"
```

Apple Silicon only (MLX = Metal). Python ≥ 3.10.

> Note: depending on `bertopic` transitively pulls PyTorch / sentence-transformers
> (BERTopic's own requirements). The MLX backends bypass them at runtime, but
> they are installed for BERTopic's base classes.

## Quick start

```python
from bertopic import BERTopic
from mlx_bertopic import MlxEmbedder, MlxUMAPWrapper, MlxCTFIDF
from mlx_hdbscan import HDBSCAN as MlxHDBSCAN

topic_model = BERTopic(
    embedding_model=MlxEmbedder("mlx-community/bge-small-en-v1.5-bf16"),
    umap_model=MlxUMAPWrapper(n_components=5, n_neighbors=15),
    hdbscan_model=MlxHDBSCAN(min_cluster_size=15),
    ctfidf_model=MlxCTFIDF(),
)
topics, probs = topic_model.fit_transform(documents)
```

Individual modules can also be used directly, e.g.:

```python
from mlx_ctfidf import CTFIDFTransformer      # c-TF-IDF aligned to BERTopic
from mlx_hdbscan import HDBSCAN               # HDBSCAN aligned to the reference
from mlx_lda import LDA                       # variational-EM LDA
from mlx_dtm import NeuralDTM                 # embedding-space semantic drift
```

### NeuralDTM: Track topic semantic drift over time

```python
from mlx_dtm import NeuralDTM

dtm = NeuralDTM(n_bins=20, smooth=True)
dtm.fit(embeddings, topics, timestamps)

# Which topics drifted most?
dtm.top_drifting_topics(10)

# When did a topic shift sharply?
dtm.changepoints(topic_id=3)

# Keywords per time bin (needs vocab embeddings)
dtm.keywords_over_time(topic_id=3, vocab_embeddings, vocab_words)

# Tidy DataFrame for plotting
dtm.drift_dataframe()
```

## Alignment contract (how we stay in sync with upstream)

Because these are reimplementations, not ports, there is no shared code to
track. Each module pins the upstream version it is aligned to and ships golden
tests that compare its output against the upstream reference:

- `mlx_ctfidf` — output vs `bertopic.vectorizers.ClassTfidfTransformer`
- `mlx_hdbscan` — cluster labels vs Cython `hdbscan`, including non-convex
  data (`make_moons` / `make_circles`), ARI ≥ 0.9
- `mlx_lda`, `mlx_dtm`, `mlx_keybert`, `mlx_outlier` — behavioral / smoke tests

When upstream releases a new version, run the tests to detect drift.

## Repository layout

```
mlx_bertopic/   mlx_ctfidf/   mlx_hdbscan/   mlx_keybert/
mlx_lda/        mlx_dtm/      mlx_outlier/   # the 7 modules
tests/          # all golden + integration tests
benchmarks/     # benchmark scripts + research notes
examples/       # usage scripts
METHODOLOGY.md  # research methodology (Apple-Silicon / Metal analysis)
NOTICES.md      # upstream attributions (BERTopic MIT; hdbscan, sklearn BSD-3)
```

## License

MIT (© 2026 Felix Lin). Upstream attributions in `NOTICES.md`.
