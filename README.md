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
| `mlx_dtm` | LSTM-based dynamic topic modeling | BERTopic topics-over-time (concept) |
| `mlx_outlier` | BERTopic outlier reduction | BERTopic outlier strategies |

## Where MLX (Metal) wins

Preliminary, measured on M4 Max (see `benchmarks/` for the methodology and full
tables; reproducible benchmark script is a tracked roadmap item):

| Stage | MLX (Metal) vs CPU | Verdict |
|---|---|---|
| Embedding | ~16× faster | ✅ must use MLX |
| UMAP | ~12× faster | ✅ must use MLX |
| HDBSCAN pairwise-distance + MST | 45–135× faster at 5K–20K points | ✅ large-scale |
| HDBSCAN cluster extraction (tree) | no GPU benefit (O(n) pointer-chasing) | ❌ keep on CPU |
| c-TF-IDF | crossover ≈ 100 topics × 50K vocab | ⚠️ scale-dependent |

**Key insight:** HDBSCAN's cluster extraction is an inherently sequential tree
traversal — it belongs on the CPU. The GPU win is concentrated in the O(n²)
pairwise-distance + MST stage. A pure-GPU HDBSCAN that cut corners (0%
agreement with the reference) is documented in the research notes as a
cautionary tale; the shipped hybrid (GPU distance+MST, CPU extraction) matches
the reference implementation.

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
