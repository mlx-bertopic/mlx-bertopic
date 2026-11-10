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
# MLX embeddings backend (GPL-3.0) — only needed for MlxEmbedder
pip install "git+https://github.com/mlx-bertopic/mlx-bertopic.git[embed]"
# HTML reports from mlx_bertopic.viz.generate_all
pip install "git+https://github.com/mlx-bertopic/mlx-bertopic.git[viz]"
```

Apple Silicon only (MLX = Metal). Python ≥ 3.10.

> `mlx-embeddings` is deliberately **not** a base dependency: it is GPL-3.0 while
> this package is MIT. Without it, `MlxEmbedder` raises an `ImportError` carrying
> the install hint, and any object with `embed(documents) -> np.ndarray` works as
> `embedding_model` instead. See `NOTICES.md`.

> Note: depending on `bertopic` transitively pulls PyTorch / sentence-transformers
> (BERTopic's own requirements). The MLX backends bypass them at runtime, but
> they are installed for BERTopic's base classes.

### Upgrading from 0.6.x

`mlx-embeddings` moved out of the base install in **0.7.0** (it is GPL-3.0; this
package is MIT). If you use `MlxEmbedder`, add the extra:

```bash
pip install -U "git+https://github.com/mlx-bertopic/mlx-bertopic.git[embed]"
```

Everything else — UMAP, HDBSCAN, c-TF-IDF, KeyBERT/MMR, LDA, NeuralDTM, outlier
reduction — installs and runs unchanged. 0.7.0 also fixes three breaks against
BERTopic 0.17 (`umap_model.fit` → `transform`, `representation_model` type
check, c-TF-IDF save/load); see `CHANGELOG.md`.

## Agent skill

`skills/mlx-bertopic/SKILL.md` is an agent-facing playbook: environment
requirements, per-module workflows, verified pitfalls and the maintenance loop.
It is kept in this repo so it moves with the API (`tests/test_skill.py` fails if
the skill names something that no longer imports).

```bash
mkdir -p ~/.hermes/skills/mlops
cp -r skills/mlx-bertopic ~/.hermes/skills/mlops/
# other harnesses: point their skill directory at skills/mlx-bertopic
```

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
skills/         # agent skill (SKILL.md playbook, drift-tested)
METHODOLOGY.md  # research methodology (Apple-Silicon / Metal analysis)
NOTICES.md      # upstream attributions + third-party licenses + credits
```

## Credits

This repo is mostly other people's work, wrapped for Metal. Thanks to:

- **[mlx-vis](https://github.com/hanxiao/mlx-vis)** (Han Xiao, Apache-2.0) — the
  Metal UMAP behind `MlxUMAPWrapper` and the `NNDescent` graph used by sparse
  HDBSCAN. The UMAP stage is ~80% of this pipeline's runtime, and it is fast
  because of their kernels. `patches/` documents a stability fix for its SGD
  loop that is proposed upstream.
- **[mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)** (Prince
  Canuma, GPL-3.0) — `MlxEmbedder`. Note the license differs from this package;
  see `NOTICES.md` for what that means for redistribution.
- **[BERTopic](https://github.com/MaartenGr/BERTopic)** (Maarten Grootendorst,
  MIT) — the component decomposition and the interfaces every backend here is
  aligned to, plus the c-TF-IDF/outlier golden references.
- **[hdbscan](https://github.com/scikit-learn-contrib/hdbscan)** and
  **[UMAP](https://github.com/lmcinnes/umap)** (Leland McInnes, BSD-3-Clause) —
  the reference implementations the parity tests and the SGD fix compare against.
- **[MLX](https://github.com/ml-explore/mlx)** (Apple ML research, MIT) and
  **scikit-learn / NumPy / SciPy** developers — framework and reference math.

Full license texts and the per-component alignment table: `NOTICES.md`.

## License

MIT (© 2026 Felix Lin). Upstream attributions in `NOTICES.md`.
