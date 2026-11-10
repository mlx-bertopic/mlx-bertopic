---
name: mlx-bertopic
description: Run BERTopic topic modeling on Apple Silicon with MLX Metal backends (embeddings, UMAP, HDBSCAN, c-TF-IDF, LDA, dynamic-topic drift, 14 HTML visualizations). Use for topic modeling, semantic clustering of embeddings, topic drift over time, cross-lingual topic comparison, or when working in the mlx-bertopic repo itself.
triggers:
  - "topic modeling"
  - "BERTopic"
  - "mlx-bertopic"
  - "主題建模"
  - "c-TF-IDF"
  - "topic drift"
  - "主題漂移"
metadata:
  tags: [mlx, bertopic, topic-modeling, apple-silicon, metal, embeddings, umap, hdbscan, ctfidf, lda, dtm]
---

# mlx-bertopic — BERTopic on Apple Silicon (MLX / Metal)

Repository: `github.com/mlx-bertopic/mlx-bertopic` (MIT). Verified on macOS 27.2,
arm64 (M4 Max), Python 3.11.18, bertopic 0.17.0, mlx 0.29.3.

Every backend is an independent MLX reimplementation, not a port; each is aligned
to a pinned upstream reference and guarded by golden tests (`mlx_ctfidf` vs
BERTopic's `ClassTfidfTransformer`, `mlx_hdbscan` vs Cython `hdbscan`).

## When to use

- Topic modeling on an M-series Mac (no CUDA, no cloud)
- Clustering or labeling >10K document embeddings
- Tracking how topics drift over time (`NeuralDTM`)
- Comparing topics across languages through a shared multilingual embedding space

## 1. Environment (hard requirements)

| Item | Value |
|---|---|
| OS / hardware | Apple Silicon only (MLX = Metal) |
| Python | ≥ 3.10 (3.11 verified) |
| Avoid | macOS system `/usr/bin/python3` (3.9) — no `mlx-embeddings`, and the repo uses 3.10+ syntax |
| Core deps | `mlx>=0.20`, `mlx-embeddings>=0.1.0`, `mlx-vis>=0.7`, `bertopic>=0.16`, `numpy`, `scipy` |
| Optional | `hdbscan` (Cython reference), `plotly`, `pandas`, `pyarrow` (viz) |

```bash
pip install "git+https://github.com/mlx-bertopic/mlx-bertopic.git"
# or from a checkout
git clone https://github.com/mlx-bertopic/mlx-bertopic.git && cd mlx-bertopic
python3.11 -m pip install -e .
```

A script that lives outside the checkout cannot import the package unless it is
installed (`pip install -e .`) or `PYTHONPATH=<checkout>` is set — `sys.path[0]`
is the script's directory, not the cwd.

## 2. Component map

| Import | Public API | Role |
|---|---|---|
| `mlx_bertopic` | `MlxEmbedder, MlxUMAPWrapper, MlxCTFIDF, MlxKeyBERT, MlxMMR, reduce_outliers_embeddings, reduce_outliers_ctfidf, generate_all_viz` | lazy umbrella (each backend imported on first use) |
| `mlx_hdbscan` | `HDBSCAN(min_cluster_size=15, min_samples=None, metric="euclidean", algorithm="auto", knn_k=None, batch_size=4096, verbose=True)` | density clustering |
| `mlx_ctfidf` | `CTFIDFTransformer(reduce_frequent_words=False, bm25_weighting=False)` | c-TF-IDF aligned to BERTopic |
| `mlx_keybert` | `KeyBERTInspired(top_n_words), MaximalMarginalRelevance(top_n_words, diversity)` | standalone keyword selection |
| `mlx_lda` | `LDA(n_topics=10, n_iter=30, learning_method="auto", batch_size=512, ...)` | variational-EM LDA |
| `mlx_dtm` | `NeuralDTM(n_bins=20, min_docs_per_bin=10, smooth=True, hidden_dim=64, n_epochs=30)` | embedding-space topic drift |
| `mlx_outlier` | `reduce_outliers_embeddings(topics, embeddings, threshold=0.0, batch_size=4096)`, `reduce_outliers_ctfidf(...)` | outlier reassignment |
| `mlx_bertopic.viz` | `generate_all(topics_assigned, topic_keywords, umap_2d=None, output_dir="viz", neural_dtm=None, title_prefix="", top_n=25, sample_size=20000)` | 14 HTML reports |

## 3. Workflow: full BERTopic pipeline (default path)

Verified end-to-end: 32 docs, 4 clusters, correct keywords with KeyBERT scores.

```python
from bertopic import BERTopic
from mlx_bertopic import MlxEmbedder, MlxUMAPWrapper, MlxCTFIDF, MlxKeyBERT
from mlx_hdbscan import HDBSCAN as MlxHDBSCAN

topic_model = BERTopic(
    embedding_model=MlxEmbedder("mlx-community/bge-small-en-v1.5-bf16"),
    umap_model=MlxUMAPWrapper(n_components=5, n_neighbors=15),
    hdbscan_model=MlxHDBSCAN(min_cluster_size=15, verbose=False),
    ctfidf_model=MlxCTFIDF(),
    representation_model=MlxKeyBERT(top_n_words=10),
    calculate_probabilities=False,
    verbose=False,
)
topics, probs = topic_model.fit_transform(documents)
topic_model.get_topic(0)[:5]          # [(word, weight), ...]
```

Already have embeddings? Pass them and BERTopic never calls the embedding model:
`topic_model.fit_transform(documents, embeddings=np.ndarray)`.

`MlxEmbedder.embed(documents)` → `np.float32[N, dim]`; batching is internal
(64 docs/request), so a 300K-doc corpus works but is bandwidth-bound.

## 4. Workflow: clustering backend choice

| `mlx_hdbscan.HDBSCAN(algorithm=...)` | Data size | Memory |
|---|---|---|
| `"dense"` | n ≤ 30K | O(n²) pairwise distances |
| `"sparse"` | n > 30K | O(n×k) via NNDescent KNN |
| `"auto"` (default) | any | picks the above from n |

The Metal path covers pairwise distance + MST; condensed-tree extraction is O(n)
pointer chasing and stays on CPU by design. Below ~10K documents the Cython
`hdbscan` package is usually faster — swap it in without touching anything else.

## 5. Workflow: c-TF-IDF

`MlxCTFIDF` is the drop-in `ctfidf_model=`. Two flags with BERTopic semantics:
`bm25_weighting=True` (BM25 IDF), `reduce_frequent_words=True` (`sqrt(TF)` after
L1). Matrices above 5e8 elements (≈ 100 topics × 50K vocab and up) fall back to a
numpy path and warn `Matrix too large for GPU (...), falling back to CPU` —
expected, not an error.

## 6. Workflow: representation models

`MlxKeyBERT` / `MlxMMR` subclass `bertopic.representation.BaseRepresentation`, so
BERTopic accepts them as `representation_model=`; when the model keeps no cached
embeddings (0.17 keeps none) they re-embed `documents['Document']` themselves.

```python
from mlx_bertopic import MlxKeyBERT, MlxMMR
topic_model = BERTopic(..., representation_model=MlxKeyBERT(top_n_words=10))
```

On a checkout older than 0.6.1 the `isinstance` check raises
`TypeError: unsupported type MlxKeyBERT for representation_model`; use upstream's
stage instead (`from bertopic.representation import KeyBERTInspired`), or call the
MLX extractor standalone after fit:

```python
from mlx_keybert import KeyBERTInspired as MlxKB, MaximalMarginalRelevance
result = MlxKB(top_n_words=10).extract_topics(
    topic_ids=[t for t in set(topics) if t >= 0],
    doc_embeddings=embeddings,
    topic_assignments=topics,
    word_embeddings=embedder.embed(vocab),
    vocab=vocab,
)   # {topic_id: [(word, score), ...]}; MaximalMarginalRelevance has the same call
```

## 7. Workflow: outlier reduction

```python
from mlx_bertopic import reduce_outliers_embeddings
topics = reduce_outliers_embeddings(topics, embeddings, threshold=0.3)
```

`threshold` is a **cosine similarity** floor: `0.0` reassigns every outlier to
the nearest topic centroid, higher values only reassign confident matches.
BERTopic's own `embeddings` strategy takes a cosine *distance*, so upstream
threshold values are not transferable.

## 8. Workflow: LDA

```python
from mlx_lda import LDA
lda = LDA(n_topics=4, n_iter=10, verbose=False).fit(documents)
lda.get_topics(10)        # [[(word, prob), ...], ...]
lda.transform(documents)  # doc-topic matrix
lda.save("/tmp/lda.npz"); LDA.load("/tmp/lda.npz")
```

`learning_method="auto"` switches between batch VI (memory fits) and online VI
(`O(batch_size × vocab)`), so 300K documents are fine; batch mode alone OOMs
above ~250K.

## 9. Workflow: dynamic topic drift (`NeuralDTM`)

```python
from mlx_dtm import NeuralDTM
dtm = NeuralDTM(n_bins=20, smooth=True).fit(embeddings, topics, timestamps)
dtm.drift_dataframe()            # columns: topic, bin, timestamp, drift
dtm.top_drifting_topics(10)      # [(topic_id, drift), ...]
dtm.changepoints(0)              # bin indices with sharp semantic shifts
dtm.keywords_over_time(0, vocab_embeddings, vocab_words)
```

Works in embedding space (topic centroid movement), not Blei's word-distribution
random walk — it needs the embeddings the topic model saw plus a timestamp per
document. Coarse timestamps (day/month) only support coarse conclusions.

## 10. Workflow: visualizations

```python
from mlx_bertopic.viz import generate_all
generate_all("assigned.parquet", "keywords.json", "umap2d.npy",
             output_dir="viz", neural_dtm=dtm, title_prefix="corpus")
```

Inputs: parquet with `topic`, `post_date`, `docs_filtered` columns; JSON
`{topic_id: [[word, score], ...]}`; optional `npy` of 2-D UMAP coordinates.
Writes 14 HTML files: `barchart, documents, heatmap, hierarchy,
topics_over_time, topic_trends, sunburst, neural_dtm_drift, word_clouds,
intertopic_map, centroid_trajectory, temporal_heatmap, topic_network,
representative_docs`. Needs `plotly`, `pandas`, `pyarrow`.

## 11. Recipe: cross-lingual topic comparison (BTM)

Not shipped as a module — it is the `METHODOLOGY.md` §2.3 recipe. Requires that
both languages were embedded by the same multilingual model:

```python
import numpy as np
def centroids(emb, topics):
    ids = [t for t in sorted(set(topics)) if t >= 0]
    C = np.stack([emb[np.asarray(topics) == t].mean(0) for t in ids])
    return ids, C / np.linalg.norm(C, axis=1, keepdims=True)
ids_a, A = centroids(emb_a, topics_a)
ids_b, B = centroids(emb_b, topics_b)
sim = A @ B.T                       # (N_a, N_b)
best = sim.max(1); match = sim.argmax(1)
# >0.85 high confidence, 0.70-0.85 medium, <0.70 low
```

## 12. Maintaining the repo

```bash
python -m pytest tests -q          # expected: 75 passed
```

- `tests/` holds golden/parity tests; a green run after a `bertopic` or
  `mlx-vis` upgrade is the drift check.
- Versions live in `pyproject.toml` and `mlx_bertopic/__init__.py`; bump both and
  add a `CHANGELOG.md` entry — this repo releases per fix.
- `tests/test_skill.py` fails if this file names an API that no longer imports;
  update the skill in the same commit as any public rename.

## 13. Pitfalls (all observed)

| Symptom | Cause | Action |
|---|---|---|
| `NotImplementedError: ... incremental transform on new data` | `MlxUMAPWrapper.transform()` only serves the fitted array; `BERTopic.transform(new_docs)` needs a refit | refit on the combined dataset (`fit_transform`) |
| `TypeError: unsupported type MlxKeyBERT for representation_model` | wrapper was not a `BaseRepresentation` subclass (checkout < 0.6.1) | upgrade, or use the §6 fallbacks |
| `ModuleNotFoundError: mlx_embeddings` | ran with a Python that has no MLX deps (e.g. macOS system 3.9) | use a 3.11+ interpreter with the deps installed |
| `ModuleNotFoundError: mlx_bertopic` | script outside the checkout and package not installed | `pip install -e .` or `PYTHONPATH=<checkout>` |
| Only 1 giant cluster, UMAP coords at ±hundreds | mlx-vis gradient divergence on >50K × >1000d | already handled by `MlxUMAPWrapper` stabilization (`MAX_NORM=15`, `HARD_CLIP=20`); don't disable it |
| `Matrix too large for GPU (...)` | c-TF-IDF above 5e8 elements | expected numpy fallback |
| Extra embedding pass per `fit_transform` | BERTopic ≥ 0.17 keeps no corpus embeddings, so the representation stage re-embeds `documents['Document']` | expected; omit `representation_model` or cache `embed()` results if it dominates |
