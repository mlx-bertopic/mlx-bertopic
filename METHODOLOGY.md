# MLX-Native Topic Modeling Pipeline: Research Outline & Methodology

## Research goal

Build a complete GPU-native topic modeling toolchain on Apple Silicon: a fully
automated pipeline from raw text to cross-lingual topic analysis, with no NVIDIA
GPU and no cloud resources.

---

## 1. Problem statement

### 1.1 Motivation

Bottlenecks of the existing large-scale topic modeling toolchain
(BERTopic + UMAP + HDBSCAN):

| Problem | Impact |
|---------|--------|
| UMAP (numba) is CPU-only | 100K+ documents take tens of minutes to hours |
| HDBSCAN (Cython) has no GPU acceleration | At scale, accuracy is the constraint even when speed is not |
| Japanese segmentation needs MeCab + external dictionaries | Convoluted environment setup, weak NER |
| LDA (sklearn) is single-threaded EM | Slow convergence at scale |
| No local Dynamic Topic Modeling | BERTopic DTM is bin-size sensitive and has no memory |
| Apple Silicon GPU sits idle | MPS supports only PyTorch compute |

### 1.2 Scope

Build the following 5 independent packages + 1 integrated pipeline:

```
┌─────────────────────────────────────────────────────────┐
│                    mlx-bertopic                          │
│      (integrated pipeline + CLI + report generation)     │
├─────────────┬──────────────┬──────────────┬─────────────┤
│  mlx-vis    │  mlx-hdbscan │   mlx-lda    │  mlx-dtm    │
│  (UMAP)     │ (clustering) │   (topics)   │  (temporal) │
│ [third-party]              │              │             │
├─────────────┴──────────────┴──────────────┴─────────────┤
│                      kwja-mlx                            │
│      (Japanese segmentation / POS / NER / reading)       │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Technical architecture

### 2.1 Shared design principles

1. **Drop-in replacement** — API identical to sklearn/hdbscan, switch with a
   single import
2. **Auto mode** — choose the algorithm path from data size and memory
3. **GPU-first, CPU-fallback** — everything that parallelizes goes on Metal;
   inherently sequential stages stay on CPU
4. **pip installable** — pyproject.toml + setuptools, `pip install -e .`

### 2.2 Per-module methodology

#### kwja-mlx (Japanese morphological analysis)

| Component | Model | Method | Accuracy |
|-----------|-------|--------|----------|
| Segmentation | DeBERTa v2 + Conv | Character-level B/I tagging | F1 ≈ 1.000 |
| POS | DeBERTa v2 | First-subword pooling + Linear | 99.91% (vs fp32) |
| NER | DeBERTa v2 + CRF | Emission scores + Viterbi decoding | 99.59%, 100% recall |
| Reading / base form | T5 base | Seq2seq generation | Identical to PyTorch |

**Quantization strategy:** q8 is recommended (fastest decode, no accuracy loss).
fp16 is actually *slower* under autoregressive decoding (Apple Silicon kernel
dispatch overhead).

**Batch inference:** `segment_batch(batch_size=16)` uses padding plus a batched
DeBERTa forward pass; 2.3× throughput.

#### mlx-hdbscan (density clustering)

| Mode | Suitable scale | Method | Characteristics |
|------|----------------|--------|-----------------|
| dense | n ≤ 30K | GPU pairwise distance + Prim MST + EOM | Exact, O(n²) memory |
| sparse | n > 30K | NNDescent KNN + sparse MRD + scipy Kruskal MST + EOM | Scalable, O(n×k) memory |
| auto | any | Selected from n | Default |

**Cluster extraction:** full Campello 2013 condensed tree with Excess-of-Mass
stability-based extraction. More conservative than the Cython version
(silhouette 33–55% higher).

**Source of divergence from Cython:** MST tie-breaking ambiguity (known
behavior; cuML documents the same).

#### mlx-lda (topic modeling)

| Mode | Method | Memory | Suitable for |
|------|--------|--------|--------------|
| batch | Batch Variational EM (Blei 2003) | O(n_docs × vocab) | < 250K docs |
| online | Online VI (Hoffman 2010) | O(batch_size × vocab) | any scale |
| auto | Selected from a memory estimate | adaptive | default |

**GPU acceleration:** the E-step (theta update) and M-step (beta update) are
both matrix multiplications; MLX Metal gives 9–16×.

#### mlx-dtm (dynamic topic modeling)

| Component | Method |
|-----------|--------|
| Time series construction | Sliding-window topic proportions (window_size, stride) |
| Smoothing | 2-layer LSTM (self-supervised: predict t+1 from 0..t) |
| Changepoint detection | Difference of the smoothed output > μ + kσ |
| Forecasting | Autoregressive generation from the final hidden state |

**Advantage over BERTopic DTM:** the LSTM hidden state carries memory across
time steps, so results do not depend on bin size and stay stable.

### 2.3 Cross-lingual comparison (BTM)

```
Language A (N_a topics) → centroid embeddings (4096D mean)
Language B (N_b topics) → centroid embeddings (4096D mean)
         ↓
L2-normalize → cosine similarity matrix (N_a × N_b)
         ↓
Per-topic best match → pairing-strength distribution
         ↓
High (>0.85) / Medium (0.7-0.85) / Low (<0.7)
```

**Precondition:** embeddings for every language come from the same multilingual
model, so they already live in one semantic space.

---

## 3. Validation

### 3.1 Correctness

| Package | Validation | Result |
|---------|------------|--------|
| kwja-mlx | 590 Japanese Wikipedia sentences × 4 precision modes | POS/NER/segmentation identical at full precision |
| kwja-mlx NER | CRF vs argmax | Zero B/I errors, 100% recall |
| mlx-hdbscan (dense) | make_blobs ARI | 0.995 |
| mlx-hdbscan (sparse) | dense vs sparse agreement | ARI = 1.000 (blobs) |
| mlx-lda | Perplexity convergence | Converges correctly (257→208) |
| mlx-dtm | Loss convergence | 0.31→0.23 |

### 3.2 Quality comparison

| Metric | MLX | Standard (sklearn/Cython) | Meaning |
|--------|-----|---------------------------|---------|
| Silhouette score | 0.52–0.61 | 0.39 | Tighter clusters |
| Intra-cluster distance | 0.35–0.38 | 0.40 | Tighter clusters |
| NER recall | 100% | — | CRF Viterbi effect |
| Topic coherence (LDA) | Semantically interpretable | char-ngram fragments | Benefit of word-level tokens |

### 3.3 Performance

| Task (scale) | MLX | Standard | Speedup |
|--------------|-----|----------|---------|
| UMAP large-scale × 4096D | 143s | ∞ (killed) | feasible vs infeasible |
| UMAP 10K | 0.5s | 17.3s | 35× |
| HDBSCAN 100K | 1.1s | 2.4s | 2.2× |
| LDA 50K | 2.3s | 21.6s | 9.3× |
| kwja segment 500 sentences | 1.55s (batched) | — | 323 sentences/sec |
| DTM training 498 steps | 8.7s | — | — |
| Full pipeline, large-scale | 18 min | impossible | — |

### 3.4 Edge cases

| Test | Result |
|------|--------|
| Empty string | ✅ Returns empty correctly |
| Over-long document (>512 tokens) | ⚠️ Caller must split into sentences (hard DeBERTa limit) |
| large-scale HDBSCAN sparse | ✅ 11.4s, no OOM |
| 300K LDA online | ✅ 80s, no OOM |
| q4 quantization degradation | POS 99.91%, NER 99.59% (usable) |

---

## 4. Toolchain inventory

| Package | Repository | Version | Status |
|---------|-----------|---------|--------|
| mlx-ctfidf | github.com/mlx-bertopic/mlx-bertopic | 0.2.0 | ✅ aligned to BERTopic 0.17.4 |
| mlx-hdbscan | github.com/mlx-bertopic/mlx-bertopic | 0.1.0 | ✅ aligned to hdbscan 0.8.x |
| mlx-keybert | github.com/mlx-bertopic/mlx-bertopic | 0.1.0 | ✅ KeyBERT + MMR |
| mlx-lda | github.com/mlx-bertopic/mlx-bertopic | 0.1.0 | ✅ variational-EM LDA |
| mlx-dtm | github.com/mlx-bertopic/mlx-bertopic | 0.1.0 | ✅ LSTM dynamic topics |
| mlx-outlier | github.com/mlx-bertopic/mlx-bertopic | 0.1.0 | ✅ outlier reduction |
| mlx-bertopic | github.com/mlx-bertopic/mlx-bertopic | 0.3.0 | ✅ umbrella pipeline |

---

## 5. Limitations and known issues

| Limitation | Cause | Mitigation |
|------------|-------|------------|
| Apple Silicon only | MLX = Metal only | By design |
| Sparse HDBSCAN labels differ from Cython | MST tie-breaking ambiguity | Known behavior; evaluate with quality metrics rather than label agreement |
| kwja full-corpus segmentation is slow (16 min / 300K) | Per-sentence DeBERTa inference | Batching already gives 2.3×; attention-mask work can improve further |
| LDA batch mode OOMs above 250K | Dense matrix | Falls back to online mode automatically |
| DTM uses document timestamps, not finer-grained timestamps | Dataset has no finer-grained timestamps | Results describe video-type evolution, not viewer behavior |
| fp16 gives no benefit for decoding | Apple Silicon kernel dispatch overhead | q8 recommended |

---

## 6. Future work

| Direction | Technique | Value |
|-----------|-----------|-------|
| kwja attention-mask fix | Eliminate the 11% segmentation discrepancy under batching | Quality |
| Topic hierarchy visualization | Plotly dendrogram + interactive | Report-ready |
| Comment timestamp acquisition | Fill in from the source platform API | Precise temporal analysis |
| Chinese support (CKIP-MLX) | Port CKIP-Transformers with the same architecture | More languages |
| Real-time topic prediction | HDBSCAN approximate_predict + DTM | Auto-assign new comments |
| Release artifacts | Wheel + sdist attached to every tag (0.7.0 onward) | Install without a local build |
