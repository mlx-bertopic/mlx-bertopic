# MLX-Native Topic Modeling Pipeline：研究大綱與方法論

## 研究目標

在 Apple Silicon 上建構完整的 GPU-native 主題建模工具鏈，實現從原始文本到跨語言主題分析的全自動化 pipeline，不依賴 NVIDIA GPU 或雲端資源。

---

## 1. 問題設定

### 1.1 動機

現有大規模主題建模工具鏈（BERTopic + UMAP + HDBSCAN）的瓶頸：

| 問題 | 影響 |
|------|------|
| UMAP (numba) CPU-only | 100K+ 文檔需數十分鐘至數小時 |
| HDBSCAN (Cython) 無 GPU 加速 | 大規模資料不是效能瓶頸但精度受限 |
| 日文分詞依賴 MeCab + 外部辭典 | 環境建置複雜，NER 能力弱 |
| LDA (sklearn) 單執行緒 EM | 大規模收斂慢 |
| 無本地 Dynamic Topic Modeling | BERTopic DTM bin-size 敏感，無記憶性 |
| Apple Silicon GPU 閒置 | MPS 不支援非 PyTorch 計算 |

### 1.2 研究範圍

建構以下 5 個獨立套件 + 1 個整合 pipeline：

```
┌─────────────────────────────────────────────────────────┐
│                    mlx-bertopic                          │
│         (整合 pipeline + CLI + 報告生成)                  │
├─────────────┬──────────────┬──────────────┬─────────────┤
│  mlx-vis    │ mlx-hdbscan  │   mlx-lda    │  mlx-dtm    │
│  (UMAP)     │ (clustering) │   (topics)   │  (temporal) │
│  [第三方]    │              │              │             │
├─────────────┴──────────────┴──────────────┴─────────────┤
│                      kwja-mlx                           │
│           (日文分詞 / POS / NER / 讀み)                   │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 技術架構

### 2.1 共通設計原則

1. **Drop-in replacement** — API 與 sklearn/hdbscan 一致，一行 import 切換
2. **Auto mode** — 根據資料規模和記憶體自動選擇演算法路徑
3. **GPU-first, CPU-fallback** — 能 GPU 加速的搬上 Metal，本質序列的留 CPU
4. **pip installable** — pyproject.toml + setuptools，`pip install -e .`

### 2.2 各模組方法論

#### kwja-mlx（日文形態素解析）

| 組件 | 模型 | 方法 | 精度 |
|------|------|------|------|
| 分詞 | DeBERTa v2 + Conv | Character-level B/I tagging | F1≈1.000 |
| POS | DeBERTa v2 | First-subword pooling + Linear | 99.91% (vs fp32) |
| NER | DeBERTa v2 + CRF | Emission scores + Viterbi decoding | 99.59%, 100% recall |
| 讀み/原形 | T5 base | Seq2seq generation | Identical to PyTorch |

**量化策略：** q8 推薦（decode 最快 + 精度無損）。fp16 在 autoregressive decode 下反而慢（Apple Silicon kernel dispatch overhead）。

**批次推理：** `segment_batch(batch_size=16)` 利用 padding + batch DeBERTa forward，2.3x throughput。

#### mlx-hdbscan（密度聚類）

| 模式 | 適用規模 | 方法 | 特點 |
|------|---------|------|------|
| dense | n ≤ 30K | GPU pairwise distance + Prim MST + EOM | 精確，O(n²) memory |
| sparse | n > 30K | NNDescent KNN + sparse MRD + scipy Kruskal MST + EOM | 可擴展，O(n×k) memory |
| auto | 任意 | 根據 n 自動選擇 | 預設模式 |

**Cluster extraction：** 完整 Campello 2013 condensed tree + Excess of Mass stability-based extraction。比 Cython 版更保守（silhouette 高 33-55%）。

**與 Cython 差異來源：** MST tie-breaking 歧義（已知行為，cuML 文件有同樣說明）。

#### mlx-lda（主題建模）

| 模式 | 方法 | 記憶體 | 適用 |
|------|------|--------|------|
| batch | Batch Variational EM (Blei 2003) | O(n_docs × vocab) | < 250K docs |
| online | Online VI (Hoffman 2010) | O(batch_size × vocab) | 任意規模 |
| auto | 根據記憶體估算選擇 | 自適應 | 預設 |

**GPU 加速點：** E-step（theta update）和 M-step（beta update）皆為矩陣乘法，MLX Metal 加速 9-16x。

#### mlx-dtm（動態主題建模）

| 組件 | 方法 |
|------|------|
| 時間序列建構 | 滑窗 topic proportion (window_size, stride) |
| 平滑 | 2-layer LSTM (self-supervised: predict t+1 from 0..t) |
| 轉折點偵測 | Smoothed output 差分 > μ + kσ |
| 預測 | Autoregressive generation from final hidden state |

**優勢 vs BERTopic DTM：** LSTM hidden state 提供跨時間步記憶，不受 bin size 影響，結果穩定。

### 2.3 跨語言比較方法 (BTM)

```
Language A (N_a topics) → centroid embeddings (4096D mean)
Language B (N_b topics) → centroid embeddings (4096D mean)
         ↓
L2-normalize → cosine similarity matrix (N_a × N_b)
         ↓
Per-topic best match → 配對強度分佈
         ↓
High (>0.85) / Medium (0.7-0.85) / Low (<0.7) 分類
```

**前提：** 所有語言的 embeddings 來自同一多語言模型，天然在同一語義空間。

---

## 3. 驗證方法

### 3.1 正確性驗證

| 套件 | 驗證方式 | 結果 |
|------|---------|------|
| kwja-mlx | Wikipedia 日本 590 句 × 4 精度模式 | POS/NER/segmentation 全精度一致 |
| kwja-mlx NER | CRF vs argmax 比較 | B/I 零錯誤，recall 100% |
| mlx-hdbscan (dense) | make_blobs ARI | 0.995 |
| mlx-hdbscan (sparse) | dense vs sparse 一致性 | ARI=1.000 (blobs) |
| mlx-lda | Perplexity 收斂 | 正確收斂 (257→208) |
| mlx-dtm | Loss 收斂 | 0.31→0.23 |

### 3.2 品質比較

| 指標 | MLX | 標準 (sklearn/Cython) | 意義 |
|------|-----|----------------------|------|
| Silhouette score | 0.52-0.61 | 0.39 | Cluster 內聚度更高 |
| Intra-cluster distance | 0.35-0.38 | 0.40 | Cluster 更 tight |
| NER recall | 100% | — | CRF Viterbi 效果 |
| Topic coherence (LDA) | 語義可解釋 | char-ngram 碎片 | word-level 優勢 |

### 3.3 效能比較

| 任務 (規模) | MLX | 標準 | 加速 |
|------------|-----|------|------|
| UMAP large-scale×4096D | 143s | ∞ (killed) | 可做 vs 不可做 |
| UMAP 10K | 0.5s | 17.3s | 35x |
| HDBSCAN 100K | 1.1s | 2.4s | 2.2x |
| LDA 50K | 2.3s | 21.6s | 9.3x |
| kwja segment 500 sent | 1.55s (batch) | — | 323 sent/sec |
| DTM training 498 steps | 8.7s | — | — |
| 全量 pipeline large-scale | 18 min | 不可能 | — |

### 3.4 邊界測試

| 測試 | 結果 |
|------|------|
| 空字串 | ✅ 正確回傳空 |
| 超長文 (>512 token) | ⚠️ 需呼叫端分句 (DeBERTa 硬限制) |
| large-scale HDBSCAN sparse | ✅ 11.4s, 無 OOM |
| 300K LDA online | ✅ 80s, 無 OOM |
| q4 精度劣化 | POS 99.91%, NER 99.59% (可用) |

---

## 5. 工具鏈清單

| 套件 | Repository | 版本 | 狀態 |
|------|-----------|------|------|
| mlx-ctfidf | github.com/mlx-bertopic/mlx-bertopic | 0.2.0 | ✅ aligned to BERTopic 0.17.4 |
| mlx-hdbscan | github.com/mlx-bertopic/mlx-bertopic | 0.1.0 | ✅ aligned to hdbscan 0.8.x |
| mlx-keybert | github.com/mlx-bertopic/mlx-bertopic | 0.1.0 | ✅ KeyBERT + MMR |
| mlx-lda | github.com/mlx-bertopic/mlx-bertopic | 0.1.0 | ✅ variational EM LDA |
| mlx-dtm | github.com/mlx-bertopic/mlx-bertopic | 0.1.0 | ✅ LSTM dynamic topics |
| mlx-outlier | github.com/mlx-bertopic/mlx-bertopic | 0.1.0 | ✅ outlier reduction |
| mlx-bertopic | github.com/mlx-bertopic/mlx-bertopic | 0.3.0 | ✅ umbrella pipeline |

---

## 6. 限制與已知問題

| 限制 | 原因 | 緩解 |
|------|------|------|
| Apple Silicon only | MLX = Metal only | 設計目標如此 |
| sparse HDBSCAN 與 Cython 結果不同 | MST tie-breaking 歧義 | 已知行為，用品質指標而非 label 一致性評估 |
| kwja 全量分詞慢 (16 min / 300K) | DeBERTa 逐句推理 | batch 已加速 2.3x，可再優化 attention mask |
| LDA batch mode OOM > 250K | Dense matrix | 自動切 online mode |
| DTM 用 document timestamps 而非 finer-grained timestamps | 資料集無 finer-grained timestamp | 結果是影片類型演化，非觀眾行為演化 |
| fp16 在 decode 場景無優勢 | Apple Silicon kernel dispatch overhead | 推薦 q8 |

---

## 7. 未來方向

| 方向 | 技術 | 價值 |
|------|------|------|
| kwja attention mask 修正 | 消除 batch 11% 分詞差異 | 品質提升 |
| Topic hierarchy 視覺化 | Plotly dendrogram + interactive | 報告可用 |
| Comment timestamp 取得 | the source platform API 補全 | 精確時序分析 |
| 中文支援 (CKIP-MLX) | 同架構移植 CKIP-Transformers | 擴展語言 |
| 即時 topic 預測 | HDBSCAN approximate_predict + DTM | 新留言自動歸類 |
| PyPI 發佈 | 正式 package release | 社群使用 |
