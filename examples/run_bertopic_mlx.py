#!/usr/bin/env python3.11
"""
End-to-end BERTopic on Apple Silicon — MLX pipeline (v0.3.0).

Runs the full BERTopic stack:
  - Embedding: mlx-embeddings (Metal GPU)
  - UMAP: mlx-vis (Metal GPU)
  - HDBSCAN: hdbscan (Cython CPU — fastest for small data)
  - c-TF-IDF: MlxCTFIDF (Metal GPU, auto-fallback for large matrices)
  - Outlier reduction: reduce_outliers_embeddings (cosine similarity)

Then compares against the default BERTopic config.
"""
import sys, time, traceback

import numpy as np
from bertopic import BERTopic
from mlx_bertopic import MlxEmbedder, MlxUMAPWrapper, MlxCTFIDF, reduce_outliers_embeddings

# ── Sample docs ──
docs = [
    "Artificial intelligence is transforming healthcare diagnostics",
    "Machine learning models predict patient outcomes",
    "Deep learning for medical image analysis",
    "Natural language processing in clinical trials",
    "Neural networks for drug discovery",
    "Gene editing CRISPR trials show promise",
    "mRNA vaccine technology advances rapidly",
    "Telemedicine adoption grows post-pandemic",
    "Wearable health monitors track vital signs",
    "Climate change impacts on coastal cities",
    "Rising sea levels threaten infrastructure",
    "Carbon emissions reach record highs globally",
    "Renewable energy adoption accelerates worldwide",
    "Solar panel efficiency breaks new records",
    "Electric vehicle sales surge in global markets",
    "Battery technology enables longer range EVs",
    "Autonomous driving safety improvements",
    "Self-driving cars navigate complex urban environments",
    "Quantum computing achieves new milestone",
    "Quantum supremacy demonstrated in optimization",
    "Cryptocurrency market volatility continues",
    "Blockchain technology beyond digital currency",
    "Space exploration missions to Mars planned",
    "Satellite networks provide global internet coverage",
    "5G networks roll out across major cities",
    "Edge computing reduces latency for IoT devices",
    "Cybersecurity threats evolve with AI",
    "Ransomware attacks target critical infrastructure",
    "Cloud computing dominates enterprise IT",
    "Microservices architecture improves scalability",
] * 5  # 150 docs — enough for clustering, not too slow

print(f"Documents: {len(docs)}")

# ═══ MLX BERTopic ══════════════════════════════════════
print("\n" + "="*55)
print("  BERTopic + MLX (Metal GPU)")
print("="*55)

try:
    import hdbscan

    t0 = time.time()

    embedding_model = MlxEmbedder("mlx-community/bge-small-en-v1.5-bf16")
    umap_model = MlxUMAPWrapper(n_components=5, n_neighbors=15, n_epochs=200)
    cluster_model = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)

    topic_model_mlx = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        ctfidf_model=MlxCTFIDF(),
        # representation_model=MlxKeyBERT(top_n_words=10),  # requires word embeddings
        verbose=True,
    )

    topics_mlx, probs_mlx = topic_model_mlx.fit_transform(docs)
    t_mlx = time.time() - t0

    n_topics = len(set(topics_mlx)) - (1 if -1 in topics_mlx else 0)
    print(f"\n  ✅ {n_topics} topics in {t_mlx:.1f}s")

    # Outlier reduction
    embeddings = embedding_model.embed(docs)
    n_outliers_before = (topics_mlx == -1).sum()
    new_topics = reduce_outliers_embeddings(np.array(topics_mlx), embeddings, threshold=0.3)
    n_outliers_after = (new_topics == -1).sum()
    print(f"  Outliers: {n_outliers_before} → {n_outliers_after}")

    print(f"\n  Topic summary:")
    freq = topic_model_mlx.get_topic_info()
    print(freq.head(15).to_string(index=False))

except Exception as e:
    print(f"\n  ❌ {e}")
    traceback.print_exc()
    t_mlx = None
    topic_model_mlx = None

# ═══ Standard BERTopic (comparison) ═══════════════════
print("\n\n" + "="*55)
print("  BERTopic + Standard (CPU)")
print("="*55)

try:
    from sentence_transformers import SentenceTransformer
    from bertopic.backend import SentenceTransformerBackend
    import umap as umap_learn

    t0 = time.time()

    st_model = SentenceTransformerBackend("all-MiniLM-L6-v2")
    umap_std = umap_learn.UMAP(n_components=5, n_neighbors=15, random_state=42)
    cluster_std = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)

    topic_model_std = BERTopic(
        embedding_model=st_model,
        umap_model=umap_std,
        hdbscan_model=cluster_std,
        verbose=True,
    )

    topics_std, probs_std = topic_model_std.fit_transform(docs)
    t_std = time.time() - t0

    n_topics_std = len(set(topics_std)) - (1 if -1 in topics_std else 0)
    print(f"\n  ✅ {n_topics_std} topics in {t_std:.1f}s")

except Exception as e:
    print(f"\n  ❌ {e}")
    traceback.print_exc()
    t_std = None

# ═══ Summary ═══════════════════════════════════════════
if t_mlx and t_std:
    print(f"\n{'='*55}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*55}")
    print(f"  MLX (Metal):   {t_mlx:.1f}s  ({n_topics} topics)")
    print(f"  Standard CPU:  {t_std:.1f}s  ({n_topics_std} topics)")
    print(f"  Speedup:       {t_std/t_mlx:.1f}x")
    print(f"{'='*55}")
