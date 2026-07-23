#!/usr/bin/env python3
"""Visualization suite for mlx-bertopic results.

Generates all standard visualizations from BERTopic + NeuralDTM outputs.
Requires: plotly, pandas, numpy, sklearn (for heatmap cosine similarity).

Usage:
    from mlx_bertopic.viz import generate_all
    generate_all(
        topics_assigned="topics_assigned.parquet",
        topic_keywords="topic_keywords.json",
        umap_2d="umap_2d.npy",
        output_dir="viz/",
        neural_dtm=ndtm,  # optional: fitted NeuralDTM instance
    )
"""

import json
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd


def generate_all(
    topics_assigned: str,
    topic_keywords: str,
    umap_2d: Optional[str] = None,
    output_dir: str = "viz",
    neural_dtm=None,
    title_prefix: str = "",
    top_n: int = 25,
    sample_size: int = 20000,
):
    """Generate all standard visualizations.

    Parameters
    ----------
    topics_assigned : str
        Path to parquet with 'topic', 'post_date', 'docs_filtered' columns.
    topic_keywords : str
        Path to JSON: {topic_id: [(word, score), ...]}.
    umap_2d : str, optional
        Path to .npy with 2D UMAP coordinates.
    output_dir : str
        Directory to save HTML files.
    neural_dtm : NeuralDTM, optional
        Fitted NeuralDTM instance for drift visualization.
    title_prefix : str
        Prefix for chart titles.
    top_n : int
        Number of top topics to show.
    sample_size : int
        Max points for document map.
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.metrics.pairwise import cosine_similarity

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(topics_assigned)
    with open(topic_keywords) as f:
        kw = json.load(f)

    labels = df["topic"].values
    topic_sizes = Counter(labels[labels >= 0])
    n_topics = len(topic_sizes)

    def topic_label(tid, n_words=3):
        words = kw.get(str(tid), [])[:n_words]
        return f"T{tid}: {' '.join(w for w, _ in words)}"

    print(f"Generating visualizations ({n_topics} topics, {len(df):,} docs)...")

    # ── 1. Barchart ───────────────────────────────────
    top_topics = topic_sizes.most_common(top_n)
    fig = go.Figure()
    for tid, count in reversed(top_topics):
        words = kw.get(str(tid), [])[:4]
        label = f"T{tid}: {' | '.join(w for w, _ in words)}"
        fig.add_trace(go.Bar(x=[count], y=[label], orientation="h", showlegend=False))
    fig.update_layout(title=f"{title_prefix}Top {top_n} Topics", height=max(600, top_n * 30),
                      margin=dict(l=350))
    fig.write_html(str(out / "barchart.html"))
    print(f"  ✓ barchart.html")

    # ── 2. Document Map ───────────────────────────────
    if umap_2d:
        coords = np.load(umap_2d)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(df), size=min(sample_size, len(df)), replace=False)
        fig = px.scatter(x=coords[idx, 0], y=coords[idx, 1],
                         color=[str(labels[i]) for i in idx],
                         title=f"{title_prefix}Document Map ({min(sample_size, len(df)):,} sample)",
                         width=1200, height=800, opacity=0.4)
        fig.update_traces(marker_size=2)
        fig.update_layout(showlegend=False)
        fig.write_html(str(out / "documents.html"))
        print(f"  ✓ documents.html")

    # ── 3. Heatmap ────────────────────────────────────
    top20 = [tid for tid, _ in topic_sizes.most_common(20)]
    all_words = set()
    for tid in top20:
        for w, _ in kw.get(str(tid), [])[:20]:
            all_words.add(w)
    word_list = sorted(all_words)
    word_idx = {w: i for i, w in enumerate(word_list)}
    mat = np.zeros((len(top20), len(word_list)))
    for i, tid in enumerate(top20):
        for w, score in kw.get(str(tid), [])[:20]:
            if w in word_idx:
                mat[i, word_idx[w]] = score
    sim = cosine_similarity(mat)
    t_labels = [topic_label(tid, 2) for tid in top20]
    fig = px.imshow(sim, x=t_labels, y=t_labels, title=f"{title_prefix}Topic Similarity (Top 20)",
                    width=900, height=800, color_continuous_scale="Blues")
    fig.update_layout(margin=dict(l=200, b=200))
    fig.write_html(str(out / "heatmap.html"))
    print(f"  ✓ heatmap.html")

    # ── 4. Hierarchy ──────────────────────────────────
    top30 = topic_sizes.most_common(30)
    hdata = [{"label": f"T{tid}: {' | '.join(w for w,_ in kw.get(str(tid),[])[:4])}", "count": c}
             for tid, c in top30]
    fig = px.bar(pd.DataFrame(hdata), x="count", y="label", orientation="h",
                 title=f"{title_prefix}Topic Hierarchy (Top 30)", height=900)
    fig.update_layout(margin=dict(l=350), yaxis={"categoryorder": "total ascending"})
    fig.write_html(str(out / "hierarchy.html"))
    print(f"  ✓ hierarchy.html")

    # ── 5. Topics Over Time ───────────────────────────
    if "post_date" in df.columns:
        df["year"] = pd.to_datetime(df["post_date"]).dt.year
        years = sorted(df["year"].unique())
        top10 = [tid for tid, _ in topic_sizes.most_common(10)]

        # Stacked area
        fig = go.Figure()
        for tid in top10:
            label = topic_label(tid)
            counts = [((df["year"] == y) & (df["topic"] == tid)).sum() for y in years]
            fig.add_trace(go.Scatter(x=years, y=counts, name=label, mode="lines", stackgroup="one"))
        fig.update_layout(title=f"{title_prefix}Topics Over Time (Top 10)", width=1200, height=600)
        fig.write_html(str(out / "topics_over_time.html"))
        print(f"  ✓ topics_over_time.html")

        # Normalized trends
        year_totals = df.groupby("year").size()
        fig = go.Figure()
        for tid in top10:
            label = topic_label(tid)
            props = [((df["year"] == y) & (df["topic"] == tid)).sum() / year_totals.get(y, 1) * 100
                     for y in years]
            fig.add_trace(go.Scatter(x=years, y=props, name=label, mode="lines+markers"))
        fig.update_layout(title=f"{title_prefix}Topic Proportion (Top 10, %)",
                          xaxis_title="Year", yaxis_title="%", width=1200, height=600)
        fig.write_html(str(out / "topic_trends.html"))
        print(f"  ✓ topic_trends.html")

    # ── 6. Sunburst ───────────────────────────────────
    # Auto-categorize by keyword matching
    CATEGORY_KEYWORDS = {
        "Social Issues": ["勞", "障", "街友", "移工", "租屋", "詐騙", "毒品", "罷工", "霸凌", "同志", "更生", "性別"],
        "Environment": ["能源", "海洋", "生態", "垃圾", "回收", "農業", "漁", "蝴蝶", "猛禽", "外來種", "濕地", "廢水", "空汙"],
        "Health": ["醫", "疫", "健保", "口罩", "糖尿", "肥胖", "憂鬱", "安寧", "視力", "皮膚", "疫苗"],
        "Culture": ["原住民", "陣頭", "工藝", "書店", "漫畫", "韓國", "音樂", "藝術", "攝影", "舞蹈", "信仰"],
        "Education": ["世新", "社團", "學生會", "畢展", "教育", "學校", "偏鄉"],
    }
    sun_labels = [title_prefix or "Topics"]
    sun_parents = [""]
    sun_values = [0]
    for cat, cat_kws in CATEGORY_KEYWORDS.items():
        cat_topics = [tid for tid in topic_sizes
                      if any(k in str(kw.get(str(tid), [])) for k in cat_kws)]
        cat_total = sum(topic_sizes.get(t, 0) for t in cat_topics)
        if cat_total == 0:
            continue
        sun_labels.append(cat)
        sun_parents.append(title_prefix or "Topics")
        sun_values.append(cat_total)
        for tid in sorted(cat_topics, key=lambda t: -topic_sizes.get(t, 0))[:6]:
            words = kw.get(str(tid), [])[:3]
            sun_labels.append(" ".join(w for w, _ in words))
            sun_parents.append(cat)
            sun_values.append(topic_sizes.get(tid, 0))
    fig = go.Figure(go.Sunburst(labels=sun_labels, parents=sun_parents, values=sun_values,
                                 branchvalues="total", maxdepth=2))
    fig.update_layout(title=f"{title_prefix}Topic Sunburst", width=800, height=800)
    fig.write_html(str(out / "sunburst.html"))
    print(f"  ✓ sunburst.html")

    # ── 7. NeuralDTM Drift ────────────────────────────
    if neural_dtm is not None:
        top_drift = neural_dtm.top_drifting_topics(5)
        fig = go.Figure()
        for tid, total_d in top_drift:
            label = f"{topic_label(tid)} (drift={total_d:.4f})"
            drift_vals = neural_dtm.get_drift(tid)[tid]
            bin_labels = neural_dtm._bin_labels
            x_labels = [bl.strftime("%Y-%m") for bl in bin_labels[:-1]]
            fig.add_trace(go.Scatter(x=x_labels, y=drift_vals, name=label, mode="lines+markers"))
        fig.update_layout(title=f"{title_prefix}NeuralDTM: Semantic Drift",
                          xaxis_title="Time", yaxis_title="Cosine Drift", width=1200, height=500)
        fig.write_html(str(out / "neural_dtm_drift.html"))
        print(f"  ✓ neural_dtm_drift.html")

    print(f"\nAll visualizations saved to {out}/")


# Alias for lazy import from mlx_bertopic
generate_all_viz = generate_all
