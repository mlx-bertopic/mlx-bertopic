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

    # ── 8. Word Clouds (HTML/SVG with web fonts) ──────
    _generate_word_clouds(kw, topic_sizes, out, title_prefix, top_n=12)

    # ── 9. Intertopic Distance Map ────────────────────
    _generate_intertopic_map(kw, topic_sizes, umap_2d, labels, out, title_prefix)

    # ── 10. Topic Centroid Trajectory (NeuralDTM) ─────
    if neural_dtm is not None and umap_2d:
        _generate_centroid_trajectory(neural_dtm, kw, out, title_prefix)

    # ── 11. Temporal Heatmap ──────────────────────────
    if "post_date" in df.columns:
        _generate_temporal_heatmap(df, labels, kw, topic_sizes, out, title_prefix)

    # ── 12. Topic Network Graph ───────────────────────
    _generate_topic_network(kw, topic_sizes, out, title_prefix)

    # ── 13. Representative Documents ──────────────────
    _generate_representative_docs(df, labels, kw, topic_sizes, out, title_prefix)

    print(f"\nAll visualizations saved to {out}/")


def _generate_word_clouds(kw, topic_sizes, out, title_prefix, top_n=12):
    """Generate word cloud HTML using inline SVG with Google Fonts."""
    import random

    # Google Fonts CSS for CJK
    font_css = '<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700&display=swap" rel="stylesheet">'

    top_topics = [tid for tid, _ in topic_sizes.most_common(top_n)]

    html_parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
{font_css}
<style>
body {{ font-family: 'Noto Sans TC', sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
.cloud-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
.cloud-box {{ background: #16213e; border-radius: 12px; padding: 20px; min-height: 200px; }}
.cloud-box h3 {{ margin: 0 0 10px; font-size: 14px; color: #a8d8ea; }}
.word {{ display: inline-block; margin: 3px 5px; transition: transform 0.2s; }}
.word:hover {{ transform: scale(1.2); }}
</style></head><body>
<h1>{title_prefix}Topic Word Clouds (Top {top_n})</h1>
<div class="cloud-grid">"""]

    colors = ["#ff6b6b", "#feca57", "#48dbfb", "#ff9ff3", "#54a0ff", "#5f27cd",
              "#01a3a4", "#f368e0", "#ee5a24", "#009432", "#0652DD", "#9980FA"]

    for i, tid in enumerate(top_topics):
        words = kw.get(str(tid), [])[:15]
        if not words:
            continue
        max_score = max(s for _, s in words) if words else 1
        color = colors[i % len(colors)]

        html_parts.append(f'<div class="cloud-box"><h3>T{tid} ({topic_sizes[tid]:,} sentences)</h3>')
        for w, score in words:
            size = 14 + int((score / max_score) * 28)
            opacity = 0.5 + (score / max_score) * 0.5
            html_parts.append(
                f'<span class="word" style="font-size:{size}px; color:{color}; opacity:{opacity}; '
                f'font-weight:{700 if score/max_score > 0.5 else 400}">{w}</span>'
            )
        html_parts.append('</div>')

    html_parts.append('</div></body></html>')
    (out / "word_clouds.html").write_text("\n".join(html_parts), encoding="utf-8")
    print(f"  ✓ word_clouds.html")


def _generate_intertopic_map(kw, topic_sizes, umap_2d_path, labels, out, title_prefix):
    """Intertopic distance map: bubble chart with topic positions and sizes."""
    import plotly.graph_objects as go

    top30 = [tid for tid, _ in topic_sizes.most_common(30)]

    if umap_2d_path:
        coords = np.load(umap_2d_path)
        # Compute topic centroids in 2D space
        topic_x, topic_y, topic_size, topic_text = [], [], [], []
        for tid in top30:
            mask = labels == tid
            if mask.sum() == 0:
                continue
            cx = coords[mask, 0].mean()
            cy = coords[mask, 1].mean()
            topic_x.append(cx)
            topic_y.append(cy)
            topic_size.append(topic_sizes[tid])
            words = kw.get(str(tid), [])[:3]
            topic_text.append(f"T{tid}: {' '.join(w for w,_ in words)}<br>({topic_sizes[tid]:,})")

        # Scale bubble size
        max_size = max(topic_size)
        sizes = [20 + (s / max_size) * 60 for s in topic_size]

        fig = go.Figure(go.Scatter(
            x=topic_x, y=topic_y, mode="markers+text",
            marker=dict(size=sizes, opacity=0.6, color=list(range(len(top30))),
                       colorscale="Viridis", showscale=False),
            text=[f"T{tid}" for tid in top30[:len(topic_x)]],
            textposition="middle center",
            textfont=dict(size=9),
            hovertext=topic_text,
            hoverinfo="text",
        ))
        fig.update_layout(
            title=f"{title_prefix}Intertopic Distance Map (Top 30)",
            width=1000, height=800,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        fig.write_html(str(out / "intertopic_map.html"))
        print(f"  ✓ intertopic_map.html")


def _generate_centroid_trajectory(neural_dtm, kw, out, title_prefix):
    """2D trajectory of topic centroids over time (from NeuralDTM)."""
    import plotly.graph_objects as go

    top5 = neural_dtm.top_drifting_topics(5)
    if not top5:
        return

    # We need 2D projections of centroids — use PCA on all centroids
    all_centroids = []
    topic_ids = []
    for tid, _ in top5:
        c = neural_dtm.get_centroids(tid)
        all_centroids.append(c)
        topic_ids.append(tid)

    stacked = np.vstack(all_centroids)  # (5 * n_bins, embed_dim)
    # PCA to 2D
    mean = stacked.mean(axis=0)
    centered = stacked - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj_2d = centered @ Vt[:2].T  # (5*n_bins, 2)

    n_bins = neural_dtm.n_bins
    bin_labels = neural_dtm._bin_labels

    fig = go.Figure()
    colors = ["#ff6b6b", "#48dbfb", "#feca57", "#ff9ff3", "#54a0ff"]

    for i, (tid, _) in enumerate(top5):
        start = i * n_bins
        end = start + n_bins
        x = proj_2d[start:end, 0]
        y = proj_2d[start:end, 1]
        words = kw.get(str(tid), [])[:2]
        label = f"T{tid}: {' '.join(w for w,_ in words)}"

        # Line trajectory
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=label,
                                 line=dict(color=colors[i], width=2),
                                 marker=dict(size=4)))
        # Start and end markers
        fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers",
                                 marker=dict(size=12, color=colors[i], symbol="circle"),
                                 name=f"{label} (start)", showlegend=False))
        fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers",
                                 marker=dict(size=12, color=colors[i], symbol="diamond"),
                                 name=f"{label} (end)", showlegend=False))

    fig.update_layout(
        title=f"{title_prefix}Topic Centroid Trajectories (PCA 2D, NeuralDTM)",
        xaxis_title="PC1", yaxis_title="PC2", width=900, height=700,
        legend=dict(font=dict(size=10)),
    )
    fig.write_html(str(out / "centroid_trajectory.html"))
    print(f"  ✓ centroid_trajectory.html")


def _generate_temporal_heatmap(df, labels, kw, topic_sizes, out, title_prefix):
    """Heatmap: rows=topics, cols=years, color=sentence count."""
    import plotly.express as px

    df["year"] = pd.to_datetime(df["post_date"]).dt.year
    years = sorted(df["year"].unique())
    top20 = [tid for tid, _ in topic_sizes.most_common(20)]

    matrix = np.zeros((len(top20), len(years)))
    for i, tid in enumerate(top20):
        for j, y in enumerate(years):
            matrix[i, j] = ((df["year"] == y) & (labels == tid)).sum()

    y_labels = [f"T{tid}: {' '.join(w for w,_ in kw.get(str(tid),[])[:2])}" for tid in top20]

    fig = px.imshow(matrix, x=[str(y) for y in years], y=y_labels,
                    title=f"{title_prefix}Topic × Year Heatmap (Top 20)",
                    color_continuous_scale="YlOrRd", width=1000, height=700,
                    aspect="auto")
    fig.update_layout(margin=dict(l=250))
    fig.write_html(str(out / "temporal_heatmap.html"))
    print(f"  ✓ temporal_heatmap.html")


def _generate_topic_network(kw, topic_sizes, out, title_prefix):
    """Force-directed network graph based on keyword overlap."""
    import plotly.graph_objects as go
    from sklearn.metrics.pairwise import cosine_similarity

    top30 = [tid for tid, _ in topic_sizes.most_common(30)]

    # Build word vectors
    all_words = set()
    for tid in top30:
        for w, _ in kw.get(str(tid), [])[:15]:
            all_words.add(w)
    word_list = sorted(all_words)
    word_idx = {w: i for i, w in enumerate(word_list)}

    mat = np.zeros((len(top30), len(word_list)))
    for i, tid in enumerate(top30):
        for w, score in kw.get(str(tid), [])[:15]:
            if w in word_idx:
                mat[i, word_idx[w]] = score

    sim = cosine_similarity(mat)

    # Simple force-directed layout (spring embedding approximation)
    np.random.seed(42)
    pos = np.random.randn(len(top30), 2) * 2
    for _ in range(100):
        for i in range(len(top30)):
            for j in range(i + 1, len(top30)):
                dx = pos[i] - pos[j]
                dist = np.sqrt((dx ** 2).sum()) + 0.01
                # Repulsion
                repulsion = dx / (dist ** 2) * 0.1
                pos[i] += repulsion
                pos[j] -= repulsion
                # Attraction (if similar)
                if sim[i, j] > 0.1:
                    attraction = -dx * sim[i, j] * 0.05
                    pos[i] += attraction
                    pos[j] -= attraction

    # Draw edges (only strong connections)
    edge_x, edge_y = [], []
    for i in range(len(top30)):
        for j in range(i + 1, len(top30)):
            if sim[i, j] > 0.15:
                edge_x.extend([pos[i, 0], pos[j, 0], None])
                edge_y.extend([pos[i, 1], pos[j, 1], None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.5, color="#888"), hoverinfo="none"))

    sizes = [15 + (topic_sizes[tid] / max(topic_sizes.values())) * 40 for tid in top30]
    texts = [f"T{tid}: {' '.join(w for w,_ in kw.get(str(tid),[])[:2])}" for tid in top30]
    hover = [f"T{tid}: {' | '.join(w for w,_ in kw.get(str(tid),[])[:4])}<br>({topic_sizes[tid]:,})" for tid in top30]

    fig.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1], mode="markers+text",
        marker=dict(size=sizes, color=list(range(len(top30))), colorscale="Turbo", opacity=0.8),
        text=texts, textposition="top center", textfont=dict(size=8),
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(
        title=f"{title_prefix}Topic Network (Top 30, edges = similarity > 0.15)",
        width=1000, height=800, showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    fig.write_html(str(out / "topic_network.html"))
    print(f"  ✓ topic_network.html")


def _generate_representative_docs(df, labels, kw, topic_sizes, out, title_prefix):
    """HTML page with representative documents per topic."""
    font_css = '<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700&display=swap" rel="stylesheet">'

    top20 = [tid for tid, _ in topic_sizes.most_common(20)]

    html_parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
{font_css}
<style>
body {{ font-family: 'Noto Sans TC', sans-serif; margin: 20px; background: #f5f5f5; color: #333; max-width: 1000px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #2c3e50; }}
.topic-section {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.topic-header {{ color: #2980b9; margin: 0 0 10px; font-size: 18px; }}
.topic-keywords {{ color: #7f8c8d; font-size: 13px; margin-bottom: 10px; }}
.doc {{ padding: 8px 12px; margin: 5px 0; background: #ecf0f1; border-radius: 4px; font-size: 14px; line-height: 1.6; }}
.doc-meta {{ font-size: 11px; color: #95a5a6; }}
</style></head><body>
<h1>{title_prefix}Representative Documents (Top 20 Topics)</h1>"""]

    sent_col = "sentence" if "sentence" in df.columns else "docs_filtered"

    for tid in top20:
        words = kw.get(str(tid), [])[:6]
        keywords_str = " · ".join(w for w, _ in words)
        topic_docs = df[labels == tid]

        # Pick representative docs: longest sentences (more informative)
        if sent_col in topic_docs.columns:
            topic_docs_sorted = topic_docs.assign(
                _len=topic_docs[sent_col].str.len()
            ).nlargest(5, "_len")
            rep = topic_docs_sorted
        else:
            rep = topic_docs.head(5)

        html_parts.append(f'<div class="topic-section">')
        html_parts.append(f'<h2 class="topic-header">Topic {tid} ({topic_sizes[tid]:,} sentences)</h2>')
        html_parts.append(f'<div class="topic-keywords">{keywords_str}</div>')

        for _, row in rep.iterrows():
            text = row.get(sent_col, "")[:200]
            date = str(row.get("post_date", ""))[:10]
            html_parts.append(f'<div class="doc">{text}</div>')
            if date:
                html_parts.append(f'<div class="doc-meta">{date}</div>')

        html_parts.append('</div>')

    html_parts.append('</body></html>')
    (out / "representative_docs.html").write_text("\n".join(html_parts), encoding="utf-8")
    print(f"  ✓ representative_docs.html")


# Alias for lazy import from mlx_bertopic
generate_all_viz = generate_all
