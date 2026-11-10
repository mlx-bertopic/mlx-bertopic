# Notices — mlx-bertopic (monorepo)

## This package
Licensed under the **MIT License**, Copyright (c) 2026 Felix Lin (see `LICENSE`).

## What this is
`mlx-bertopic` is a monorepo of independent **MLX (Metal) re-implementations** of
BERTopic's components for Apple Silicon. These are **not ports** of upstream
source code; each component is reimplemented from the algorithm and
**behaviorally aligned** to its upstream reference, verified by tests.

No upstream source is vendored into the packages. The only places upstream code
appears verbatim are `patches/` (bug reports we intend to send upstream) and the
quoted excerpts in `METHODOLOGY.md`; both are attributed below.

## Components & upstream alignment

| Component | Reimplements / aligns to | Upstream license | Verified by |
|---|---|---|---|
| `mlx_ctfidf` | BERTopic `ClassTfidfTransformer` (c-TF-IDF) | MIT (BERTopic) | golden tests vs `bertopic.vectorizers.ClassTfidfTransformer` |
| `mlx_hdbscan` | HDBSCAN (Campello 2013; McInnes 2017) | **BSD-3-Clause** (hdbscan) | golden tests vs Cython `hdbscan` (incl. non-convex, ARI ≥ 0.9) |
| `mlx_keybert` | KeyBERT-inspired + MMR keyword selection | MIT (KeyBERT) | behavioral tests |
| `mlx_lda` | Latent Dirichlet Allocation (Blei 2003) | **BSD-3-Clause** (scikit-learn) | behavioral tests |
| `mlx_dtm` | LSTM-based dynamic topic modeling | — (original architecture) | smoke tests |
| `mlx_outlier` | BERTopic outlier-reduction strategies | MIT (BERTopic) | behavioral tests |
| `mlx_bertopic` | BERTopic backend integration (umbrella) | MIT (BERTopic) | integration + golden tests |

Aligned to: **BERTopic 0.17.4**, **hdbscan 0.8.x**.

## Runtime dependencies (code we do not own)

These are used through their public APIs, not reimplemented. Versions shown are
the ones this release was tested against; `pyproject.toml` holds the floors.

| Package | Tested | License | Author | Source |
|---|---|---|---|---|
| `mlx-vis` | 0.7.0 | **Apache-2.0** | Han Xiao | https://github.com/hanxiao/mlx-vis |
| `mlx-embeddings` | 0.1.0 | **GPL-3.0** | Prince Canuma | https://github.com/Blaizzy/mlx-embeddings |
| `mlx` | 0.32.0 | MIT | MLX Contributors | https://github.com/ml-explore/mlx |
| `bertopic` | 0.17.0 | MIT | Maarten P. Grootendorst | https://github.com/MaartenGr/BERTopic |
| `hdbscan` | 0.8.40 | BSD-3-Clause | Leland McInnes | https://github.com/scikit-learn-contrib/hdbscan |
| `scikit-learn` | 1.6.1 | BSD-3-Clause | The scikit-learn developers | https://scikit-learn.org |
| `numpy` | 2.3.5 | BSD-3-Clause | NumPy Developers | https://numpy.org |
| `scipy` | 1.17.0 | BSD-3-Clause | SciPy Developers | https://scipy.org |

Where each one is called:

- **`mlx-vis`** — `mlx_vis.UMAP` behind `mlx_bertopic.umap.MlxUMAPWrapper`
  (UMAP backend), and `mlx_vis.NNDescent` in `mlx_hdbscan`'s sparse path
  (`algorithm="sparse"`, n > 30K). It is a load-bearing dependency, not an
  optional convenience.
- **`mlx-embeddings`** — only inside `mlx_bertopic.embedder.MlxEmbedder`
  (`load()` / `generate()`); every other module runs without it. It is licensed
  **GPL-3.0**, unlike this package (MIT). Importing it does not relicense this
  code, but a GPL-3.0 work is present in any environment where `MlxEmbedder` is
  used or a bundle is redistributed. If that matters for your deployment, pass
  any object with an `embed(documents) -> np.ndarray` method instead of
  `MlxEmbedder`, or install `mlx-embeddings` separately and review its terms.
- **`bertopic`** — the API contract (base classes, expected model interfaces)
  and the reference for c-TF-IDF/outlier parity tests.

## Code derived from upstream (attribution)

- `patches/mlx_vis_umap_stability.md` and `patches/mlx_vis_issue_draft.md` quote
  `mlx_vis/_umap/umap.py::_sgd_step` (© Han Xiao, Apache-2.0) and propose a
  change to it. Under Apache-2.0 §4 the upstream notice is retained here, and
  the modification is stated: we propose clipping the *final displacement*
  rather than the intermediate gradient, plus a per-epoch coordinate bound.
  Nothing in those files is compiled into the package, and the change is a
  proposal — upstream has not accepted it.
- `mlx_bertopic.umap.MlxUMAPWrapper._stabilize()` (L2-norm clip + hard bound) is
  original code in this repo, written against observed divergence; it is not
  copied from `mlx-vis`.
- MLX UMAP on large high-dimensional data is the subject of mlx-vis's paper
  (arXiv:2603.04035); see therein for the algorithm, not for this workaround.

## Acknowledgements

Thanks to the people whose work this builds on:

- **Han Xiao** and **mlx-vis** — Metal UMAP, t-SNE, NNDescent and friends. The
  UMAP stage is 80% of this pipeline's runtime, and it exists because of
  mlx-vis; this repo's benchmarks are largely a measurement of their kernels.
- **Prince Canuma** and **mlx-embeddings** — embeddings on Metal with no PyTorch
  in the hot path.
- **Maarten Grootendorst** and **BERTopic** — the component decomposition,
  the interfaces every backend here is aligned to, and the golden references.
- **Leland McInnes** — `hdbscan` and the UMAP reference implementation; the
  condensed-tree/EOM semantics and the SGD behavior we test against are his.
- **The MLX team** (Apple Machine Learning Research) — the framework.
- **scikit-learn**, **NumPy**, **SciPy** developers — the reference math for
  c-TF-IDF/LDA and the numerical substrate.

## BSD 3-Clause License (applies to hdbscan- and scikit-learn-derived algorithms)

```
Copyright (c) Leland McInnes and contributors (hdbscan)
Copyright (c) 2007-2024 The scikit-learn developers (scikit-learn)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.
```

## Upstream references

- mlx-vis — https://github.com/hanxiao/mlx-vis (Apache-2.0)
- mlx-embeddings — https://github.com/Blaizzy/mlx-embeddings (GPL-3.0)
- BERTopic — https://github.com/MaartenGr/BERTopic (MIT)
- hdbscan — https://github.com/scikit-learn-contrib/hdbscan (BSD-3-Clause)
- UMAP — https://github.com/lmcinnes/umap (BSD-3-Clause)
- scikit-learn — https://scikit-learn.org (BSD-3-Clause)
- KeyBERT — https://github.com/MaartenGr/KeyBERT (MIT)
- MLX — https://github.com/ml-explore/mlx (MIT)
