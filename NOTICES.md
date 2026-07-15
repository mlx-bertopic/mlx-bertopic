# Notices — mlx-bertopic (monorepo)

## This package
Licensed under the **MIT License**, Copyright (c) 2026 Felix Lin (see `LICENSE`).

## What this is
`mlx-bertopic` is a monorepo of independent **MLX (Metal) re-implementations** of
BERTopic's components for Apple Silicon. These are **not ports** of upstream
source code; each component is reimplemented from the algorithm and
**behaviorally aligned** to its upstream reference, verified by tests.

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
- BERTopic — https://github.com/MaartenGr/BERTopic (MIT)
- hdbscan — https://github.com/scikit-learn-contrib/hdbscan (BSD-3-Clause)
- scikit-learn — https://scikit-learn.org (BSD-3-Clause)
- KeyBERT — https://github.com/MaartGr/KeyBERT (MIT)
- MLX — https://github.com/ml-explore/mlx (MIT)
