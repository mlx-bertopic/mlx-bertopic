# Changelog

## 0.5.0 (2026-07-21)

### Fixed

- **UMAP stability on large datasets**: `MlxUMAPWrapper` now includes post-hoc
  embedding stabilization (L2 norm clip + hard bound) to work around mlx-vis
  UMAP's gradient divergence on >100K high-dimensional inputs. This prevents
  HDBSCAN from collapsing all points into a single cluster.
  
  Validated on 164K×4096d (133 topics, was 21 without fix) and 100K×4096d
  (132 topics, was 83). Datasets below 50K are unaffected.

- **Gradient clip ordering** documented: the upstream mlx-vis clips gradient
  *before* alpha multiplication; the correct behavior (matching lmcinnes/umap)
  is to clip the *final displacement*. Patch details in
  `patches/mlx_vis_umap_stability.md`.

### Added

- `patches/` directory with upstream fix documentation and issue draft for
  `hanxiao/mlx-vis`.
- `benchmarks/bench_umap_e2e.py`: end-to-end UMAP + HDBSCAN benchmark script
  comparing MLX vs CPU across dataset sizes.
- `benchmarks/RESULTS.md`: benchmark results on M4 Max.

### Changed

- `pyproject.toml`: version bump to 0.5.0; requires `mlx-vis>=0.7.0`.
- `MlxUMAPWrapper.fit_transform()` now returns stabilized output by default.
  The `MAX_NORM` and `HARD_CLIP` class attributes can be adjusted if needed.

## 0.4.0 (2026-07-16)

### Fixed

- `MlxCTFIDF`: add `_idf_diag` property for BERTopic `model.save()` compatibility.

### Added

- Parity tests (`test_ctfidf_parity.py`) comparing MLX c-TF-IDF output against
  BERTopic's `ClassTfidfTransformer`.
- `MlxEmbedder`: batch size cap to prevent OOM on large document sets.

### Changed

- Version alignment: all modules pinned to upstream reference versions.
- Benchmark suite with timing breakdowns.

## 0.3.0 (2026-07-14)

- Initial public release.
- Modules: `mlx_bertopic`, `mlx_ctfidf`, `mlx_hdbscan`, `mlx_keybert`,
  `mlx_lda`, `mlx_dtm`, `mlx_outlier`.
- Full pipeline: 24.6× faster than CPU on 214K documents.
