#!/usr/bin/env python3.11
"""Benchmark BM25: scipy sparse vs MLX dense on Metal GPU."""
import numpy as np
import time
from scipy import sparse


def bm25_scipy(tf, doc_lens, avgdl, idf, k1=1.5, b=0.75):
    """BM25 scoring using scipy sparse."""
    len_norm = k1 * (1 - b + b * doc_lens / avgdl)

    # Work on a copy
    result = tf.copy().astype(np.float32)

    # BM25 numerator/denominator per nonzero element
    # score[i,j] = (tf[i,j] * (k1+1)) / (tf[i,j] + len_norm[i]) * idf[j]
    rows = result.nonzero()[0]
    cols = result.nonzero()[1]

    tf_vals = result.data
    denom = tf_vals + len_norm[rows]
    result.data = (tf_vals * (k1 + 1)) / denom * idf[cols]

    return result


def bm25_mlx(tf_dense, doc_lens, avgdl, idf, k1=1.5, b=0.75):
    """BM25 scoring using MLX dense on Metal GPU.
    tf_dense: (n_docs, vocab) dense float32 array.
    """
    import mlx.core as mx

    tf_mx = mx.array(tf_dense)
    doc_lens_mx = mx.array(doc_lens.astype(np.float32))
    idf_mx = mx.array(idf.astype(np.float32))

    # len_norm per doc: (n_docs, 1)
    len_norm = k1 * (1 - b + b * doc_lens_mx / avgdl)
    len_norm = len_norm[:, None]  # broadcast over vocab dimension

    # BM25: (tf * (k1+1)) / (tf + len_norm) * idf
    numerator = tf_mx * (k1 + 1.0)
    denominator = tf_mx + len_norm
    result = (numerator / denominator) * id_mx  # broadcast idf over rows

    mx.eval(result)
    return result


def bm25_mlx_optimized(tf_dense, doc_lens, avgdl, idf, k1=1.5, b=0.75):
    """BM25 optimized: avoid materializing full n×v dense matrix.
    Instead compute scores only for nonzero entries.
    Input is sparse, converted to dense coordinate arrays for GPU.
    """
    import mlx.core as mx

    # Get nonzero indices from the sparse matrix
    tf_csr = sparse.csr_matrix(tf_dense) if not sparse.issparse(tf_dense) else tf_dense
    tf_coo = tf_csr.tocoo()
    rows = mx.array(tf_coo.row.astype(np.int32))
    cols = mx.array(tf_coo.col.astype(np.int32))
    tf_vals = mx.array(tf_coo.data.astype(np.float32))
    doc_lens_mx = mx.array(doc_lens.astype(np.float32))
    idf_mx = mx.array(idf.astype(np.float32))

    # Gather: len_norm[rows], idf[cols]
    len_norm_vals = k1 * (1 - b + b * doc_lens_mx / avgdl)
    len_gathered = len_norm_vals[rows]  # (nnz,)
    idf_gathered = idf_mx[cols]        # (nnz,)

    # BM25 per nonzero element
    result = (tf_vals * (k1 + 1.0)) / (tf_vals + len_gathered) * idf_gathered

    mx.eval(result)
    return result


if __name__ == "__main__":
    import mlx.core as mx
    mx.eval(mx.array([1.0]))  # warmup

    print("BM25 Benchmark: scipy sparse vs MLX dense vs MLX gather\n")
    print(f"{'n_docs':>8} {'vocab':>8} {'nnz':>10} {'scipy':>10} {'MLX dense':>10} {'MLX gather':>11} {'dense mem':>10}")
    print("-" * 80)

    for n_docs, vocab, density in [
        (1000, 10000, 0.01),
        (5000, 20000, 0.005),
        (10000, 30000, 0.003),
        (50000, 50000, 0.002),
        (100000, 100000, 0.001),
        (500000, 200000, 0.0005),
    ]:
        # Generate sparse TF matrix
        tf = sparse.random(n_docs, vocab, density=density, format='csr', dtype=np.float32)
        tf.data = np.ceil(tf.data * 10)
        tf.eliminate_zeros()

        doc_lens = np.array(tf.sum(axis=1)).flatten()
        avgdl = float(doc_lens.mean())
        df = np.array(tf.sum(axis=0)).flatten()
        df = np.maximum(df, 0.5)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1).astype(np.float32)

        nnz = tf.nnz
        dense_mem = n_docs * vocab * 4 / 1e9  # GB for float32 dense

        # scipy sparse BM25
        t0 = time.time()
        bm25_scipy(tf, doc_lens, avgdl, idf)
        t_scipy = time.time() - t0

        # MLX dense (only if dense matrix fits in memory)
        t_mlx_dense = None
        if dense_mem < 40:  # skip if > 40GB
            tf_dense = tf.toarray().astype(np.float32)
            # Fix: idf broadcast
            import mlx.core as mx
            tf_mx = mx.array(tf_dense)
            doc_lens_mx = mx.array(doc_lens.astype(np.float32))
            idf_mx = mx.array(idf)
            len_norm = (k1_val := 1.5) * (1 - (b_val := 0.75) + b_val * doc_lens_mx / avgdl)

            t0 = time.time()
            len_norm_b = len_norm[:, None]
            num = tf_mx * 2.5
            den = tf_mx + len_norm_b
            res = num / den * idf_mx
            mx.eval(res)
            t_mlx_dense = time.time() - t0
            del tf_dense, tf_mx

        # MLX gather (sparse → coordinate arrays → GPU gather)
        t0 = time.time()
        bm25_mlx_optimized(tf, doc_lens, avgdl, idf)
        t_mlx_gather = time.time() - t0

        scipy_str = f"{t_scipy*1000:.1f}ms"
        mlx_d_str = f"{t_mlx_dense*1000:.1f}ms" if t_mlx_dense else "skip (>40GB)"
        mlx_g_str = f"{t_mlx_gather*1000:.1f}ms"

        print(f"{n_docs:>8} {vocab:>8} {nnz:>10} {scipy_str:>10} {mlx_d_str:>10} {mlx_g_str:>11} {dense_mem:>8.1f}GB")
