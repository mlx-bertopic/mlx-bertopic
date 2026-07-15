"""mlx-ctfidf — c-TF-IDF on Apple Silicon via MLX (Metal GPU).

Faithful port of BERTopic's ``ClassTfidfTransformer``
(``bertopic/vectorizers/_ctfidf.py``). The TF/IDF/reduce math matches
BERTopic exactly so the result is a true drop-in replacement.
"""

import mlx.core as mx
import numpy as np

__version__ = "0.2.0"


class CTFIDFTransformer:
    """c-TF-IDF transformer using MLX Metal GPU.

    Computes class-based TF-IDF: all documents in a cluster are treated as
    one concatenated document. Term frequencies are L1-normalized per class,
    optionally square-rooted (``reduce_frequent_words``), then multiplied by
    an IDF vector.

    The IDF, BM25 variant and reduce-frequent-words behaviour mirror
    BERTopic's ``ClassTfidfTransformer`` exactly:

      * ``df``                 = total term count per word across classes
                                  (``X.sum(axis=0)``), NOT a binary class count.
      * ``avg_nr_samples``     = average number of words per class
                                  (``int(X.sum(axis=1).mean())``).
      * idf                    = ``log((avg_nr_samples / df) + 1)``.
      * bm25 idf               = ``log(1 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))``.
      * reduce_frequent_words  = ``sqrt(TF)`` applied after L1, before IDF.
      * No L2 normalization is applied to the output (BERTopic does not).

    Args:
        reduce_frequent_words: Take the square root of the L1-normalized term
            frequencies before applying IDF. Reduces the impact of words that
            appear too frequently.
        bm25_weighting: Use the BM25-inspired IDF weighting instead of the
            standard c-TF-IDF IDF.
    """

    def __init__(self, reduce_frequent_words: bool = False, bm25_weighting: bool = False):
        self.reduce_frequent_words = reduce_frequent_words
        self.bm25_weighting = bm25_weighting
        self._idf: mx.array | None = None

    def fit(self, X) -> "CTFIDFTransformer":
        """Fit IDF values from a class-term count matrix.

        Args:
            X: ``(n_classes, vocab_size)`` term-frequency matrix (dense numpy
                array or array-like). Each row is the bag-of-words of one
                cluster/class.
        """
        X_np = np.asarray(X, dtype=np.float64)
        X_mx = mx.array(X_np.astype(np.float32))

        # Total term count per word across all classes (NOT a binary count).
        df = mx.sum(X_mx, axis=0).astype(mx.float32)

        # Average number of words per class, truncated to int (matches BERTopic).
        avg_nr_samples = float(int(X_np.sum(axis=1).mean()))

        if self.bm25_weighting:
            self._idf = mx.log(1.0 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))
        else:
            self._idf = mx.log((avg_nr_samples / df) + 1.0)

        # All-zero columns (df == 0) give inf idf in the standard formula; in
        # dense arithmetic tf(0) * idf(inf) = NaN. BERTopic's sparse path leaves
        # such columns at 0 (the inf idf never multiplies a stored non-zero), so
        # mask idf to 0 where df == 0 to match that result exactly.
        self._idf = mx.where(df > 0, self._idf, mx.array(0.0))

        mx.eval(self._idf)
        return self

    def transform(self, X) -> np.ndarray:
        """Transform a class-term count matrix to c-TF-IDF.

        Args:
            X: ``(n_classes, vocab_size)`` term-frequency matrix.

        Returns:
            ``(n_classes, vocab_size)`` dense numpy c-TF-IDF matrix. Rows are
            NOT L2-normalized (matching BERTopic).
        """
        if self._idf is None:
            raise RuntimeError("Must call fit() before transform().")

        X_mx = mx.array(np.asarray(X, dtype=np.float32))

        # TF: L1-normalize rows.
        row_sums = mx.maximum(mx.sum(X_mx, axis=1, keepdims=True), 1e-12)
        tf = X_mx / row_sums

        if self.reduce_frequent_words:
            # sqrt of the L1-normalized TF, applied before IDF (BERTopic semantics).
            tf = mx.sqrt(tf)

        # TF * IDF (idf broadcasts over rows).
        result = tf * self._idf

        mx.eval(result)
        return np.array(result)

    def fit_transform(self, X) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


__all__ = ["CTFIDFTransformer"]
