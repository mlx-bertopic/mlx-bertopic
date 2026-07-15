"""Integration tests — full BERTopic pipeline with MLX components.

Tests the end-to-end flow with pre-computed random embeddings (no model download).
"""
import numpy as np
import pytest
import scipy.sparse as sp
from bertopic import BERTopic
from hdbscan import HDBSCAN

from mlx_bertopic import MlxCTFIDF, MlxUMAPWrapper, reduce_outliers_embeddings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 5 topic clusters × 10 documents each = 50 documents
DOCS = (
    # Cluster 0: sports
    [
        "The team won the championship game last night",
        "Soccer players trained hard for the tournament",
        "The basketball coach designed a new play",
        "Athletes prepare for the Olympic games",
        "The football match ended in a draw",
        "Tennis rankings were updated after the open",
        "Swimmers broke the world record today",
        "The baseball season starts next month",
        "Rugby players celebrated their victory",
        "The marathon runner crossed the finish line",
    ]
    # Cluster 1: technology
    + [
        "The new processor chip is faster than ever",
        "Software engineers deployed the update",
        "Machine learning models improved accuracy",
        "The database server crashed overnight",
        "Cloud computing reduces infrastructure costs",
        "Developers fixed the critical security bug",
        "The smartphone has an upgraded camera sensor",
        "Artificial intelligence transforms industries",
        "The network latency was reduced significantly",
        "Open source software powers the internet",
    ]
    # Cluster 2: cooking
    + [
        "The chef prepared a delicious pasta dish",
        "Baking bread requires patience and flour",
        "The recipe calls for fresh garlic and herbs",
        "Grilled vegetables taste better with olive oil",
        "The cake was decorated with chocolate frosting",
        "Simmering the soup brings out the flavors",
        "Fresh ingredients make the salad taste great",
        "The kitchen was filled with wonderful aromas",
        "Marinating the chicken overnight adds flavor",
        "The dessert menu featured homemade ice cream",
    ]
    # Cluster 3: space
    + [
        "The rocket launched successfully into orbit",
        "Astronomers discovered a new exoplanet",
        "The space station orbits Earth every ninety minutes",
        "Mars rovers send back amazing photographs",
        "The telescope captured images of distant galaxies",
        "Satellites monitor weather patterns from space",
        "Astronauts conducted experiments in microgravity",
        "The lunar mission plans are ahead of schedule",
        "A comet will be visible next week",
        "The nebula shines brightly in infrared light",
    ]
    # Cluster 4: music
    + [
        "The concert hall was packed with fans",
        "Guitar strings vibrate to produce sound",
        "The orchestra performed a beautiful symphony",
        "Piano lessons improve coordination and focus",
        "The drummer kept perfect rhythm all night",
        "Streaming platforms changed how we listen to music",
        "The singer released a new album this week",
        "Jazz musicians improvise complex melodies",
        "The choir sang in perfect harmony",
        "Vinyl records are making a comeback",
    ]
)


@pytest.fixture
def documents():
    """50 short-sentence documents across 5 topic clusters."""
    return DOCS


@pytest.fixture
def embeddings():
    """Pre-computed random embeddings (50 × 64) with cluster structure.

    We inject cluster structure so HDBSCAN can find meaningful groups:
    each cluster is centered around a distinct random centroid.
    """
    rng = np.random.default_rng(42)
    n_docs = 50
    dim = 64
    n_clusters = 5
    docs_per_cluster = 10

    # Create distinct centroids far apart
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centroids *= 5.0  # spread them out

    embs = np.zeros((n_docs, dim), dtype=np.float32)
    for i in range(n_clusters):
        start = i * docs_per_cluster
        end = start + docs_per_cluster
        noise = rng.standard_normal((docs_per_cluster, dim)).astype(np.float32) * 0.3
        embs[start:end] = centroids[i] + noise

    return embs


# ---------------------------------------------------------------------------
# Test 1: Basic pipeline — MlxUMAPWrapper + HDBSCAN + MlxCTFIDF
# ---------------------------------------------------------------------------


class TestBasicPipeline:
    """Full BERTopic pipeline with MLX UMAP + standard HDBSCAN + MlxCTFIDF."""

    def test_fit_transform_no_crash(self, documents, embeddings):
        """Pipeline runs to completion without errors."""
        topic_model = BERTopic(
            umap_model=MlxUMAPWrapper(n_components=5, n_neighbors=10),
            hdbscan_model=HDBSCAN(min_cluster_size=5, min_samples=3),
            ctfidf_model=MlxCTFIDF(),
            # No representation_model — skip KeyBERT/MMR which need real embeddings
            calculate_probabilities=False,
        )
        topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)

        assert len(topics) == 50
        # Should find at least 2 distinct non-outlier topics
        unique_topics = set(topics)
        non_outlier = {t for t in unique_topics if t >= 0}
        assert len(non_outlier) >= 2, f"Expected ≥2 topics, got {non_outlier}"

    def test_topics_are_integers(self, documents, embeddings):
        """All topic assignments are integers."""
        topic_model = BERTopic(
            umap_model=MlxUMAPWrapper(n_components=5, n_neighbors=10),
            hdbscan_model=HDBSCAN(min_cluster_size=5, min_samples=3),
            ctfidf_model=MlxCTFIDF(),
            calculate_probabilities=False,
        )
        topics, _ = topic_model.fit_transform(documents, embeddings=embeddings)

        for t in topics:
            assert isinstance(t, (int, np.integer)), f"Topic {t} is {type(t)}"


# ---------------------------------------------------------------------------
# Test 2: Outlier reduction
# ---------------------------------------------------------------------------


class TestOutlierReduction:
    """Outlier reduction via reduce_outliers_embeddings."""

    def test_reduces_outliers(self, embeddings):
        """Manually create outlier assignments and verify they get reduced."""
        n = len(embeddings)
        # Assign first half as outliers (-1), second half as real topics
        topics = np.full(n, -1, dtype=np.int32)
        # Assign cluster labels to the last 25 docs (they have real structure)
        topics[25:35] = 0
        topics[35:45] = 1
        topics[45:50] = 2

        n_outliers_before = np.sum(topics == -1)
        assert n_outliers_before == 25

        # reduce_outliers_embeddings should reassign outliers to nearest centroid
        new_topics = reduce_outliers_embeddings(
            topics, embeddings, threshold=0.0
        )

        n_outliers_after = np.sum(new_topics == -1)
        assert n_outliers_after < n_outliers_before, (
            f"Expected fewer outliers: before={n_outliers_before}, after={n_outliers_after}"
        )

    def test_threshold_controls_reassignment(self, embeddings):
        """Higher threshold means fewer reassignments (more remain outliers)."""
        n = len(embeddings)
        topics = np.full(n, -1, dtype=np.int32)
        topics[0:10] = 0
        topics[10:20] = 1

        # threshold=0.0 should reassign all outliers
        new_low = reduce_outliers_embeddings(topics, embeddings, threshold=0.0)
        # threshold=0.99 should reassign very few
        new_high = reduce_outliers_embeddings(topics, embeddings, threshold=0.99)

        outliers_low = np.sum(new_low == -1)
        outliers_high = np.sum(new_high == -1)

        assert outliers_low <= outliers_high, (
            f"Low threshold should reduce more outliers: "
            f"low_thresh_outliers={outliers_low}, high_thresh_outliers={outliers_high}"
        )


# ---------------------------------------------------------------------------
# Test 3: MlxCTFIDF with scipy sparse input
# ---------------------------------------------------------------------------


class TestMlxCTFIDFSparse:
    """MlxCTFIDF correctly handles scipy sparse input and returns sparse output."""

    def test_sparse_input_returns_sparse(self):
        """fit_transform with sparse input returns scipy sparse CSR matrix."""
        rng = np.random.default_rng(123)
        # Simulate a bag-of-words matrix: 10 topics × 200 vocab
        dense = rng.poisson(lam=2, size=(10, 200)).astype(np.float32)
        sparse_input = sp.csr_matrix(dense)

        ctfidf = MlxCTFIDF()
        result = ctfidf.fit_transform(sparse_input)

        assert sp.issparse(result), f"Expected sparse, got {type(result)}"
        assert result.shape == (10, 200)

    def test_fit_then_transform_sparse(self):
        """Separate fit() + transform() with sparse input."""
        rng = np.random.default_rng(456)
        dense = rng.poisson(lam=3, size=(8, 150)).astype(np.float32)
        sparse_input = sp.csr_matrix(dense)

        ctfidf = MlxCTFIDF()
        ctfidf.fit(sparse_input)
        result = ctfidf.transform(sparse_input)

        assert sp.issparse(result), f"Expected sparse, got {type(result)}"
        assert result.shape == (8, 150)

    def test_output_values_finite(self):
        """All output values should be finite (no NaN/Inf)."""
        rng = np.random.default_rng(789)
        dense = rng.poisson(lam=1, size=(5, 100)).astype(np.float32)
        sparse_input = sp.csr_matrix(dense)

        ctfidf = MlxCTFIDF()
        result = ctfidf.fit_transform(sparse_input)

        result_dense = result.toarray()
        assert np.all(np.isfinite(result_dense)), "Output contains NaN or Inf"

    def test_dense_input_also_returns_sparse(self):
        """Even dense numpy input should return sparse output."""
        rng = np.random.default_rng(101)
        dense = rng.poisson(lam=2, size=(6, 80)).astype(np.float32)

        ctfidf = MlxCTFIDF()
        result = ctfidf.fit_transform(dense)

        assert sp.issparse(result), f"Expected sparse, got {type(result)}"
        assert result.shape == (6, 80)
