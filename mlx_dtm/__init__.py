"""mlx-dtm: LSTM-based Dynamic Topic Modeling on Apple Silicon."""

__version__ = "0.1.0"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from typing import Optional
import time


class TopicLSTM(nn.Module):
    """LSTM for smoothing and predicting topic proportion sequences."""

    def __init__(self, n_topics: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.n_topics = n_topics
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # Stack LSTM layers manually (MLX LSTM is single-layer)
        self.layers = [
            nn.LSTM(input_size=n_topics if i == 0 else hidden_dim, hidden_size=hidden_dim)
            for i in range(n_layers)
        ]
        self.proj = nn.Linear(hidden_dim, n_topics)

    def __call__(self, x):
        """Forward pass. x: (seq_len, n_topics) → (seq_len, n_topics) smoothed.

        MLX LSTM:
          - Input: (seq_len, input_size) or (..., seq_len, input_size)
          - Returns: (all_hidden, all_cell), each (seq_len, hidden_size)
        """
        h = x
        for layer in self.layers:
            all_hidden, _ = layer(h)  # (seq_len, hidden_dim)
            h = all_hidden
        logits = self.proj(h)  # (seq_len, n_topics)
        return mx.softmax(logits, axis=-1)

    def predict_next(self, x, steps: int = 1):
        """Autoregressive prediction from sequence x: (seq_len, n_topics)."""
        # Forward through full sequence to get final states
        h = x
        final_hiddens = []
        final_cells = []
        for layer in self.layers:
            all_hidden, all_cell = layer(h)
            # Last timestep state
            final_hiddens.append(all_hidden[..., -1, :])
            final_cells.append(all_cell[..., -1, :])
            h = all_hidden

        # Last output → first prediction
        last = mx.softmax(self.proj(all_hidden[..., -1:, :]), axis=-1)
        preds = [last.squeeze(-2)]

        for _ in range(steps - 1):
            inp = last
            new_hiddens = []
            new_cells = []
            for i, layer in enumerate(self.layers):
                all_h, all_c = layer(inp, hidden=final_hiddens[i], cell=final_cells[i])
                new_hiddens.append(all_h[..., -1, :])
                new_cells.append(all_c[..., -1, :])
                inp = all_h
            final_hiddens = new_hiddens
            final_cells = new_cells
            last = mx.softmax(self.proj(inp[..., -1:, :]), axis=-1)
            preds.append(last.squeeze(-2))

        return mx.stack(preds, axis=0)


class DynamicTopics:
    """LSTM-based Dynamic Topic Modeling.

    Takes BERTopic topic assignments + timestamps, produces smooth topic trends.

    Parameters
    ----------
    n_topics : int
        Number of topics (from BERTopic).
    hidden_dim : int
        LSTM hidden dimension.
    window_size : int
        Number of documents per time step.
    stride : int
        Sliding window stride.
    n_epochs : int
        Training epochs for self-supervised smoothing.
    lr : float
        Learning rate.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        n_topics: int = 50,
        hidden_dim: int = 128,
        window_size: int = 100,
        stride: int = 50,
        n_epochs: int = 50,
        lr: float = 0.001,
        verbose: bool = True,
    ):
        self.n_topics = n_topics
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.stride = stride
        self.n_epochs = n_epochs
        self.lr = lr
        self.verbose = verbose
        self._model = None
        self._raw_proportions = None
        self._smooth_proportions = None
        self._timestamps = None

    def fit(
        self,
        topics: list[int],
        timestamps: list,
        topic_ids: Optional[list[int]] = None,
    ) -> "DynamicTopics":
        """Fit dynamic topic model.

        Parameters
        ----------
        topics : list of int
            Topic assignment per document (from BERTopic).
        timestamps : list
            Timestamp per document (sortable).
        topic_ids : list of int, optional
            Which topic IDs to track. Default: top n_topics by frequency.
        """
        t0 = time.time()
        if self.verbose:
            print(f"Building topic sequence (window={self.window_size}, stride={self.stride})...", flush=True)

        # Sort by time
        df = pd.DataFrame({"topic": topics, "ts": timestamps})
        df = df.sort_values("ts").reset_index(drop=True)

        # Determine which topics to track
        if topic_ids is None:
            from collections import Counter
            top = Counter(t for t in topics if t >= 0).most_common(self.n_topics)
            topic_ids = [t for t, _ in top]
        self._topic_ids = topic_ids
        topic_to_idx = {t: i for i, t in enumerate(topic_ids)}

        # Sliding window → proportion sequence
        proportions = []
        window_times = []
        for start in range(0, len(df) - self.window_size, self.stride):
            window = df.iloc[start:start + self.window_size]
            counts = np.zeros(len(topic_ids), dtype=np.float32)
            for t in window["topic"]:
                if t in topic_to_idx:
                    counts[topic_to_idx[t]] += 1
            total = counts.sum()
            if total > 0:
                counts /= total
            proportions.append(counts)
            window_times.append(window["ts"].iloc[len(window) // 2])

        self._raw_proportions = np.array(proportions)  # (T, n_topics)
        self._timestamps = window_times
        seq_len = len(proportions)

        if self.verbose:
            print(f"  Sequence length: {seq_len} steps", flush=True)
            print(f"Training LSTM...", flush=True)

        # Model
        self._model = TopicLSTM(len(topic_ids), self.hidden_dim)

        # Self-supervised training: predict step t+1 from steps 0..t
        X = mx.array(self._raw_proportions[:-1])  # (T-1, n_topics)
        Y = mx.array(self._raw_proportions[1:])   # (T-1, n_topics)

        optimizer = optim.Adam(learning_rate=self.lr)

        def loss_fn(model, x, y):
            pred = model(x)  # (T-1, n_topics)
            # KL divergence: sum(y * log(y / pred))
            eps = 1e-8
            kl = mx.sum(y * (mx.log(y + eps) - mx.log(pred + eps)))
            return kl / y.shape[0]

        loss_and_grad = nn.value_and_grad(self._model, loss_fn)

        for epoch in range(self.n_epochs):
            loss, grads = loss_and_grad(self._model, X, Y)
            optimizer.update(self._model, grads)
            mx.eval(self._model.parameters(), optimizer.state)

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.n_epochs} loss={loss.item():.4f}", flush=True)

        # Generate smooth proportions
        smooth = self._model(mx.array(self._raw_proportions))
        mx.eval(smooth)
        self._smooth_proportions = np.array(smooth)

        if self.verbose:
            print(f"  Done in {time.time()-t0:.1f}s", flush=True)

        return self

    def topic_trends(self, smooth: bool = True) -> pd.DataFrame:
        """Return topic proportions over time as DataFrame."""
        props = self._smooth_proportions if smooth else self._raw_proportions
        df = pd.DataFrame(props, columns=[f"topic_{t}" for t in self._topic_ids])
        df["timestamp"] = self._timestamps
        return df

    def predict(self, steps: int = 5) -> np.ndarray:
        """Predict future topic proportions.

        Note: MLX does not build an autograd graph during the forward pass by
        default (gradients are only materialised when ``mx.grad`` /
        ``nn.value_and_grad`` are used), so there is no ``mx.no_grad`` context
        to enter for inference.
        """
        seq = mx.array(self._raw_proportions)
        preds = self._model.predict_next(seq, steps=steps)
        return np.array(preds)

    def changepoints(self, threshold: float = 2.0) -> list[int]:
        """Detect changepoints where topic distribution shifts sharply."""
        if self._smooth_proportions is None:
            raise ValueError("Call .fit() first")
        diffs = np.abs(np.diff(self._smooth_proportions, axis=0)).sum(axis=1)
        mean_diff = diffs.mean()
        std_diff = diffs.std()
        return [i for i, d in enumerate(diffs) if d > mean_diff + threshold * std_diff]

    def plot_trends(self, top_n: int = 10):
        """Plot smooth topic trends as stacked area chart."""
        import plotly.express as px
        df = self.topic_trends(smooth=True)
        cols = [c for c in df.columns if c.startswith("topic_")][:top_n]
        melted = df.melt(id_vars=["timestamp"], value_vars=cols,
                         var_name="topic", value_name="proportion")
        fig = px.area(melted, x="timestamp", y="proportion", color="topic",
                      title="Dynamic Topic Trends (LSTM smoothed)",
                      groupnorm="percent")
        return fig
