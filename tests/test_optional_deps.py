"""Optional-dependency behaviour: mlx-embeddings is an extra, not a requirement."""

import sys
from pathlib import Path

import pytest

from mlx_bertopic.embedder import _INSTALL_HINT, MlxEmbedder

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_missing_mlx_embeddings_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_embeddings", None)
    with pytest.raises(ImportError) as excinfo:
        MlxEmbedder("mlx-community/bge-small-en-v1.5-bf16")
    message = str(excinfo.value)
    assert "[embed]" in message
    assert "git+https://github.com/mlx-bertopic/mlx-bertopic.git" in message


def test_install_hint_names_license_and_alternative():
    assert "GPL-3.0" in _INSTALL_HINT
    assert "embed(documents)" in _INSTALL_HINT


def test_mlx_embeddings_is_only_an_extra():
    tomllib = pytest.importorskip("tomllib", reason="needs Python 3.11+ to parse pyproject")
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    base = data["project"]["dependencies"]
    extras = data["project"]["optional-dependencies"]

    assert not any(dep.startswith("mlx-embeddings") for dep in base), base
    assert any(dep.startswith("mlx-embeddings") for dep in extras["embed"])
