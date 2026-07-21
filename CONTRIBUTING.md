# Contributing

## Setup

```bash
git clone git@github.com:mlx-bertopic/mlx-bertopic.git
cd mlx-bertopic
pip install -e ".[dev]"
```

## Tests

```bash
pytest tests/ -q
```

All tests should pass on Apple Silicon (M1+). Tests compare MLX output against
upstream references (BERTopic, hdbscan, scikit-learn) — they are behavioral
tests, not unit tests of internals.

## Code style

- No formatter enforced (yet). Match surrounding code.
- Type hints appreciated but not required.
- Docstrings for public APIs.

## Adding a new module

1. Create `mlx_<name>/` with `__init__.py`
2. Add golden tests in `tests/test_<name>.py` comparing against upstream
3. Wire into `mlx_bertopic/__init__.py` exports
4. Update `pyproject.toml` packages list
5. Document in `README.md` table + `NOTICES.md`

## Reporting issues

- Include: macOS version, Apple Silicon model, Python version, mlx version
- For numerical issues: include dataset size + dimensionality
- For performance issues: include timing comparison vs CPU baseline
