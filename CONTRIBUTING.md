# Contributing

Thanks for your interest in `shearwave`.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
make install-dev      # uv sync --dev
# or, with plain pip:
pip install -e ".[dev,jax]"
```

## Running the tests

```bash
make test             # uv run pytest
# or:
pytest -q
```

The JAX backend is optional. With `jax` installed, the NumPy/JAX parity tests
run; without it they are skipped. CI runs the full suite (including JAX) on
Python 3.10–3.12.

## Style

Code is formatted and linted with [ruff](https://docs.astral.sh/ruff/)
(configuration in `ruff.toml`):

```bash
ruff check .
ruff format .
```

## Pull requests

- Branch from `main` and open a PR against `main`.
- Keep changes focused; add or update tests for behavior changes.
- Numerical/solver changes should include a regression test that would fail
  without the change (see `tests/test_projection_static.py` for examples).
