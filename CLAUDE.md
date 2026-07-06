# HawkesPyLib — contributor & agent guide

Python library for simulation and inference of univariate Hawkes processes.
The public API is three modules under `src/HawkesPyLib/`: `simulation`,
`inference`, and `processes`. Hot numerical paths live in `core/` and are
JIT-compiled with numba.

> This file is tool-neutral on purpose. The real project knowledge lives in the
> sources linked below (readable by any human or agent). `CLAUDE.md` is just an
> entry point — mirror it to `AGENTS.md` (`cp CLAUDE.md AGENTS.md`) for other
> agentic tools if you use them.

## Getting oriented
- **Architecture map (platform-independent):** if present, `graphify-out/GRAPH_REPORT.md`
  is a knowledge-graph overview of the modules, their communities and relationships
  (`graphify-out/graph.html` is the interactive version). The map is produced by the
  `graphify` tool — (re)generate it if it is missing or stale.
- **Docs:** `README.md` and https://simbold.github.io/HawkesPyLib/

## Working in this repo
- **Tests:** `tox` (Python 3.10–3.13), or in a venv: `pip install -e ".[testing]"`
  then `pytest`.
- **Lint:** `ruff check src/HawkesPyLib tests` (config in `pyproject.toml`).
- **Supported Python:** ≥ 3.10. Runtime deps: `numpy`, `scipy`, `numba`
  (numpy 2.x compatible).

## Releasing
Publishing is tag-driven via GitHub Actions with PyPI Trusted Publishing (OIDC —
no tokens). Bump `version` in `pyproject.toml`, then push a matching tag; CI checks
the tag equals the package version before publishing.
- **Test release → TestPyPI:** version `X.Y.ZrcN`, tag `vX.Y.ZrcN` (e.g. `v0.3.0rc1`).
- **Final release → PyPI:** version `X.Y.Z`, tag `vX.Y.Z` (e.g. `v0.3.0`).

## Conventions that matter
- Functions in `src/HawkesPyLib/core/` are compiled with numba `@njit` using
  **explicit type signatures**, e.g.
  `@njit(float64[:](float64, float64, int32), nogil=True)`. When you change a
  function's arguments, update its signature to match or compilation fails at
  call time.
- Use `np.random.default_rng(...)` for randomness (as in `inference`).
