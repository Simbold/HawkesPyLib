# TODO — next release

Tracking items for the next HawkesPyLib release (after `0.3.0`).

## Python support
- [ ] Keep the minimum supported Python at **3.10+** only. `pyproject.toml` already sets
      `requires-python = ">=3.10"` and classifiers `3.10–3.13`; make sure the CI test
      matrix and docs stay in sync and that no 3.8/3.9 references creep back in.

## Release / publishing (pre-release → TestPyPI)
- [ ] Fix the pre-release → TestPyPI flow. The `v0.3.0rc1` publish failed because the
      "verify tag matches version" guard requires the `pyproject` version to equal the
      tag, and `pyproject` was `0.3.0`. Before the next pre-release, either:
    - adopt `setuptools_scm` so the git tag is the single source of truth (no manual
      version bump, and the guard becomes unnecessary), **or**
    - bump `pyproject` `version` to the exact rc string (e.g. `0.3.1rc1`) before
      tagging `v0.3.1rc1`.
- [ ] Do a real TestPyPI dry-run for the next pre-release: tag `vX.Y.ZrcN`, confirm the
      `Release (TestPyPI)` workflow is green, and `pip install` from test.pypi.org
      before cutting the PyPI release.

## Docs
- [ ] Review the rendered README on the PyPI project page (badges, LaTeX math, and the
      Quickstart code block) once `0.3.x` is published.
