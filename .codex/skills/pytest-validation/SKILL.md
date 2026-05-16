---
name: pytest-validation
description: "Use when running neuralib2 package tests across every package or a package subset: create or refresh each package .venv with uv, install package test dependencies, run pytest from that package, remove the venv only when tests pass, and keep failing venvs for debugging."
---

# Pytest Validation

Use the shared repo workflow at `docs/agent-workflows/pytest-validation.md`.

Core rule: each package gets its own local `.venv`. If that package passes, remove its `.venv` afterward. If install or tests fail, report the failure and keep that `.venv` in place for debugging.

Required steps:

- Read the shared workflow before starting.
- Resolve packages from `packages/*/pyproject.toml`.
- For each target package, run sync/install and test commands from the package root.
- If `.venv` already exists, keep it but resync it with the selected package extra before testing.
- Use `uv venv` and `uv pip install -e '.[test]'` when the package exposes a `test` extra; otherwise use `uv pip install -e '.[all]'` when an `all` extra exists, falling back to `uv pip install -e .`.
- Always install `pytest` explicitly into the package venv.
- Run `.venv/bin/python -m pytest` against package-local test paths, excluding any `manual/` tests.
- Remove only passing package `.venv` directories.
- Preserve failing package `.venv` directories and report their paths.
- In the final response, provide a package-by-package pass/fail table, commands used, removed venvs, preserved venvs, and the first useful failure detail for each failure.
