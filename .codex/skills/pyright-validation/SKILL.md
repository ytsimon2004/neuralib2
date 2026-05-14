---
name: pyright-validation
description: Use when checking or fixing Pyright errors in a Python package or submodule. Requires installing the package dependencies into the target environment before trusting Pyright output, then applying narrow type fixes or suppressions only after dependency validation.
---

# Pyright Validation

Use the shared repo workflow at `docs/agent-workflows/pyright-validation.md`.

Core rule: install the target package dependencies into the same environment Pyright will use before trusting diagnostics or adding suppressions.

Required steps:

- Read the shared workflow before starting.
- Install the package dependencies first, usually with `uv pip install -e '.[all]'` from the package root.
- If `pyproject.toml`, extras, or lock files changed, treat the venv as stale and reinstall before Pyright.
- If dependency metadata is being validated or dependencies were removed, consider a clean venv because normal installs may leave old packages behind.
- Run `uvx pyright -p path/to/package/pyrightconfig.json`.
- Do not use broad `reportMissingImports=false` for an incomplete environment.
- Prefer real typing fixes and narrow inline suppressions.
- In the final response, state the install command, Pyright command/result, and any remaining suppressions.
