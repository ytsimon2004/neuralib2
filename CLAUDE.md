# Agent Workflow Notes

This repository is developed with both Claude Code and Codex. Shared agent workflows live in `docs/agent-workflows/`.

When checking or fixing Pyright diagnostics in any Python package or submodule, follow `docs/agent-workflows/pyright-validation.md`.

Most important rule: install the package dependencies into the same environment Pyright will use before trusting diagnostics or adding suppressions.
