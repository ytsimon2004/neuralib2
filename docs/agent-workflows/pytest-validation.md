# Pytest Validation Workflow

Use this workflow to install isolated environments and run tests for neuralib2 packages.

## Package Discovery

Target packages are directories under `packages/` that contain `pyproject.toml`.

Current package set:

- `packages/neuralib-atlas`
- `packages/neuralib-imaging`
- `packages/neuralib-metric`
- `packages/neuralib-parser`
- `packages/neuralib-utils`

If the user asks for every package, discover dynamically:

```bash
find packages -maxdepth 2 -name pyproject.toml -print
```

Run packages one at a time so a failure can preserve only that package’s `.venv`.

## Per-Package Commands

Run all commands from the package root. If `.venv` already exists, do not assume it is current. First resync it with the dependency set selected below, then install `pytest` explicitly.

If the package defines a `test` optional dependency, use:

```bash
cd packages/<package-name>
uv venv
uv pip install -e '.[test]'
uv pip install pytest
```

If the package does not define a `test` optional dependency but does define an `all` optional dependency, use:

```bash
uv pip install -e '.[all]'
uv pip install pytest
```

If neither `test` nor `all` is defined, use:

```bash
uv pip install -e .
uv pip install pytest
```

Then run tests through the package-local venv:

```bash
.venv/bin/python -m pytest <test-paths>
```

Build `<test-paths>` from package-local tests, excluding any `manual/` directory. Do not run manual tests by default. For example:

```bash
.venv/bin/python -m pytest test/unit
```

If a package has tests directly under `test/` plus a `test/manual/` subtree, include the non-manual test files/directories explicitly and omit `test/manual`. If all tests are under `test/manual`, skip testing that package and report that only manual tests exist.

Do not use `uv run pytest test` from package roots in this repository. Because these packages are inside the repo workspace, `uv run` can resolve against the workspace root instead of the package-local `.venv`, causing false import failures.

## Cleanup Rule

After each package:

- If install and tests pass, remove that package’s `.venv`.
- If install or tests fail, keep that package’s `.venv`.
- Never remove a failing `.venv` unless the user explicitly asks.

Passing cleanup command:

```bash
rm -rf .venv
```

Preserved failure path:

```text
packages/<package-name>/.venv
```

## Failure Handling

Do not stop at the first failure unless the user asks for fail-fast. Continue to the remaining packages.

For each failed package, capture:

- Whether failure happened during venv creation, install, or pytest.
- The failing command.
- The first useful error summary.
- The preserved `.venv` path.

Avoid dumping full logs in the final response. Provide enough detail for the next debugging step and mention where the environment was kept.

## Sandbox And Approvals

These commands often need network/cache access:

- `uv venv`
- `uv pip install ...`
- `.venv/bin/python -m pytest ...` if pytest imports packages that read external caches

If a command fails with sandbox, cache, or network permission errors, rerun the same command with approval rather than changing the workflow.

## Final Report

Use a compact table:

| Package | Install | Tests | Venv |
| --- | --- | --- | --- |
| `neuralib-atlas` | pass | pass | removed |
| `neuralib-imaging` | pass | fail | kept: `packages/neuralib-imaging/.venv` |

Also include:

- Commands used.
- Any package skipped and why.
- Preserved `.venv` paths.
- First useful failure detail for each failing package.
