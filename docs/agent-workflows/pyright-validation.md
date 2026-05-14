# Pyright Validation Workflow

Use this workflow when checking or fixing Pyright errors in a Python package or submodule.

## Core Rule

Install the target package dependencies into the same environment Pyright will use before trusting Pyright diagnostics or adding suppressions.

Do not add broad suppressions to compensate for an incomplete environment.

## Workflow

1. Identify the package root and metadata.
   - Read `pyproject.toml`.
   - Note `requires-python`, base dependencies, and optional dependency groups.
   - Prefer the package's existing `.venv` if it exists.

2. Ensure Pyright uses the installed environment.
   - If package-local `.venv` exists, use a package-local `pyrightconfig.json` like:

```json
{
  "venvPath": ".",
  "venv": ".venv",
  "pythonVersion": "3.11",
  "include": ["src"]
}
```

   - Adjust `pythonVersion` to match the package metadata or existing environment.
   - If no package-local venv exists, inspect the repo convention before creating one.

3. Install dependencies before checking.
   - From the package root, usually run:

```bash
uv pip install -e '.[all]'
```

   - If there is no `all` extra, install the relevant extras for the code being checked.
   - If the package uses `uv sync`, `tox`, `nox`, or another local convention, prefer that convention.
   - If install needs network or cache access and sandboxing blocks it, request approval and rerun the same install.
   - If `pyproject.toml`, dependency extras, or lock files changed, treat the existing `.venv` as stale and reinstall before running Pyright.
   - Remember that `uv pip install -e ...` can add or upgrade packages but may leave dependencies installed that were removed from `pyproject.toml`. If validating dependency metadata or investigating missing imports, prefer a clean environment or ask before recreating the package `.venv`.

4. Run Pyright consistently from the repo root or package root:

```bash
uvx pyright -p path/to/package/pyrightconfig.json
```

5. Classify diagnostics before editing.
   - Real typing bug: fix the annotation, guard, overload, generic, `TypedDict`, or return type.
   - Optional dependency: keep the import optional and use an inline ignore only on that import line if the dependency is intentionally absent from the installed extras.
   - Third-party stub mismatch: prefer a local cast, type alias, or narrow inline ignore near the affected call.
   - Dynamic GUI/runtime state: prefer explicit `None` guards or attributes where practical; use file-level suppressions only if the file is dominated by dynamic framework patterns and local fixes would obscure behavior.
   - Missing import after install: do not blanket-disable `reportMissingImports`; verify whether the dependency is optional or missing from package metadata.

6. Suppression policy.
   - Prefer no suppression.
   - Prefer narrow inline ignores with diagnostic codes.
   - Avoid file-level suppressions unless they are justified by repeated framework, stub, or dynamic-state diagnostics in one file.
   - Never use file-level `reportMissingImports=false` to compensate for not installing dependencies.
   - If a file-level suppression is necessary, keep `reportMissingImports` out of it unless every missing import in that file is proven optional and documented locally.

7. Re-run validation.
   - Re-run Pyright after each meaningful batch of fixes.
   - Run syntax validation:

```bash
uv run python -m compileall -q path/to/package/src
```

   - Run diff whitespace checks:

```bash
git diff --check -- path/to/package
```

8. Final report.
   - State the dependency install command that was run.
   - State the exact Pyright command and result.
   - Mention any remaining suppressions and why each is justified.
   - Mention if the check used an existing venv or a newly created one.
   - Mention any generated artifacts removed or left untracked.

## Review Checklist

Before finalizing, verify:

- Dependencies were installed before final Pyright.
- The dependency install was rerun after any `pyproject.toml`, extras, or lock-file change.
- Stale venv risk was considered, especially when dependencies were removed from metadata.
- Pyright config points to the environment that received the install.
- Missing imports were not hidden broadly.
- Suppressions are narrow and diagnostic-specific where practical.
- Runtime behavior was not changed just to satisfy Pyright unless the change is clearly correct.
- Unrelated untracked files are not staged or included.
