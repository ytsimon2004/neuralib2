# Publish Log Workflow

Use this workflow when preparing a changelog-backed package release for this repository.

## Package Map

Packages live under `packages/` and each package has its own `pyproject.toml` and `CHANGELOG.md`.

Known package/tag conventions:

| Package | Path | Release tag pattern | Publish workflow |
| --- | --- | --- | --- |
| `neuralib-atlas` | `packages/neuralib-atlas` | `neuralib-atlas-v<version>` | `.github/workflows/neuralib-atlas-publish.yml` |
| `neuralib-imaging` | `packages/neuralib-imaging` | `neuralib-imaging-v<version>` | `.github/workflows/neuralib-imaging-publish.yml` |
| `neuralib-metric` | `packages/neuralib-metric` | `neuralib-metric-v<version>` | `.github/workflows/neuralib-metric-publish.yml` |
| `neuralib-parser` | `packages/neuralib-parser` | `neuralib-parser-v<version>` | `.github/workflows/neuralib-parser-publish.yml` |
| `neuralib-utils` | `packages/neuralib-utils` | `neuralib-utils-v<version>` | `.github/workflows/neuralib-utils-publish.yml` |

Pushing one of these tags triggers the corresponding PyPI publish workflow.

## Inputs To Resolve

Resolve these from the user request or local files:

- Package name, such as `neuralib-imaging`.
- Release version, normally from the package `pyproject.toml`.
- Changelog notes, normally from existing `CHANGELOG.md` entries or user-provided release notes.
- Whether the user wants only file edits, or also commit, push, tag, and draft release creation.

If package or version is ambiguous, inspect `packages/*/pyproject.toml`. Ask only if ambiguity remains after local inspection.

## Changelog Update

Use the package-local changelog:

```text
packages/<package-name>/CHANGELOG.md
```

Keep the existing Keep a Changelog shape:

```md
## [<version>] - YYYY-MM-DD

### Added

- ...

### Changed

- ...

### Fixed

- ...
```

Rules:

- Use the current local date for the release date.
- Preserve existing sections and ordering.
- If there is an `Unreleased` section, move relevant bullets into the new version section.
- If there is no `Unreleased` section, add the new version section above the latest released version.
- Do not invent release notes. Summarize only local diffs, issue/PR context, or user-provided notes.
- Keep package names, CLI commands, and tag names exact.

## Verification

Before git or GitHub release actions:

1. Read `packages/<package-name>/pyproject.toml` and verify `name` and `version`.
2. Read `.github/workflows/<package-name>-publish.yml` and verify the tag pattern is `<package-name>-v*`.
3. Set the release tag as `<package-name>-v<version>`.
4. Check the tag does not already exist locally:

   ```bash
   git tag --list '<package-name>-v<version>'
   ```

5. If network/GitHub access is available, also check the remote tag or release before creating one.
6. Run package-appropriate validation only when requested or when the release notes imply code changes. Prefer narrow checks over broad unrelated test runs.

## Git Workflow

Never include unrelated user changes in release commits.

Recommended flow:

```bash
git status --short
git diff -- packages/<package-name>/CHANGELOG.md packages/<package-name>/pyproject.toml
git add packages/<package-name>/CHANGELOG.md
git commit -m "Prepare <package-name> v<version> release"
git push
```

If `pyproject.toml` also needs a version bump, include it in the same release-prep commit.

Confirm before creating or pushing a release tag, because tag push triggers PyPI publishing:

```bash
git tag <package-name>-v<version>
git push origin <package-name>-v<version>
```

## Draft GitHub Release

Create a draft release, not a published release, unless the user explicitly asks to publish.

Use the changelog section for `<version>` as the release body. Title format:

```text
<package-name> v<version>
```

Tag format:

```text
<package-name>-v<version>
```

With GitHub CLI, the intended command shape is:

```bash
gh release create <package-name>-v<version> --title "<package-name> v<version>" --notes-file <notes-file> --draft
```

If using the GitHub app connector instead, create the equivalent draft release only if release-creation tooling is available. If no release tool is available, prepare the title, tag, and body for the user and state that the release was not created.

## Final Response

Report:

- Package and version.
- Tag name.
- Changelog file and section updated.
- Validation commands run and results.
- Git commit/push/tag actions performed, if any.
- Draft release URL, if created.
- Whether PyPI publish was triggered. This should be “no” unless a release tag was pushed.
