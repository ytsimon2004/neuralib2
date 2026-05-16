---
name: publish-log
description: "Use when preparing a neuralib2 package release from changelog notes: update the package CHANGELOG.md, verify package version and tag naming, optionally commit and push release-prep changes, and draft a GitHub release without publishing it unless explicitly requested."
---

# Publish Log

Use the shared repo workflow at `docs/agent-workflows/publish-log.md`.

Core rule: treat this as release preparation, not an automatic production publish. Creating or pushing a release tag can trigger the package PyPI publish workflow, so confirm before tag creation or tag push.

Required steps:

- Read the shared workflow before starting.
- Resolve the target package under `packages/<package-name>`.
- Verify `pyproject.toml`, `CHANGELOG.md`, and `.github/workflows/<package-name>-publish.yml` agree on the package name and version/tag convention.
- Update the package changelog using Keep a Changelog style.
- Show the changed files before any git write operation.
- Commit and push only when requested or clearly part of the user’s task.
- Draft the GitHub release from the changelog section; do not publish the release unless explicitly requested.
- In the final response, state the package, version, tag name, changelog section used, git actions taken, release draft URL if created, and whether PyPI publishing was triggered.
