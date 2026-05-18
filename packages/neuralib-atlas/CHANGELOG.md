# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.2] - 2026-05-18

### Added

- Added `brainglobe-atlasapi` as a required dependency for atlas data and slice-view APIs.
- Added regression coverage for atlas-name propagation, Plotly sunburst input, slice-view indexing, and overlap channel copying.

### Changed

- Converted atlas data tests to pytest-style functions.
- Updated README installation guidance for optional `brainrender` support.

### Fixed

- Fixed `get_leaf_in_annotation()` to use the requested atlas when building the annotation leaf map.
- Fixed `plot_sunburst_acronym()` to use `acronym` and `parent_acronym` columns from paired structure-tree data.
- Fixed integer slice-plane indexing, plane-bound clamping, and non-micrometer slice extents.
- Fixed overlap channel copying to compare channel names by equality.

## [0.7.1] - 2026-03-21

### Added

- New `brainrender` optional extra for heavy dependencies (brainrender, numba)

### Changed

- Moved `brainrender`, `numba`, and `argclz` from required to optional `brainrender` extra
- Added `all` extra that pulls in `brainrender` extra

### Fixed

- Typo in CLI entry point: `nl_brainredner` → `nl_brainrender`

## [0.7.0] - 2026-01-10

### Added

- Initial release of `neuralib-atlas` as a standalone package
- Migrated from monolithic [neuralib](https://github.com/ytsimon2004/neuralib)
  repository ([neura-library](https://pypi.org/project/neura-library/) on PyPI)
- Modular architecture separating atlas tools

### Changed

- Package renamed from `neura-library` to `neuralib-atlas`
- Python 3.11+ required (dropped Python 3.10 support)
