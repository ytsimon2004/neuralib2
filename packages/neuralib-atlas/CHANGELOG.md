# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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