# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] - 2026-02-07

### Added

- New preprocessing CLI pipeline for widefield calcium imaging (`neuralib.widefield.preproc`)
  - ΔF/F₀ computation with sliding window baseline estimation
  - Motion correction support
  - Image rotation and spatial processing
  - Parallel processing with joblib
  - GPU acceleration via CuPy (optional)
  - Numba JIT compilation for performance (optional)
- New metadata module for preprocessing results (`neuralib.widefield.meta`)
  - `PreprocessMeta` TypedDict for structured metadata
  - `load_preprocess_meta()` for loading JSON metadata files
- Added README documentation for package usage

### Fixed

- CI test and dependency issues

## [0.7.0] - 2026-01-10

### Added

- Initial release of `neuralib-imaging` as a standalone package
- Migrated from monolithic [neuralib](https://github.com/ytsimon2004/neuralib)
  repository ([neura-library](https://pypi.org/project/neura-library/) on PyPI)
- Modular architecture separating imaging tools

### Changed

- Package renamed from `neura-library` to `neuralib-imaging`
- Python 3.11+ required (dropped Python 3.10 support)
