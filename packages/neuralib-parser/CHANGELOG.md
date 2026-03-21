# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] - 2026-03-21

### Added

- New `cellpose` optional extra (`roifile`)
- New `opt` optional extra for heavier optional dependencies (tensorflow, napari, vedo, csbdeep, tables)
- Added `ruamel.yaml` to `all` extra for DLC support
- Updated `all` extra to include `cellpose`, `czi`, and `opt` extras

### Changed

- Simplified `stardist` extra (heavy deps moved to `opt`)
- Moved `StarDist2DOptions` import inside test method to avoid unconditional import

## [0.7.0] - 2026-01-10

### Added

- Initial release of `neuralib-parser` as a standalone package
- Migrated from monolithic [neuralib](https://github.com/ytsimon2004/neuralib)
  repository ([neura-library](https://pypi.org/project/neura-library/) on PyPI)
- Modular architecture separating data parsers

### Changed

- Package renamed from `neura-library` to `neuralib-parser`
- Python 3.11+ required (dropped Python 3.10 support)