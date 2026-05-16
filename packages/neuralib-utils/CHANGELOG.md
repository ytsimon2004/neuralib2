# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.3] - 2026-05-16

### Changed

- Enabled ruff lint rule groups `I`, `UP`, `B`, `C4` for the package
- Migrated `Sequence`, `Callable`, `Iterator` imports from `typing` to `collections.abc`
- Replaced `Optional[X]` with `X | None` and `Type` with `type` throughout
- Import blocks sorted consistently (stdlib → third-party → local)
- Replaced `getattr`/`setattr` calls with direct attribute access in `persistence.py` and `verbose.py`
- Tightened `zip(..., strict=True)` where iterables are guaranteed equal-length (`colormap.py`, `gspread.py`)
- Documented silent trailing-element truncation in `grouped_iter` (`segments.py`)

## [0.7.2] - 2026-03-21

### Added

- Added `pdf2image` to `opt` optional dependencies

## [0.7.1] - 2026-01-11

### Changed

- Minor fixes for the dependencies/module path import

## [0.7.0] - 2026-01-10

### Added

- Initial release of `neuralib-utils` as a standalone package
- Migrated from monolithic [neuralib](https://github.com/ytsimon2004/neuralib)
  repository ([neura-library](https://pypi.org/project/neura-library/) on PyPI)
- Core utilities foundation for the neuralib ecosystem

### Changed

- Package renamed from `neura-library` to `neuralib-utils`
- Python 3.11+ required (dropped Python 3.10 support)
