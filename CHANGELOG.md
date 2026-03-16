# Changelog

## v1.2.0 — 2026-03-16

### Changed
- Updated CD conversion logic in `apply_event`.
- Removed strict `sufficient_pde_concentration_mask` gating condition.
- Implemented dynamic PDE mass redistribution when uniform removal is not possible.

### Added
- Equal redistribution algorithm for removing one particle of PDE mass.
- Additional tests covering:
  - uniform removal
  - redistribution
  - subunit probabilistic conversion

### Fixed
- Corrected control flow bug where redistribution could fall through into DC logic.


