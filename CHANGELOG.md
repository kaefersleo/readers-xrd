# Changelog

## Unreleased

### Added

- Return nominal acquisition times, scan-boundary timestamps and supported
  user/software/optics metadata with source paths. Unreported values remain absent.

### Changed

- Preserve declared signal units: count-scale input stays dimensionless and
  explicit rates use `count/s`. Reject unknown units and mixed count/rate XRDML scans.
- **Breaking:** RAW4 reports `2Theta` coordinates, native wavelengths and nominal
  step times in seconds. Consumers using the former `Theta` label must adapt.
  Shared Kα references provide fallback values when needed.

### Fixed

- Omit the shared XRDML scan-axis label when scans disagree. Warn once about the
  axis if a logger is supplied.
- Accept a single fixed 2Theta value. A reported Omega scan may omit 2Theta when
  Omega and intensity have the same shape. Return RAW4 source values as Pint
  quantities.

- Respect RAW4 block lengths, record sizes and point counts; reject unsupported
  modes, multiple ranges and truncated input.
- Keep all intensity and count samples in repeated scans and single-point scans.
  Use the axis reported by each RASX profile and prefer recorded BRML coordinates.
  Omit nominal counting times if any scan lacks a valid time.
