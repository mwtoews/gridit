# ChangeLog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.5] - 2023-12-07
### Added
- Add vector (polygon) attributes and functions, including `cell_geoms` property, a `write_vector()` method.
- Add `cell_geoseries()` and `cell_geodataframe()` methods.
- Allow other geometry types to be used (not just polygon).
### Changed
- Change `array_from_vector` to use `refine=None`, to auto-select appropriate value.
- Change `mask_from_vector` to `use refine=1` and `all_touched=True`.
- Better filter for `Grid.from_vector` class method.
- Build Grid from FloPy `modelgrid` object.
### Fixed
- Allow write_raster to write bool array; fix issue with nodata.
- Fix resampling option for `array_from_raster` and `array_from_array`.

## [0.4] - 2022-11-09
### Added
- Add `all_touched` option for `array_from_vector`.
- Add Dependabot to project.
### Fixed
- Fix issue calculating catchment weights with time stats.
- Fix splitting option on absolute paths with drives (e.g. "C:")

## [0.3] - 2022-09-19
### Added
- Add `array_to_array` function to resample numpy arrays.
- Add write outputs to cli, including `--write-image`, `--write-raster` and `--write-text`.
- Add `Grid.from_modflow` and `mask_from_modflow` functions.

### Changed
- Better missing data handling with `array_from_vector` function.
- Remove `--array-from-raster-bidx`, use `--array-from-raster FILE[:BIDX]` instead.
- Handle additional indexing with netCDF files.

## [0.2] - 2022-08-30
### Added
- Specify Zenodo DOI for project.
- Use setuptools_scm for version and file management.
- Start this change log file.

## [0.1] - 2022-08-29
### Added
- Initial version.

[Unreleased]: https://github.com/mwtoews/gridit/compare/0.5...HEAD
[0.5]: https://github.com/mwtoews/gridit/compare/0.4...0.5
[0.4]: https://github.com/mwtoews/gridit/compare/0.3...0.4
[0.3]: https://github.com/mwtoews/gridit/compare/0.2...0.3
[0.2]: https://github.com/mwtoews/gridit/compare/0.1...0.2
[0.1]: https://github.com/mwtoews/gridit/tree/0.1
