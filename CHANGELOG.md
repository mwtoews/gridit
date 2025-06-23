# ChangeLog
All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
### Changed
### Fixed

## [0.8] - 2025-06-23
### Added
- Add `Grid.array_from_geom()` method for Shapely objects (#60).
- Add `Grid.array_from_geopandas()` method for GeoPandas objects (#61).
### Changed
- Drop Python 3.9; set minimum Python 3.10 (#57).
- Refactor `array_from_vector()` method behaviour (#58).
- Refactor `array_from_vector()` method for Polygon without attribute (#59).

## [0.7] - 2025-01-30
### Added
- Add `snap` parameter to allow different snap modes used for grid size and offsets (#38).
- Create modelgrid without flopy; allow loading from `.dis.grb` (#39).
- Support pickle serialization for Grid objects (#46).
- Add `Grid.corner_coordinates` attribute (#48).
### Changed
- Drop Python 3.8 (#42).
- Allow kwargs for raster/vector creation options (#40).
- Allow negative buffer to contract bounds (#50).
- Enhance buffer to allow a tuple of two or four items (#52).
### Fixed
- Handle unset nodata and correctly set mask for expanded areas (#37).
- `Grid.from_raster` should keep array shape and bounds by default (#41).
- Fix version checking for Fiona 1.10 for some options (#47).

## [0.6] - 2024-06-26
### Changed
- Change `Grid(projection)` to be `None` (#34).
- Change build to hatch, use only ruff format and lint tools, add workflow to publish to PyPI (#33).
### Fixed
- Fix tests with latest numpy, xarray, fiona and rasterio/GDAL versions (#32).

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

[Unreleased]: https://github.com/mwtoews/gridit/compare/0.8...HEAD
[0.8]: https://github.com/mwtoews/gridit/compare/0.7...0.8
[0.7]: https://github.com/mwtoews/gridit/compare/0.6...0.7
[0.6]: https://github.com/mwtoews/gridit/compare/0.5...0.6
[0.5]: https://github.com/mwtoews/gridit/compare/0.4...0.5
[0.4]: https://github.com/mwtoews/gridit/compare/0.3...0.4
[0.3]: https://github.com/mwtoews/gridit/compare/0.2...0.3
[0.2]: https://github.com/mwtoews/gridit/compare/0.1...0.2
[0.1]: https://github.com/mwtoews/gridit/tree/0.1
