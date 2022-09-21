# Gridit

[![Tests](https://github.com/mwtoews/gridit/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/mwtoews/gridit/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/530041277.svg)](https://zenodo.org/badge/latestdoi/530041277)

## Description

Gridit provides spatial tools to translate raster or vector polygon data to regular grids.

## Installation

This package primarily depends on NumPy and SciPy, and has several optional dependencies.

Pip can be used to install all optional dependencies:
```bash
$ pip install gridit[optional]
```

Or from a clone of this repository, create an "editable" install:
```bash
$ pip install -e .[optional]
```

### Testing

Run `pytest -v`.

## Examples

### Python

```python
>>> import matplotlib.pyplot as plt  # optional
>>> from gridit import Grid

>>> grid = Grid.from_vector("tests/data/Mana_polygons.shp", 100)
>>> print(grid)
<Grid: resolution=100.0, shape=(24, 18), top_left=(1748600.0, 5451200.0) />

>>> ar_vec = grid.array_from_vector("tests/data/Mana_polygons.shp", "K_m_d")
>>> plt.imshow(ar_vec)
<matplotlib.image.AxesImage at 0x7fb6c7dacf10>

>>> ar_rast = grid.array_from_raster("tests/data/Mana.tif")
>>> plt.imshow(ar_rast)
<matplotlib.image.AxesImage at 0x7fb6bc4ad6d0>
```

### Command line

Grid and array from vector, write PNG image:
```bash
$ gridit --grid-from-vector tests/data/Mana_polygons.shp --resolution 100 \
    --array-from-vector tests/data/Mana_polygons.shp \
    --array-from-vector-attribute=K_m_d \
    --write-image /tmp/Mana_Kmd.png
```

Grid from bounding box, array from raster, write GeoTIFF raster:
```bash
$ gridit --grid-from-bbox 1748600 5448800 1750400 5451200 --resolution 100 \
    --array-from-raster tests/data/Mana.tif \
    --write-raster /tmp/Mana_100m.tif
```

Grid from vector, array from netCDF, write text array file for each time stat:
```bash
$ gridit --grid-from-vector tests/data/waitaku2.shp --resolution 250 \
    --array-from-vector tests/data/waitaku2.shp \
    --array-from-vector-attribute rid \
    --array-from-netcdf tests/data/waitaku2.nc:rid:myvar:0 \
    --time-stats "quantile(0.75),max" \
    --write-text /tmp/waitaku2_cat.ref
```

Grid from MODFLOW, array from vector, write text array file:
```bash
$ gridit --grid-from-modflow tests/data/modflow/mfsim.nam:h6 \
    --array-from-vector tests/data/waitaku2.shp \
    --array-from-vector-attribute rid \
    --write-text /tmp/waitaku2_rid.txt
```

See other options with:
```bash
$ gridit -h
```

## Funding
Funding for the development of gridit has been provided by New Zealand Strategic Science Investment Fund as part of GNS Science’s (https://www.gns.cri.nz/) Groundwater Research Programme.
