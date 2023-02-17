import pytest

from .conftest import datadir, requires_pkg

from gridit import Grid


mana_dem_path = datadir / "Mana.tif"
mana_polygons_path = datadir / "Mana_polygons.shp"


def test_grid_from_bbox():
    grid = Grid.from_bbox(
        1748762.8, 5448908.9, 1749509, 5449749, 25)
    expected = Grid(25.0, (34, 31), (1748750.0, 5449750.0))
    assert grid == expected
    assert grid.bounds == (1748750.0, 5448900.0, 1749525.0, 5449750.0)


def test_grid_from_bbox_buffer():
    grid = Grid.from_bbox(
        1748762.8, 5448908.9, 1749509, 5449749, 25, 20, "EPSG:2193")
    expected = Grid(
        25.0, (35, 31), (1748750.0, 5449775.0), "EPSG:2193")
    assert grid == expected


@pytest.fixture
def grid_from_raster():
    return Grid.from_raster(mana_dem_path)


@requires_pkg("rasterio")
def test_grid_from_raster(grid_from_raster):
    grid = grid_from_raster
    expected = Grid(8.0, (278, 209), (1748688.0, 5451096.0), grid.projection)
    assert grid == expected


@requires_pkg("rasterio")
def test_grid_from_raster_resolution():
    grid = Grid.from_raster(mana_dem_path, 10.0)
    expected = Grid(10.0, (223, 168), (1748680.0, 5451100.0), grid.projection)
    assert grid == expected


@requires_pkg("rasterio")
def test_grid_from_raster_buffer():
    grid = Grid.from_raster(mana_dem_path, buffer=16.0)
    expected = Grid(8.0, (282, 213), (1748672.0, 5451112.0), grid.projection)
    assert grid == expected


@requires_pkg("rasterio")
def test_grid_from_raster_resolution_buffer():
    grid = Grid.from_raster(mana_dem_path, 10.0, 20.0)
    expected = Grid(10.0, (227, 171), (1748670.0, 5451120.0), grid.projection)
    assert grid == expected


@requires_pkg("rasterio")
def test_mask_from_raster(grid_from_raster):
    mask = grid_from_raster.mask_from_raster(mana_dem_path)
    assert mask.shape == (278, 209)
    assert mask.dtype == "bool"
    assert mask.sum() == 23782


@pytest.fixture
def grid_from_vector_all():
    return Grid.from_vector(mana_polygons_path, 100)


@requires_pkg("fiona")
def test_grid_from_vector_all(grid_from_vector_all):
    grid = grid_from_vector_all
    expected = Grid(100.0, (24, 18), (1748600.0, 5451200.0), grid.projection)
    assert grid == expected


@requires_pkg("fiona", "rasterio")
def test_mask_from_vector_all(grid_from_vector_all):
    mask = grid_from_vector_all.mask_from_vector(mana_polygons_path)
    assert mask.shape == (24, 18)
    assert mask.dtype == "bool"
    assert mask.sum() == 193


@requires_pkg("fiona", "rasterio")
def test_mask_from_vector_layer(grid_from_vector_all):
    mask = grid_from_vector_all.mask_from_vector(datadir, "mana_polygons")
    assert mask.shape == (24, 18)
    assert mask.dtype == "bool"
    assert mask.sum() == 193


@pytest.fixture
def grid_from_vector_filter():
    return Grid.from_vector(mana_polygons_path, 100, {"name": "South-east"})


@requires_pkg("fiona")
def test_grid_from_vector_filter(grid_from_vector_filter):
    grid = grid_from_vector_filter
    expected = Grid(100.0, (14, 13), (1749100.0, 5450400.0), grid.projection)
    assert grid == expected


@requires_pkg("fiona", "rasterio")
def test_grid_from_vector_buffer():
    grid = Grid.from_vector(mana_polygons_path, 100, buffer=500)
    expected = Grid(100.0, (32, 27), (1748200.0, 5451600.0), grid.projection)
    assert grid == expected


@requires_pkg("fiona")
def test_grid_from_vector_layer():
    grid = Grid.from_vector(datadir, 100, layer="mana_polygons")
    expected = Grid(100.0, (24, 18), (1748600.0, 5451200.0), grid.projection)
    assert grid == expected
