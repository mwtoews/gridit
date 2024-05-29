import pytest

from gridit import Grid

from .conftest import datadir, requires_pkg

mana_dem_path = datadir / "Mana.tif"
mana_polygons_path = datadir / "Mana_polygons.shp"
lines_path = datadir / "waitaku2_lines.shp"
points_path = datadir / "waitaku2_points.shp"
nocrs_path = datadir / "nocrs.tif"
nocrspoints_path = datadir / "points.csv"


def test_grid_from_bbox():
    grid = Grid.from_bbox(1748762.8, 5448908.9, 1749509, 5449749, 25)
    expected = Grid(25.0, (34, 31), (1748750.0, 5449750.0))
    assert grid == expected
    assert grid.bounds == (1748750.0, 5448900.0, 1749525.0, 5449750.0)


def test_grid_from_bbox_point():
    grid = Grid.from_bbox(1748762.8, 5448908.9, 1748762.8, 5448908.9, 25)
    expected = Grid(25.0, (1, 1), (1748750.0, 5448925.0))
    assert grid == expected
    assert grid.bounds == (1748750.0, 5448900.0, 1748775.0, 5448925.0)


def test_grid_from_bbox_buffer():
    grid = Grid.from_bbox(1748762.8, 5448908.9, 1749509, 5449749, 25, 20, "EPSG:2193")
    expected = Grid(25.0, (35, 31), (1748750.0, 5449775.0), "EPSG:2193")
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

    # with buffer
    grid = Grid.from_raster(mana_dem_path, 10.0, 20.0)
    expected = Grid(10.0, (227, 171), (1748670.0, 5451120.0), grid.projection)
    assert grid == expected


@requires_pkg("rasterio")
def test_grid_from_raster_nocrs():
    grid = Grid.from_raster(nocrs_path)
    expected = Grid(10.0, (2, 3), (1749700.0, 5449800.0))
    assert grid == expected
    assert grid.projection is None


@requires_pkg("fiona")
def test_grid_from_vector_point():
    # all
    grid = Grid.from_vector(points_path, 250)
    expected = Grid(250.0, (50, 28), (1810500.0, 5877750.0), grid.projection)
    assert grid == expected

    # filter
    grid = Grid.from_vector(points_path, 250, {"id": [5, 9]})
    expected = Grid(250.0, (19, 7), (1810500.0, 5873750.0), grid.projection)
    assert grid == expected

    grid = Grid.from_vector(points_path, 250, {"id": 5})
    expected = Grid(250.0, (1, 1), (1812000.0, 5869250.0), grid.projection)
    assert grid == expected

    # filter + buffer
    grid = Grid.from_vector(points_path, 250, {"id": 5}, buffer=240)
    expected = Grid(250.0, (2, 2), (1811750.0, 5869250.0), grid.projection)
    assert grid == expected


@requires_pkg("fiona")
def test_grid_from_vector_nocrs():
    # all
    grid = Grid.from_vector(nocrspoints_path, 250)
    expected = Grid(250.0, (8, 6), (1748750.0, 5451000.0))
    assert grid == expected
    assert grid.projection is None


@requires_pkg("fiona")
def test_grid_from_vector_polygon():
    # all
    grid = Grid.from_vector(mana_polygons_path, 100)
    expected = Grid(100.0, (24, 18), (1748600.0, 5451200.0), grid.projection)
    assert grid == expected

    # layer
    grid = Grid.from_vector(datadir, 100, layer="mana_polygons")
    assert grid == expected

    # filter
    grid = Grid.from_vector(mana_polygons_path, 100, {"name": "South-east"})
    expected = Grid(100.0, (14, 13), (1749100.0, 5450400.0), grid.projection)
    assert grid == expected

    # buffer
    grid = Grid.from_vector(mana_polygons_path, 100, buffer=500)
    expected = Grid(100.0, (32, 27), (1748200.0, 5451600.0), grid.projection)
    assert grid == expected


@requires_pkg("fiona")
def test_grid_from_vector_line():
    # all
    grid = Grid.from_vector(lines_path, 250)
    expected = Grid(250.0, (67, 64), (1803250.0, 5878500.0), grid.projection)
    assert grid == expected

    # filter
    grid = Grid.from_vector(lines_path, 250, {"StreamOrde": 5})
    expected = Grid(250.0, (19, 14), (1808750.0, 5877000.0), grid.projection)
    assert grid == expected

    grid = Grid.from_vector(lines_path, 250, {"StreamOrde": [4, 5]})
    expected = Grid(250.0, (28, 41), (1804750.0, 5877000.0), grid.projection)
    assert grid == expected

    # buffer
    grid = Grid.from_vector(lines_path, 250, buffer=500)
    expected = Grid(250.0, (70, 66), (1803000.0, 5878750.0), grid.projection)
    assert grid == expected


@requires_pkg("fiona")
def test_grid_from_vector_filter_sql_where():
    import fiona

    if fiona.__version__[0:3] < "1.9":
        pytest.skip("Fiona 1.9 or later required to use SQL WHERE")

    # filter
    grid = Grid.from_vector(lines_path, 250, "StreamOrde>=5")
    expected = Grid(250.0, (19, 14), (1808750.0, 5877000.0), grid.projection)
    assert grid == expected
