import pytest

from gridit import Grid
from gridit.classmethods import get_shape_top_left

from .conftest import datadir, requires_pkg

# low-level method


@pytest.mark.parametrize(
    "resolution, snap, top_left",
    [
        (0.2, "full", (42.0, 73.0)),
        (0.2, "half", (41.9, 73.1)),
        (0.2, "top-left", (42.0, 73.0)),
        (0.2, "top-right", (41.8, 73.0)),
        (0.2, "bottom-left", (42.0, 73.2)),
        (0.2, "bottom-right", (41.8, 73.2)),
        (0.2, (1.0, 1.0), (42.0, 73.0)),
        (0.2, (2.0, 2.0), (42.0, 73.0)),
        (1.0, "full", (42.0, 73.0)),
        (1.0, "half", (41.5, 73.5)),
        (1.0, "top-left", (42.0, 73.0)),
        (1.0, "top-right", (41.0, 73.0)),
        (1.0, "bottom-left", (42.0, 74.0)),
        (1.0, "bottom-right", (41.0, 74.0)),
        (1.0, (1.0, 1.0), (42.0, 73.0)),
        (1.0, (2.0, 2.0), (42.0, 73.0)),
        (2.0, "full", (42.0, 74.0)),
        (2.0, "half", (41.0, 73.0)),
        (2.0, "top-left", (42.0, 73.0)),
        (2.0, "top-right", (40.0, 73.0)),
        (2.0, "bottom-left", (42.0, 75.0)),
        (2.0, "bottom-right", (40.0, 75.0)),
        (2.0, (1.0, 1.0), (41.0, 73.0)),
        (2.0, (2.0, 2.0), (42.0, 74.0)),
        (5.0, "full", (40.0, 75.0)),
        (5.0, "half", (37.5, 77.5)),
        (5.0, (1.0, 2.0), (41.0, 77.0)),
        (5.0, (2.8, 2.7), (37.8, 77.7)),
        (10.0, "full", (40.0, 80.0)),
        (20.0, "full", (40.0, 80.0)),
        (33.3, "full", (33.3, 99.9)),
    ],
)
def test_get_shape_top_left_point(resolution, snap, top_left):
    x, y = 42.0, 73.0  # point
    buffer = 0.0
    shape, (minx, maxy) = get_shape_top_left((x, y, x, y), resolution, buffer, snap)
    maxx = minx + resolution
    miny = maxy - resolution
    assert top_left == (minx, maxy)
    assert shape == (1, 1)
    # check if point is within bounds
    assert minx <= x <= maxx
    assert miny <= y <= maxy
    # check snapped edges touch
    if isinstance(snap, str):
        if "top" in snap:
            assert maxy == y
        if "bottom" in snap:
            assert miny == y
        if "left" in snap:
            assert minx == x
        if "right" in snap:
            assert maxx == x
    else:
        # tuple snaps to top left
        snapx, snapy = snap
        assert (snapx % resolution) == pytest.approx(minx % resolution)
        assert (snapy % resolution) == pytest.approx(maxy % resolution)


@pytest.mark.parametrize(
    "resolution, buffer, snap, top_left, shape",
    [
        (0.2, 0.01, "full", (41.8, 73.2), (2, 2)),
        (0.2, 0.01, "half", (41.9, 73.1), (1, 1)),
        (0.2, 0.01, "top-left", (41.99, 73.01), (1, 1)),
        (0.2, 0.01, "top-right", (41.81, 73.01), (1, 1)),
        (0.2, 0.01, "bottom-left", (41.99, 73.19), (1, 1)),
        (0.2, 0.01, "bottom-right", (41.81, 73.19), (1, 1)),
        (0.2, 0.01, (1.0, 1.0), (41.8, 73.2), (2, 2)),
        (0.2, 0.01, (12.9, 12.71), (41.9, 73.11), (1, 1)),
        (1.0, 0.5, "half", (41.5, 73.5), (1, 1)),
        (1.0, 1.0, "full", (41.0, 74.0), (2, 2)),
        (1.0, 1.0, "half", (40.5, 74.5), (3, 3)),
        (1.0, 1.0, "top-left", (41.0, 74.0), (2, 2)),
        (1.0, 1.0, "top-right", (41.0, 74.0), (2, 2)),
        (1.0, 1.0, "bottom-left", (41.0, 74.0), (2, 2)),
        (1.0, 1.0, "bottom-right", (41.0, 74.0), (2, 2)),
        (1.0, 1.0, (-1.0, -1.0), (41.0, 74.0), (2, 2)),
        (1.0, 1.0, (2.0, 2.0), (41.0, 74.0), (2, 2)),
        (2.0, 2.0, "full", (40.0, 76.0), (3, 2)),
        (2.0, 2.0, "half", (39.0, 75.0), (2, 3)),
        (2.0, 2.0, "top-left", (40.0, 75.0), (2, 2)),
        (2.0, 2.0, "top-right", (40.0, 75.0), (2, 2)),
        (2.0, 2.0, "bottom-left", (40.0, 75.0), (2, 2)),
        (2.0, 2.0, "bottom-right", (40.0, 75.0), (2, 2)),
        (2.0, 2.0, (-1.0, -1.0), (39.0, 75.0), (2, 3)),
        (2.0, 2.0, (2022.0, 2222.0), (40.0, 76.0), (3, 2)),
        (33.3, 0.01, "full", (33.3, 99.9), (1, 1)),
        (33.3, 6.7, "full", (33.3, 99.9), (2, 1)),
        (33.3, 9.0, "full", (0.0, 99.9), (2, 2)),
        (33.3, 9.0, "full", (0.0, 99.9), (2, 2)),
    ],
)
def test_get_shape_top_left_buffer(resolution, buffer, snap, top_left, shape):
    x, y = 42.0, 73.0  # point
    (ny, nx), (minx, maxy) = get_shape_top_left((x, y, x, y), resolution, buffer, snap)
    assert top_left == (minx, maxy)
    assert shape == (ny, nx)
    # check if point is within bounds
    maxx = minx + resolution * nx
    miny = maxy - resolution * ny
    assert minx <= x <= maxx
    assert miny <= y <= maxy
    # check snapped edges touch
    if isinstance(snap, str):
        if "top" in snap:
            assert maxy == pytest.approx(y + buffer)
        if "bottom" in snap:
            assert miny == pytest.approx(y - buffer)
        if "left" in snap:
            assert minx == pytest.approx(x - buffer)
        if "right" in snap:
            assert maxx == pytest.approx(x + buffer)
    else:
        # tuple snaps to top left
        snapx, snapy = snap
        assert (snapx % resolution) == pytest.approx(minx % resolution)
        assert (snapy % resolution) == pytest.approx(maxy % resolution)


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
    grid = Grid.from_bbox(
        1748762.8, 5448908.9, 1749509, 5449749, 25, buffer=20, projection="EPSG:2193"
    )
    expected = Grid(25.0, (36, 33), (1748725.0, 5449775.0), "EPSG:2193")
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

    # with snap mode
    grid = Grid.from_raster(mana_dem_path, buffer=16.0, snap="bottom-right")
    assert grid == expected

    # with snap half mode
    grid = Grid.from_raster(mana_dem_path, buffer=16.0, snap="half")
    expected = Grid(8.0, (283, 214), (1748668.0, 5451116.0), grid.projection)
    assert grid == expected

    # with buffer
    grid = Grid.from_raster(mana_dem_path, 10.0, buffer=20.0)
    expected = Grid(10.0, (227, 172), (1748660.0, 5451120.0), grid.projection)
    assert grid == expected

    # with buffer + snap modes
    grid = Grid.from_raster(mana_dem_path, 10.0, buffer=20.0, snap="top-left")
    expected = Grid(10.0, (227, 172), (1748668.0, 5451116.0), grid.projection)
    assert grid == expected

    grid = Grid.from_raster(mana_dem_path, 10.0, buffer=20.0, snap="bottom-left")
    expected = Grid(10.0, (227, 172), (1748668.0, 5451122.0), grid.projection)
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

    # all + half snap mode
    grid = Grid.from_vector(points_path, 250, snap="half")
    expected = Grid(250.0, (49, 27), (1810625.0, 5877625.0), grid.projection)
    assert grid == expected

    # filter
    grid = Grid.from_vector(points_path, 250, filter={"id": [5, 9]})
    expected = Grid(250.0, (19, 7), (1810500.0, 5873750.0), grid.projection)
    assert grid == expected

    grid = Grid.from_vector(points_path, 250, filter={"id": 5})
    expected = Grid(250.0, (1, 1), (1812000.0, 5869250.0), grid.projection)
    assert grid == expected

    # filter + buffer
    grid = Grid.from_vector(points_path, 250, filter={"id": 5}, buffer=240)
    expected = Grid(250.0, (3, 3), (1811750.0, 5869500.0), grid.projection)
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

    # layer + half snap mode
    grid = Grid.from_vector(datadir, 100, layer="mana_polygons", snap="half")
    expected = Grid(100.0, (23, 18), (1748650.0, 5451150.0), grid.projection)
    assert grid == expected

    # filter
    grid = Grid.from_vector(mana_polygons_path, 100, filter={"name": "South-east"})
    expected = Grid(100.0, (14, 13), (1749100.0, 5450400.0), grid.projection)
    assert grid == expected

    # buffer
    grid = Grid.from_vector(mana_polygons_path, 100, buffer=500)
    expected = Grid(100.0, (34, 28), (1748100.0, 5451700.0), grid.projection)
    assert grid == expected


@requires_pkg("fiona")
def test_grid_from_vector_line():
    # all
    grid = Grid.from_vector(lines_path, 250)
    expected = Grid(250.0, (67, 64), (1803250.0, 5878500.0), grid.projection)
    assert grid == expected

    # filter
    grid = Grid.from_vector(lines_path, 250, filter={"StreamOrde": 5})
    expected = Grid(250.0, (19, 14), (1808750.0, 5877000.0), grid.projection)
    assert grid == expected

    grid = Grid.from_vector(lines_path, 250, filter={"StreamOrde": [4, 5]})
    expected = Grid(250.0, (28, 41), (1804750.0, 5877000.0), grid.projection)
    assert grid == expected

    # buffer
    grid = Grid.from_vector(lines_path, 250, buffer=500)
    expected = Grid(250.0, (71, 68), (1802750.0, 5879000.0), grid.projection)
    assert grid == expected

    # buffer + half snap mode
    grid = Grid.from_vector(lines_path, 250, buffer=500, snap="half")
    expected = Grid(250.0, (71, 67), (1802875.0, 5878875.0), grid.projection)
    assert grid == expected


@requires_pkg("fiona")
def test_grid_from_vector_filter_sql_where():
    import fiona

    if fiona.__version__[0:3] < "1.9":
        pytest.skip("Fiona 1.9 or later required to use SQL WHERE")

    # filter
    grid = Grid.from_vector(lines_path, 250, filter="StreamOrde>=5")
    expected = Grid(250.0, (19, 14), (1808750.0, 5877000.0), grid.projection)
    assert grid == expected
