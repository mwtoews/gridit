import numpy as np
import pytest

from .conftest import datadir, has_pkg, requires_pkg

if has_pkg("rasterio"):
    import rasterio

from gridit import Grid


mana_dem_path = datadir / "Mana.tif"
mana_polygons_path = datadir / "Mana_polygons.shp"
mana_hk_nan_path = datadir / "Mana_hk_nan.tif"


@pytest.fixture
def grid_basic():
    return Grid(10, (20, 30), (1000.0, 2000.0))


def test_grid_basic(grid_basic):
    grid = grid_basic
    assert isinstance(grid, Grid)
    assert grid.resolution == 10.0
    assert grid.shape == (20, 30)
    assert grid.top_left == (1000.0, 2000.0)
    assert grid.projection == ""


def test_grid_dict(grid_basic):
    grid_d = dict(grid_basic)
    assert list(grid_d.keys()) == \
        ["resolution", "shape", "top_left", "projection"]
    assert grid_d["resolution"] == 10.0
    assert grid_d["shape"] == (20, 30)
    assert grid_d["top_left"] == (1000.0, 2000.0)
    assert grid_d["projection"] == ""


def test_grid_repr(grid_basic):
    expected = \
        "<Grid: resolution=10.0, shape=(20, 30), "\
        "top_left=(1000.0, 2000.0) />"
    assert repr(grid_basic) == expected
    assert str(grid_basic) == expected


def test_grid_eq_hash():
    grid1 = Grid(10, (20, 30), (1000.0, 2000.0))
    grid2 = Grid(10, (20, 30), (1001.0, 2000.0))
    grid3 = Grid(10, (20, 30), (1000.0, 2000.0), "EPSG:2193")
    grid4 = Grid(10, (20, 30), (1000.0, 2000.0))
    assert grid1 != grid2
    assert grid1 != grid3  # projection is different
    assert grid1 == grid4
    hash1 = hash(grid1)
    hash2 = hash(grid2)
    hash3 = hash(grid3)
    hash4 = hash(grid4)
    assert hash1 != hash2
    assert hash1 != hash3  # projection is different
    assert hash1 == hash4


def test_grid_bounds(grid_basic):
    assert grid_basic.bounds == (1000.0, 1800.0, 1300.0, 2000.0)


@requires_pkg("affine")
def test_grid_transform(grid_basic):
    from affine import Affine

    assert grid_basic.transform == \
        Affine(10.0, 0.0, 1000.0,
               0.0, -10.0, 2000.0)


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


@requires_pkg("fiona", "rasterio")
def test_array_from_raster_all(grid_from_vector_all):
    ar = grid_from_vector_all.array_from_raster(mana_dem_path)
    assert ar.shape == (24, 18)
    assert ar.dtype == "float32"
    # there are a few different possiblities, depending on GDAL version
    mask_sum = ar.mask.sum()
    if mask_sum == 170:
        np.testing.assert_almost_equal(ar.min(), 1.833, 3)
        np.testing.assert_almost_equal(ar.max(), 115.471, 3)
    elif mask_sum == 182:
        np.testing.assert_almost_equal(ar.min(), 2.521, 3)
        np.testing.assert_almost_equal(ar.max(), 115.688, 3)
    else:
        assert mask_sum is False


@requires_pkg("fiona", "rasterio")
def test_array_from_raster_filter(grid_from_vector_filter):
    ar = grid_from_vector_filter.array_from_raster(mana_dem_path)
    assert ar.shape == (14, 13)
    assert ar.dtype == "float32"
    # there are a few different possiblities, depending on GDAL version
    mask_sum = ar.mask.sum()
    if mask_sum == 34:
        np.testing.assert_almost_equal(ar.min(), 1.833, 3)
        np.testing.assert_almost_equal(ar.max(), 101.613, 3)
    elif mask_sum == 36:
        np.testing.assert_almost_equal(ar.min(), 2.521, 3)
        np.testing.assert_almost_equal(ar.max(), 101.692, 3)
    else:
        assert mask_sum is False


@requires_pkg("fiona", "rasterio")
def test_array_from_raster_filter_nan(grid_from_vector_filter):
    ar = grid_from_vector_filter.array_from_raster(mana_hk_nan_path)
    assert ar.shape == (14, 13)
    assert ar.dtype == "float32"
    assert ar.mask.sum() == 32
    assert np.isnan(ar.fill_value)
    assert np.isnan(ar.data).sum() == 32
    np.testing.assert_almost_equal(ar.min(), 0.012, 3)
    np.testing.assert_almost_equal(ar.max(), 12.3, 3)


@requires_pkg("rasterio")
def test_array_from_raster_same_grid(grid_from_raster):
    ar = grid_from_raster.array_from_raster(mana_dem_path)
    assert ar.shape == (278, 209)
    assert ar.dtype == "float32"
    assert ar.mask.sum() == 23782
    with rasterio.open(mana_dem_path, "r") as ds:
        expected = ds.read(1, masked=True)
    np.testing.assert_equal(ar, expected)


@requires_pkg("rasterio")
def test_array_from_raster_same_grid_nan(grid_from_raster):
    ar = grid_from_raster.array_from_raster(mana_hk_nan_path)
    assert ar.shape == (278, 209)
    assert ar.dtype == "float32"
    assert ar.mask.sum() == 21151
    assert np.isnan(ar.fill_value)
    assert np.isnan(ar.data).sum() == 21151
    with rasterio.open(mana_hk_nan_path, "r") as ds:
        expected = ds.read(1, masked=True)
    np.testing.assert_equal(ar, expected)


@requires_pkg("fiona", "rasterio")
def test_array_from_raster_refine():
    # use bilinear resampling method
    grid = Grid.from_vector(mana_polygons_path, 5, {"name": "South-east"})
    ar = grid.array_from_raster(mana_dem_path)
    assert ar.shape == (254, 244)
    assert ar.dtype == "float32"
    assert ar.mask.sum() == 12802
    np.testing.assert_almost_equal(ar.min(), 1.268, 3)
    np.testing.assert_almost_equal(ar.max(), 103.592, 3)


@requires_pkg("fiona", "rasterio")
def test_array_from_raster_refine_nan():
    # use bilinear resampling method
    grid = Grid.from_vector(mana_polygons_path, 5, {"name": "South-east"})
    ar = grid.array_from_raster(mana_hk_nan_path)
    assert ar.shape == (254, 244)
    assert ar.dtype == "float32"
    assert ar.mask.sum() == 9728
    assert np.isnan(ar.fill_value)
    assert np.isnan(ar.data).sum() == 9728
    np.testing.assert_almost_equal(ar.min(), 0.012, 3)
    np.testing.assert_almost_equal(ar.max(), 12.3, 3)


@requires_pkg("fiona", "rasterio")
def test_array_from_vector(grid_from_vector_all):
    ar = grid_from_vector_all.array_from_vector(mana_polygons_path, "K_m_d")
    assert ar.shape == (24, 18)
    assert ar.fill_value == 0.0
    assert np.issubdtype(ar.dtype, np.floating)
    assert ar.mask.sum() == 193
    np.testing.assert_almost_equal(ar.min(), 0.00012)
    np.testing.assert_almost_equal(ar.max(), 12.3)
    assert len(np.unique(ar)) == 5


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_refine_2(grid_from_vector_all):
    ar = grid_from_vector_all.array_from_vector(
        mana_polygons_path, "K_m_d", 1e10, refine=2)
    assert ar.shape == (24, 18)
    assert np.issubdtype(ar.dtype, np.floating)
    assert ar.fill_value == 1e10
    assert ar.mask.sum() == 175
    np.testing.assert_almost_equal(ar.min(), 0.00012)
    np.testing.assert_almost_equal(ar.max(), 12.3)
    assert len(np.unique(ar)) == 18


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_refine_5(grid_from_vector_all):
    ar = grid_from_vector_all.array_from_vector(
        mana_polygons_path, "K_m_d", 1e10, refine=5)
    assert ar.shape == (24, 18)
    assert np.issubdtype(ar.dtype, np.floating)
    assert ar.mask.sum() == 165
    np.testing.assert_almost_equal(ar.min(), 0.00012)
    np.testing.assert_almost_equal(ar.max(), 12.3)
    assert len(np.unique(ar)) == 47


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_layer(grid_from_vector_all):
    ar = grid_from_vector_all.array_from_vector(
        datadir, "K_m_d", layer="mana_polygons")
    assert ar.shape == (24, 18)
    assert ar.fill_value == 0.0
    assert np.issubdtype(ar.dtype, np.floating)
    assert ar.mask.sum() == 193
    np.testing.assert_almost_equal(ar.min(), 0.00012)
    np.testing.assert_almost_equal(ar.max(), 12.3)
    assert len(np.unique(ar)) == 5


@requires_pkg("rasterio")
def test_array_from_raster_no_projection():
    grid = Grid.from_bbox(
        1748762.8, 5448908.9, 1749509, 5449749, 25)
    assert grid.projection == ""
    ar = grid.array_from_raster(mana_dem_path)
    assert ar.shape == (34, 31)
    assert ar.mask.sum() == 160


@requires_pkg("rasterio")
def test_array_from_raster_same_projection():
    grid = Grid.from_bbox(
        1748762.8, 5448908.9, 1749509, 5449749, 25, projection="EPSG:2193")
    assert grid.projection == "EPSG:2193"
    ar = grid.array_from_raster(mana_dem_path)
    assert ar.shape == (34, 31)
    assert ar.mask.sum() == 160


@requires_pkg("rasterio")
def test_array_from_raster_different_projection():
    grid = Grid.from_bbox(
        19455906, -5026598, 19457499, -5024760, 25, projection="EPSG:3857")
    assert grid.projection == "EPSG:3857"
    ar = grid.array_from_raster(mana_dem_path)
    assert ar.shape == (74, 64)
    assert ar.mask.sum() == 1077


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_no_projection():
    grid = Grid.from_bbox(
        1748762.8, 5448908.9, 1749509, 5449749, 25)
    assert grid.projection == ""
    ar = grid.array_from_vector(mana_polygons_path, "K_m_d")
    assert ar.shape == (34, 31)
    assert ar.mask.sum() == 146


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_same_projection():
    # TODO: EPSG:2193 != tests/data/Mana_polygons.prj due to axis order
    projection = mana_polygons_path.with_suffix(".prj").read_text().strip()
    grid = Grid.from_bbox(
        1748762.8, 5448908.9, 1749509, 5449749, 25, projection=projection)
    assert grid.projection == projection
    ar = grid.array_from_vector(mana_polygons_path, "K_m_d")
    assert ar.shape == (34, 31)
    assert ar.mask.sum() == 146


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_different_projection():
    grid = Grid.from_bbox(
        19455906, -5026598, 19457499, -5024760, 25, projection="EPSG:3857")
    assert grid.projection == "EPSG:3857"
    ar = grid.array_from_vector(mana_polygons_path, "K_m_d")
    assert ar.shape == (74, 64)
    assert ar.mask.sum() == 950
