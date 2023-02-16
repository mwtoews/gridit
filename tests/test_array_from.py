import logging
import numpy as np
import pytest
from hashlib import md5

from .conftest import datadir, has_pkg, requires_pkg

if has_pkg("rasterio"):
    import rasterio

from gridit import Grid


mana_dem_path = datadir / "Mana.tif"
mana_polygons_path = datadir / "Mana_polygons.shp"
mana_hk_nan_path = datadir / "Mana_hk_nan.tif"


@requires_pkg("rasterio")
def test_array_from_array(caplog):
    coarse_grid = Grid(8, (3, 4))
    fine_grid = Grid(4, (6, 8))
    # same resolution
    in_ar = np.arange(12).reshape((3, 4))
    with caplog.at_level(logging.INFO):
        out_ar = coarse_grid.array_from_array(coarse_grid, in_ar)
        assert "nearest resampling" in caplog.messages[-1]
    np.testing.assert_array_equal(out_ar, in_ar)
    # fine to coarse
    in_ar = np.arange(12).reshape((3, 4))
    with caplog.at_level(logging.INFO):
        out_ar = fine_grid.array_from_array(coarse_grid, in_ar)
        assert "nearest resampling" in caplog.messages[-1]
    np.testing.assert_array_equal(
        out_ar,
        np.ma.array([
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 0, 1, 1, 2, 2, 3, 3],
            [4, 4, 5, 5, 6, 6, 7, 7],
            [4, 4, 5, 5, 6, 6, 7, 7],
            [8, 8, 9, 9, 10, 10, 11, 11],
            [8, 8, 9, 9, 10, 10, 11, 11]]))
    with caplog.at_level(logging.INFO):
        out_ar = fine_grid.array_from_array(coarse_grid, in_ar.astype(float))
        assert "bilinear resampling" in caplog.messages[-1]
    np.testing.assert_array_equal(
        out_ar,
        np.ma.array([
            [0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.0],
            [1.0, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.0],
            [3.0, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.0],
            [5.0, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.0],
            [7.0, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.0],
            [8.0, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.0]]))
    # coarse to fine
    in_ar = np.arange(48).reshape((6, 8))
    with caplog.at_level(logging.INFO):
        out_ar = coarse_grid.array_from_array(fine_grid, in_ar)
        assert "mode resampling" in caplog.messages[-1]
    np.testing.assert_array_equal(
        out_ar,
        np.ma.array([
            [0, 2, 4, 6],
            [16, 18, 20, 22],
            [32, 34, 36, 38]]))
    with caplog.at_level(logging.INFO):
        out_ar = coarse_grid.array_from_array(fine_grid, in_ar.astype(float))
        assert "average resampling" in caplog.messages[-1]
    np.testing.assert_array_equal(
        out_ar,
        np.ma.array([
            [4.5, 6.5, 8.5, 10.5],
            [20.5, 22.5, 24.5, 26.5],
            [36.5, 38.5, 40.5, 42.5]]))
    # 3D fine to coarse
    R, C = np.mgrid[0:3, 0:4]
    in_ar = np.stack([R, C])
    with caplog.at_level(logging.INFO):
        out_ar = fine_grid.array_from_array(coarse_grid, in_ar)
        assert "nearest resampling" in caplog.messages[-1]
    np.testing.assert_array_equal(
        out_ar,
        np.ma.array([
            [[0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2]],
            [[0, 0, 1, 1, 2, 2, 3, 3],
             [0, 0, 1, 1, 2, 2, 3, 3],
             [0, 0, 1, 1, 2, 2, 3, 3],
             [0, 0, 1, 1, 2, 2, 3, 3],
             [0, 0, 1, 1, 2, 2, 3, 3],
             [0, 0, 1, 1, 2, 2, 3, 3]]]))
    # errors
    with pytest.raises(TypeError, match="expected grid to be a Grid"):
        fine_grid.array_from_array(1, in_ar)
    with pytest.raises(TypeError, match="expected array to be array_like"):
        fine_grid.array_from_array(coarse_grid, 1)
    with pytest.raises(ValueError, match="array has different shape than gri"):
        fine_grid.array_from_array(coarse_grid, np.ones((2, 3)))
    with pytest.raises(ValueError, match="array has different shape than gri"):
        fine_grid.array_from_array(coarse_grid, np.ones((4, 2, 3)))


@requires_pkg("rasterio")
def test_array_from_raster_all():
    grid = Grid(100, (24, 18), (1748600.0, 5451200.0))
    ar = grid.array_from_raster(mana_dem_path)
    assert ar.shape == (24, 18)
    assert ar.dtype == "float32"
    # there are a few different possiblities, depending on GDAL version
    mask_hash = md5(ar.mask.tobytes()).hexdigest()[:7]
    if mask_hash == "":  # todo: this was an older version?
        assert ar.mask.sum() == 170
        np.testing.assert_almost_equal(ar.min(), 1.833, 3)
        np.testing.assert_almost_equal(ar.max(), 115.471, 3)
    elif mask_hash == "d44fae9":
        assert ar.mask.sum() == 182
        np.testing.assert_almost_equal(ar.min(), 2.521, 3)
        np.testing.assert_almost_equal(ar.max(), 115.688, 3)
    elif mask_hash == "9f8b542":
        assert ar.mask.sum() == 170
        np.testing.assert_almost_equal(ar.min(), 1.810, 3)
        np.testing.assert_almost_equal(ar.max(), 115.688, 3)
    else:
        raise AssertionError((mask_hash, ar.mask.sum()))


@requires_pkg("rasterio")
def test_array_from_raster_filter():
    grid = Grid(100, (14, 13), (1749100.0, 5450400.0))
    ar = grid.array_from_raster(mana_dem_path)
    assert ar.shape == (14, 13)
    assert ar.dtype == "float32"
    # there are a few different possiblities, depending on GDAL version
    mask_hash = md5(ar.mask.tobytes()).hexdigest()[:7]
    if mask_hash == "":  # todo: this was an older version?
        assert ar.mask.sum() == 32
        np.testing.assert_almost_equal(ar.min(), 1.833, 3)
        np.testing.assert_almost_equal(ar.max(), 101.613, 3)
    elif mask_hash == "c408a2a":
        assert ar.mask.sum() == 36
        np.testing.assert_almost_equal(ar.min(), 2.521, 3)
        np.testing.assert_almost_equal(ar.max(), 101.692, 3)
    elif mask_hash == "95d7608":
        assert ar.mask.sum() == 34
        np.testing.assert_almost_equal(ar.min(), 1.810, 3)
        np.testing.assert_almost_equal(ar.max(), 101.692, 3)
    else:
        raise AssertionError((mask_hash, ar.mask.sum()))


@requires_pkg("rasterio")
def test_array_from_raster_filter_nan():
    grid = Grid(100, (14, 13), (1749100.0, 5450400.0))
    ar = grid.array_from_raster(mana_hk_nan_path)
    assert ar.shape == (14, 13)
    assert ar.dtype == "float32"
    # there are a few different possiblities, depending on GDAL version
    mask_hash = md5(ar.mask.tobytes()).hexdigest()[:7]
    if mask_hash == "4071d94":
        assert ar.mask.sum() == 32
        assert np.isnan(ar.data).sum() == 32
    elif mask_hash == "bb113c4":
        assert ar.mask.sum() == 29
        assert np.isnan(ar.data).sum() == 29
    else:
        raise AssertionError((mask_hash, ar.mask.sum()))
    np.testing.assert_almost_equal(ar.min(), 0.012, 3)
    np.testing.assert_almost_equal(ar.max(), 12.3, 3)
    assert np.isnan(ar.fill_value)


@pytest.fixture
def grid_from_raster():
    return Grid.from_raster(mana_dem_path)


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


@requires_pkg("rasterio")
def test_array_from_raster_refine():
    # use bilinear resampling method
    grid = Grid(5, (254, 244), (1749120.0, 5450360.0))
    ar = grid.array_from_raster(mana_dem_path)
    assert ar.shape == (254, 244)
    assert ar.dtype == "float32"
    assert ar.mask.sum() == 12802
    np.testing.assert_almost_equal(ar.min(), 1.268, 3)
    np.testing.assert_almost_equal(ar.max(), 103.592, 3)


@requires_pkg("rasterio")
def test_array_from_raster_refine_nan():
    # use bilinear resampling method
    grid = Grid(5, (254, 244), (1749120.0, 5450360.0))
    ar = grid.array_from_raster(mana_hk_nan_path)
    assert ar.shape == (254, 244)
    assert ar.dtype == "float32"
    assert ar.mask.sum() == 9728
    assert np.isnan(ar.fill_value)
    assert np.isnan(ar.data).sum() == 9728
    np.testing.assert_almost_equal(ar.min(), 0.012, 3)
    np.testing.assert_almost_equal(ar.max(), 12.3, 3)


@pytest.fixture
def grid_from_vector_all():
    return Grid.from_vector(mana_polygons_path, 100)


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
    ar = grid_from_vector_all.array_from_vector(
        mana_polygons_path, "K_m_d", all_touched=True)
    assert ar.mask.sum() == 153
    np.testing.assert_almost_equal(ar.min(), 0.00012)
    np.testing.assert_almost_equal(ar.max(), 12.3)
    assert len(np.unique(ar)) == 5


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_refine_2(grid_from_vector_all):
    ar = grid_from_vector_all.array_from_vector(
        mana_polygons_path, "K_m_d", fill=1e10, refine=2)
    assert ar.shape == (24, 18)
    assert np.issubdtype(ar.dtype, np.floating)
    assert ar.fill_value == 1e10
    assert ar.mask.sum() == 175
    np.testing.assert_almost_equal(ar.min(), 0.00012)
    np.testing.assert_almost_equal(ar.max(), 12.3)
    assert len(np.unique(ar)) == 18
    ar = grid_from_vector_all.array_from_vector(
        mana_polygons_path, "K_m_d", fill=1e10, refine=2, all_touched=True)
    assert ar.mask.sum() == 153
    np.testing.assert_almost_equal(ar.min(), 0.00012)
    np.testing.assert_almost_equal(ar.max(), 12.3)
    assert len(np.unique(ar)) == 18


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_refine_5(grid_from_vector_all):
    ar = grid_from_vector_all.array_from_vector(
        mana_polygons_path, "K_m_d", fill=1e10, refine=5)
    assert ar.shape == (24, 18)
    assert np.issubdtype(ar.dtype, np.floating)
    assert ar.mask.sum() == 165
    np.testing.assert_almost_equal(ar.min(), 0.00012)
    np.testing.assert_almost_equal(ar.max(), 12.3)
    assert len(np.unique(ar)) == 47
    ar = grid_from_vector_all.array_from_vector(
        mana_polygons_path, "K_m_d", fill=1e10, refine=5, all_touched=True)
    assert ar.mask.sum() == 153
    np.testing.assert_almost_equal(ar.min(), 0.00012)
    np.testing.assert_almost_equal(ar.max(), 12.3)
    assert len(np.unique(ar)) == 44


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
    ar = grid_from_vector_all.array_from_vector(
        datadir, "K_m_d", layer="mana_polygons", all_touched=True)
    assert ar.mask.sum() == 153
    np.testing.assert_almost_equal(ar.min(), 0.00012)
    np.testing.assert_almost_equal(ar.max(), 12.3)
    assert len(np.unique(ar)) == 5


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_layer_intnull(grid_from_vector_all):
    ar = grid_from_vector_all.array_from_vector(
        datadir, "intnull", layer="mana_polygons")
    assert ar.shape == (24, 18)
    assert ar.fill_value == 0
    assert np.issubdtype(ar.dtype, np.integer)
    assert ar.mask.sum() == 228
    assert ar.min() == 4
    assert ar.max() == 51
    assert ar.sum() == 3487
    ar = grid_from_vector_all.array_from_vector(
        datadir, "intnull", layer="mana_polygons", all_touched=True)
    assert np.issubdtype(ar.dtype, np.integer)
    assert ar.mask.sum() == 181
    assert ar.min() == 4
    assert ar.max() == 51
    assert ar.sum() == 5072


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_layer_floatnull(grid_from_vector_all):
    ar = grid_from_vector_all.array_from_vector(
        datadir, "floatnull", layer="mana_polygons")
    assert ar.shape == (24, 18)
    assert ar.fill_value == 0.0
    assert np.issubdtype(ar.dtype, np.floating)
    assert ar.mask.sum() == 228
    np.testing.assert_almost_equal(ar.min(), 0.002)
    np.testing.assert_almost_equal(ar.max(), 2452.0)
    np.testing.assert_almost_equal(ar.sum(), 126963.862)
    ar = grid_from_vector_all.array_from_vector(
        datadir, "floatnull", layer="mana_polygons", all_touched=True)
    assert ar.mask.sum() == 181
    np.testing.assert_almost_equal(ar.min(), 0.002)
    np.testing.assert_almost_equal(ar.max(), 2452.0)
    np.testing.assert_almost_equal(ar.sum(), 193418.014)


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_layer_allnull(grid_from_vector_all):
    ar = grid_from_vector_all.array_from_vector(
        datadir, "allnull", layer="mana_polygons")
    assert ar.shape == (24, 18)
    assert ar.fill_value == 0
    assert np.issubdtype(ar.dtype, np.integer)
    assert ar.mask.all()
    assert ar.data.min() == ar.data.max()
    ar = grid_from_vector_all.array_from_vector(
        datadir, "allnull", layer="mana_polygons", all_touched=True)
    assert ar.mask.all()
    assert ar.data.min() == ar.data.max()


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
    ar = grid.array_from_vector(mana_polygons_path, "K_m_d", all_touched=True)
    assert ar.mask.sum() == 128


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
    ar = grid.array_from_vector(mana_polygons_path, "K_m_d", all_touched=True)
    assert ar.mask.sum() == 128


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_different_projection():
    grid = Grid.from_bbox(
        19455906, -5026598, 19457499, -5024760, 25, projection="EPSG:3857")
    assert grid.projection == "EPSG:3857"
    ar = grid.array_from_vector(mana_polygons_path, "K_m_d")
    assert ar.shape == (74, 64)
    assert ar.mask.sum() == 950
    ar = grid.array_from_vector(mana_polygons_path, "K_m_d", all_touched=True)
    assert ar.mask.sum() == 873


@requires_pkg("fiona", "rasterio")
def test_mask_from_vector_layer(grid_from_vector_all):
    mask = grid_from_vector_all.mask_from_vector(datadir, "mana_polygons")
    assert mask.shape == (24, 18)
    assert mask.dtype == "bool"
    assert mask.sum() == 193


@requires_pkg("rasterio")
def test_mask_from_raster(grid_from_raster):
    mask = grid_from_raster.mask_from_raster(mana_dem_path)
    assert mask.shape == (278, 209)
    assert mask.dtype == "bool"
    assert mask.sum() == 23782


@requires_pkg("fiona", "rasterio")
def test_mask_from_vector_all(grid_from_vector_all):
    mask = grid_from_vector_all.mask_from_vector(mana_polygons_path)
    assert mask.shape == (24, 18)
    assert mask.dtype == "bool"
    assert mask.sum() == 193
