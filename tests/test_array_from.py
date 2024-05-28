import logging
from hashlib import md5

import numpy as np
import pytest

from .conftest import datadir, has_pkg, outdir, requires_pkg

if has_pkg("rasterio"):
    import rasterio
    from rasterio import gdal_version

from gridit import Grid

mana_dem_path = datadir / "Mana.tif"
mana_polygons_path = datadir / "Mana_polygons.shp"
mana_hk_nan_path = datadir / "Mana_hk_nan.tif"
lines_path = datadir / "waitaku2_lines.shp"
points_path = datadir / "waitaku2_points.shp"
nocrs_path = datadir / "nocrs.tif"


@requires_pkg("rasterio")
def test_array_from_array(caplog):
    from rasterio.enums import Resampling

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
        np.ma.array(
            [
                [0, 0, 1, 1, 2, 2, 3, 3],
                [0, 0, 1, 1, 2, 2, 3, 3],
                [4, 4, 5, 5, 6, 6, 7, 7],
                [4, 4, 5, 5, 6, 6, 7, 7],
                [8, 8, 9, 9, 10, 10, 11, 11],
                [8, 8, 9, 9, 10, 10, 11, 11],
            ]
        ),
    )
    with caplog.at_level(logging.INFO):
        out_ar = fine_grid.array_from_array(coarse_grid, in_ar.astype(float))
        assert "bilinear resampling" in caplog.messages[-1]
    np.testing.assert_array_equal(
        out_ar,
        np.ma.array(
            [
                [0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.0],
                [1.0, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.0],
                [3.0, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.0],
                [5.0, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.0],
                [7.0, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.0],
                [8.0, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.0],
            ]
        ),
    )
    # specify resampling method
    with caplog.at_level(logging.INFO):
        out_ar = fine_grid.array_from_array(
            coarse_grid, in_ar.astype(float), resampling="nearest"
        )
        assert "nearest resampling" in caplog.messages[-1]
    with caplog.at_level(logging.INFO):
        out_ar = fine_grid.array_from_array(
            coarse_grid, in_ar.astype(float), resampling=Resampling.nearest
        )
        assert "nearest resampling" in caplog.messages[-1]
    # coarse to fine
    in_ar = np.arange(48).reshape((6, 8))
    with caplog.at_level(logging.INFO):
        out_ar = coarse_grid.array_from_array(fine_grid, in_ar)
        assert "mode resampling" in caplog.messages[-1]
    np.testing.assert_array_equal(
        out_ar, np.ma.array([[0, 2, 4, 6], [16, 18, 20, 22], [32, 34, 36, 38]])
    )
    with caplog.at_level(logging.INFO):
        out_ar = coarse_grid.array_from_array(fine_grid, in_ar.astype(float))
        assert "average resampling" in caplog.messages[-1]
    np.testing.assert_array_equal(
        out_ar,
        np.ma.array(
            [[4.5, 6.5, 8.5, 10.5], [20.5, 22.5, 24.5, 26.5], [36.5, 38.5, 40.5, 42.5]]
        ),
    )
    # 3D fine to coarse
    R, C = np.mgrid[0:3, 0:4]
    in_ar = np.stack([R, C])
    with caplog.at_level(logging.INFO):
        out_ar = fine_grid.array_from_array(coarse_grid, in_ar)
        assert "nearest resampling" in caplog.messages[-1]
    np.testing.assert_array_equal(
        out_ar,
        np.ma.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                ],
                [
                    [0, 0, 1, 1, 2, 2, 3, 3],
                    [0, 0, 1, 1, 2, 2, 3, 3],
                    [0, 0, 1, 1, 2, 2, 3, 3],
                    [0, 0, 1, 1, 2, 2, 3, 3],
                    [0, 0, 1, 1, 2, 2, 3, 3],
                    [0, 0, 1, 1, 2, 2, 3, 3],
                ],
            ]
        ),
    )
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
def test_array_from_raster_all(write_files):
    grid = Grid(100, (24, 18), (1748600.0, 5451200.0))
    ar = grid.array_from_raster(mana_dem_path)
    assert ar.shape == (24, 18)
    assert ar.dtype == "float32"
    # there are a few different possiblities, depending on GDAL version
    hash = md5(ar.tobytes()).hexdigest()[:7]
    if write_files:
        fname = f"test_array_from_raster_all_{gdal_version()}_{hash}.tif"
        grid.write_raster(ar, outdir / fname)
    if hash == "5b6815e":
        assert ar.mask.sum() == 182
        np.testing.assert_almost_equal(ar.min(), 2.521, 3)
        np.testing.assert_almost_equal(ar.max(), 115.688, 3)
    elif hash == "b84b6ef":
        assert ar.mask.sum() == 170
        np.testing.assert_almost_equal(ar.min(), 1.810, 3)
        np.testing.assert_almost_equal(ar.max(), 115.688, 3)
    elif hash == "cd4c1ce":
        assert ar.mask.sum() == 171
        np.testing.assert_almost_equal(ar.min(), 1.810, 3)
        np.testing.assert_almost_equal(ar.max(), 115.688, 3)
    else:
        raise AssertionError((hash, ar.mask.sum()))


@requires_pkg("rasterio")
def test_array_from_raster_filter(write_files):
    grid = Grid(100, (14, 13), (1749100.0, 5450400.0))
    ar = grid.array_from_raster(mana_dem_path)
    assert ar.shape == (14, 13)
    assert ar.dtype == "float32"
    # there are a few different possiblities, depending on GDAL version
    hash = md5(ar.tobytes()).hexdigest()[:7]
    if write_files:
        fname = f"test_array_from_raster_filter_{gdal_version()}_{hash}.tif"
        grid.write_raster(ar, outdir / fname)
    if hash == "d7b8da4":
        assert ar.mask.sum() == 36
        np.testing.assert_almost_equal(ar.min(), 2.521, 3)
        np.testing.assert_almost_equal(ar.max(), 101.692, 3)
    elif hash == "0e8711a":
        assert ar.mask.sum() == 34
        np.testing.assert_almost_equal(ar.min(), 1.810, 3)
        np.testing.assert_almost_equal(ar.max(), 101.692, 3)
    else:
        raise AssertionError((hash, ar.mask.sum()))


@requires_pkg("rasterio")
def test_array_from_raster_filter_nan(write_files):
    grid = Grid(100, (14, 13), (1749100.0, 5450400.0))
    ar = grid.array_from_raster(mana_hk_nan_path)
    assert ar.shape == (14, 13)
    assert ar.dtype == "float32"
    # there are a few different possiblities, depending on GDAL version
    hash = md5(ar.tobytes()).hexdigest()[:7]
    if write_files:
        fname = f"test_array_from_raster_filter_nan_{gdal_version()}_{hash}.tif"
        grid.write_raster(ar, outdir / fname)
    if hash == "ae56822":
        assert ar.mask.sum() == 32
        assert np.isnan(ar.data).sum() == 32
    elif hash == "b8da817":
        assert ar.mask.sum() == 29
        assert np.isnan(ar.data).sum() == 29
    else:
        raise AssertionError((hash, ar.mask.sum()))
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
def test_array_from_raster_refine(caplog):
    from rasterio.enums import Resampling

    # use bilinear resampling method
    grid = Grid(5, (254, 244), (1749120.0, 5450360.0))
    with caplog.at_level(logging.INFO):
        ar = grid.array_from_raster(mana_dem_path)
        assert "bilinear resampling" in caplog.messages[2]
    assert ar.shape == (254, 244)
    assert ar.dtype == "float32"
    assert ar.mask.sum() == 12802
    np.testing.assert_almost_equal(ar.min(), 1.268, 3)
    np.testing.assert_almost_equal(ar.max(), 103.592, 3)
    # specify resampling method via str or enum
    caplog.clear()
    with caplog.at_level(logging.INFO):
        ar = grid.array_from_raster(mana_dem_path, resampling="nearest")
        assert "nearest resampling" in caplog.messages[2]
    np.testing.assert_almost_equal(ar.min(), 1.2556334, 3)
    np.testing.assert_almost_equal(ar.max(), 103.65819, 3)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        ar = grid.array_from_raster(mana_dem_path, resampling=Resampling.bilinear)
        assert "bilinear resampling" in caplog.messages[2]
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
@pytest.mark.parametrize("attribute", [None, "K_m_d"])
@pytest.mark.parametrize("refine", [None, 1, 2, 5])
@pytest.mark.parametrize("all_touched", [False, True])
def test_array_from_vector(
    write_files, caplog, grid_from_vector_all, attribute, refine, all_touched
):
    with caplog.at_level(logging.INFO):
        ar = grid_from_vector_all.array_from_vector(
            mana_polygons_path,
            attribute=attribute,
            refine=refine,
            all_touched=all_touched,
        )
        if refine is None:
            # re-define default if None
            refine = 1 if attribute is None else 5
            assert (
                sum(
                    f"selecting default refine={refine} for Polygon" in msg
                    for msg in caplog.messages
                )
                == 1
            ), caplog.messages
        else:
            assert (
                sum("selecting default" in msg for msg in caplog.messages) == 0
            ), caplog.messages
            assert (
                sum(
                    f"using refine={refine} for Polygon" in msg
                    for msg in caplog.messages
                )
                == 1
            ), caplog.messages
    if write_files:
        hash = md5(ar.tobytes()).hexdigest()[:7]
        fname = f"test_array_from_vector_{attribute}_{refine}_"
        if all_touched:
            fname += "all_touched_"
        fname += f"{gdal_version()}_{hash}.tif"
        grid_from_vector_all.write_raster(ar, outdir / fname)
    assert ar.shape == (24, 18)
    assert np.ma.isMaskedArray(ar)
    if attribute is None:
        assert np.issubdtype(ar.dtype, np.integer)
        assert ar.fill_value == 0
        assert set(np.unique(ar).filled()) == {0, 1}
    else:
        assert np.issubdtype(ar.dtype, np.floating)
        assert ar.fill_value == 0.0
        assert ar.min() == pytest.approx(0.00012)
        assert ar.max() == pytest.approx(12.3)
        num_unique = len(np.unique(ar))
        if refine == 1:
            assert num_unique == 5
        elif refine == 2:
            assert num_unique in (17, 18)
        elif refine == 5:
            if all_touched:
                assert num_unique in (44, 36)
            else:
                assert num_unique in (47, 38)
        else:
            raise KeyError(refine)
    if all_touched is False:
        expected_ar_mask_sum = {1: 193, 2: 175, 5: 165}[refine]
    else:
        expected_ar_mask_sum = 153
    assert ar.mask.sum() == expected_ar_mask_sum


@requires_pkg("fiona", "rasterio")
@pytest.mark.parametrize("refine", [None, 1, 2, 5])
@pytest.mark.parametrize("all_touched", [False, True])
def test_array_from_vector_other_fill(
    write_files, grid_from_vector_all, refine, all_touched
):
    ar = grid_from_vector_all.array_from_vector(
        mana_polygons_path,
        attribute="K_m_d",
        fill=1e10,
        refine=refine,
        all_touched=all_touched,
    )
    if write_files:
        hash = md5(ar.tobytes()).hexdigest()[:7]
        fname = f"test_array_from_vector_other_fill_{refine}_"
        if all_touched:
            fname += "all_touched_"
        fname += f"{gdal_version()}_{hash}.tif"
        grid_from_vector_all.write_raster(ar, outdir / fname)
    assert ar.shape == (24, 18)
    assert np.ma.isMaskedArray(ar)
    assert np.issubdtype(ar.dtype, np.floating)
    assert ar.fill_value == 1e10
    assert ar.min() == pytest.approx(0.00012)
    assert ar.max() == pytest.approx(12.3)
    num_unique = len(np.unique(ar))
    if refine == 1:
        assert num_unique == 5
    elif refine == 2:
        assert num_unique in (17, 18)
    elif refine in (None, 5):  # this is the default
        if all_touched:
            assert num_unique in (36, 44)
        else:
            assert num_unique in (47, 38)
    else:
        raise KeyError(refine)


@requires_pkg("fiona", "rasterio")
@pytest.mark.parametrize("refine", [None, 1, 2, 5])
@pytest.mark.parametrize("all_touched", [False, True])
def test_array_from_vector_layer_intnull(grid_from_vector_all, refine, all_touched):
    ar = grid_from_vector_all.array_from_vector(
        datadir,
        attribute="intnull",
        layer="mana_polygons",
        refine=refine,
        all_touched=all_touched,
    )
    assert ar.shape == (24, 18)
    assert ar.fill_value == 0
    assert np.issubdtype(ar.dtype, np.integer)
    assert ar.min() == 4
    assert ar.max() == 51
    if refine is None:
        refine = 5  # this is the default
    if all_touched is False:
        expected_ar_sum = {1: 3487, 2: 4125, 5: 4224}[refine]
        expected_ar_mask_sum = {1: 228, 2: 204, 5: 193}[refine]
    else:
        expected_ar_sum = {1: 5072, 2: 4892, 5: 4594}[refine]
        expected_ar_mask_sum = 181
    assert ar.sum() == expected_ar_sum
    assert ar.mask.sum() == expected_ar_mask_sum


@requires_pkg("fiona", "rasterio")
@pytest.mark.parametrize("refine", [None, 1, 2, 5])
@pytest.mark.parametrize("all_touched", [False, True])
def test_array_from_vector_layer_floatnull(grid_from_vector_all, refine, all_touched):
    ar = grid_from_vector_all.array_from_vector(
        datadir,
        attribute="floatnull",
        layer="mana_polygons",
        refine=refine,
        all_touched=all_touched,
    )
    assert ar.shape == (24, 18)
    assert ar.fill_value == 0.0
    assert np.issubdtype(ar.dtype, np.floating)
    np.testing.assert_almost_equal(ar.min(), 0.002)
    np.testing.assert_almost_equal(ar.max(), 2452.0)
    if refine is None:
        refine = 5  # this is the default
    if all_touched is False:
        expected_ar_sum = {1: 126963.862, 2: 146399.38, 5: 157549.255}[refine]
        expected_ar_mask_sum = {1: 228, 2: 204, 5: 193}[refine]
    else:
        expected_ar_sum = {1: 193418.014, 2: 178515.09, 5: 174303.1076}[refine]
        expected_ar_mask_sum = 181
    assert ar.sum() == pytest.approx(expected_ar_sum)
    assert ar.mask.sum() == expected_ar_mask_sum


@requires_pkg("fiona", "rasterio")
@pytest.mark.parametrize("refine", [None, 1, 2])
@pytest.mark.parametrize("all_touched", [False, True])
def test_array_from_vector_layer_allnull(grid_from_vector_all, refine, all_touched):
    ar = grid_from_vector_all.array_from_vector(
        datadir,
        attribute="allnull",
        layer="mana_polygons",
        refine=refine,
        all_touched=all_touched,
    )
    assert ar.shape == (24, 18)
    assert ar.fill_value == 0
    assert np.issubdtype(ar.dtype, np.integer)
    assert ar.mask.all()
    assert ar.data.min() == ar.data.max()


@requires_pkg("fiona", "rasterio")
@pytest.mark.parametrize("res_shape", [(1000, (13, 8)), (2000, (7, 4))])
@pytest.mark.parametrize("attribute", [None, "id"])
@pytest.mark.parametrize("refine", [None, 1, 5])
@pytest.mark.parametrize("all_touched", [False, True])
def test_array_from_vector_points(caplog, res_shape, attribute, refine, all_touched):
    resolution, shape = res_shape
    grid = Grid(resolution, shape, (1810000.0, 5878000.0))
    with caplog.at_level(logging.INFO):
        ar = grid.array_from_vector(
            points_path, attribute=attribute, refine=refine, all_touched=all_touched
        )
        if refine is None:
            refine = 1  # this is the default
            assert (
                sum(
                    f"selecting default refine={refine} for Point" in msg
                    for msg in caplog.messages
                )
                == 1
            ), caplog.messages
        else:
            assert (
                sum("selecting default" in msg for msg in caplog.messages) == 0
            ), caplog.messages
            assert (
                sum(
                    f"using refine={refine} for Point" in msg for msg in caplog.messages
                )
                == 1
            ), caplog.messages
    assert ar.shape == shape
    assert np.ma.isMaskedArray(ar)
    assert np.issubdtype(ar.dtype, np.integer)
    assert ar.fill_value == 0
    ar_s = set(np.unique(ar).filled())
    if attribute is None:
        assert ar_s == {0, 1}
    else:
        all11 = set(range(11))
        if resolution == 2000:
            # when grid contains more than one point, values are not consistent
            assert len(ar_s) == 9
            assert ar_s.issubset(all11)
            assert ar_s.issuperset(all11.difference({4, 8, 1, 10}))
        else:
            assert ar_s == all11
    assert ar.mask.sum() == {1000: 94, 2000: 20}[resolution]


@requires_pkg("fiona", "rasterio")
@pytest.mark.parametrize("attribute", [None, "nzsegment", "StreamOrde"])
@pytest.mark.parametrize("refine", [None, 1, 2, 5])
@pytest.mark.parametrize("all_touched", [False, True])
def test_array_from_vector_lines(caplog, attribute, refine, all_touched):
    grid = Grid(250, (67, 64), (1803250.0, 5878500.0))
    with caplog.at_level(logging.INFO):
        ar = grid.array_from_vector(
            lines_path, attribute=attribute, refine=refine, all_touched=all_touched
        )
        if refine is None:
            refine = 1  # this is the default
            assert (
                sum(
                    f"selecting default refine={refine} for 3D LineString" in msg
                    for msg in caplog.messages
                )
                == 1
            ), caplog.messages
        else:
            assert (
                sum("selecting default" in msg for msg in caplog.messages) == 0
            ), caplog.messages
            assert (
                sum(
                    f"using refine={refine} for 3D LineString" in msg
                    for msg in caplog.messages
                )
                == 1
            ), caplog.messages
    assert ar.shape == (67, 64)
    assert np.ma.isMaskedArray(ar)
    assert np.issubdtype(ar.dtype, np.integer)
    assert ar.fill_value == 0
    if attribute is None:
        assert set(np.unique(ar).filled()) == {0, 1}
    elif attribute == "nzsegment":
        assert (
            len(np.unique(ar))
            == {
                1: 245 if all_touched is False else 240,
                2: 250,
                5: 258,
            }[refine]
        )
    elif attribute == "StreamOrde":
        assert set(np.unique(ar).filled()) == {0, 1, 2, 3, 4, 5}
    if all_touched is False:
        expected_ar_mask_sum = {1: 3239, 2: 3210, 5: 3180}[refine]
    else:
        expected_ar_mask_sum = 3136
    assert ar.mask.sum() == expected_ar_mask_sum


@requires_pkg("rasterio")
def test_array_from_raster_no_projection():
    grid = Grid.from_bbox(1748762.8, 5448908.9, 1749509, 5449749, 25)
    assert grid.projection is None
    ar = grid.array_from_raster(mana_dem_path)
    assert ar.shape == (34, 31)
    assert ar.mask.sum() == 160


@requires_pkg("rasterio")
def test_array_from_raster_without_projection():
    grid = Grid.from_raster(nocrs_path)
    assert grid.projection is None
    ar = grid.array_from_raster(nocrs_path)
    assert ar.shape == (2, 3)
    assert ar.mask.sum() == 0
    np.testing.assert_array_equal(ar, np.arange(6).reshape(2, 3))


@requires_pkg("rasterio")
def test_array_from_raster_same_projection():
    grid = Grid.from_bbox(
        1748762.8, 5448908.9, 1749509, 5449749, 25, projection="EPSG:2193"
    )
    assert grid.projection == "EPSG:2193"
    ar = grid.array_from_raster(mana_dem_path)
    assert ar.shape == (34, 31)
    assert ar.mask.sum() == 160


@requires_pkg("rasterio")
def test_array_from_raster_different_projection():
    grid = Grid.from_bbox(
        19455906, -5026598, 19457499, -5024760, 25, projection="EPSG:3857"
    )
    assert grid.projection == "EPSG:3857"
    ar = grid.array_from_raster(mana_dem_path)
    assert ar.shape == (74, 64)
    assert ar.mask.sum() == 1077


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_no_projection():
    grid = Grid.from_bbox(1748762.8, 5448908.9, 1749509, 5449749, 25)
    assert grid.projection is None
    ar = grid.array_from_vector(mana_polygons_path, attribute="K_m_d", refine=1)
    assert ar.shape == (34, 31)
    assert ar.mask.sum() == 146
    ar = grid.array_from_vector(mana_polygons_path, attribute="K_m_d", all_touched=True)
    assert ar.mask.sum() == 128


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_same_projection():
    # TODO: EPSG:2193 != tests/data/Mana_polygons.prj due to axis order
    projection = mana_polygons_path.with_suffix(".prj").read_text().strip()
    grid = Grid.from_bbox(
        1748762.8, 5448908.9, 1749509, 5449749, 25, projection=projection
    )
    assert grid.projection == projection
    ar = grid.array_from_vector(mana_polygons_path, attribute="K_m_d", refine=1)
    assert ar.shape == (34, 31)
    assert ar.mask.sum() == 146
    ar = grid.array_from_vector(mana_polygons_path, attribute="K_m_d", all_touched=True)
    assert ar.mask.sum() == 128


@requires_pkg("fiona", "rasterio")
def test_array_from_vector_different_projection():
    grid = Grid.from_bbox(
        19455906, -5026598, 19457499, -5024760, 25, projection="EPSG:3857"
    )
    assert grid.projection == "EPSG:3857"
    ar = grid.array_from_vector(mana_polygons_path, attribute="K_m_d", refine=1)
    assert ar.shape == (74, 64)
    assert ar.mask.sum() == 950
    ar = grid.array_from_vector(mana_polygons_path, attribute="K_m_d", all_touched=True)
    assert ar.mask.sum() == 873


@requires_pkg("fiona", "rasterio")
def test_mask_from_vector(grid_from_vector_all):
    def check(ar):
        assert ar.shape == (24, 18)
        assert ar.dtype == "bool"
        assert ar.sum() == 153

    # use layer
    mask1 = grid_from_vector_all.mask_from_vector(datadir, layer="mana_polygons")
    check(mask1)

    # use path
    mask2 = grid_from_vector_all.mask_from_vector(mana_polygons_path)
    check(mask2)
    np.testing.assert_array_equal(mask1, mask2)


@requires_pkg("rasterio")
def test_mask_from_raster(grid_from_raster):
    mask = grid_from_raster.mask_from_raster(mana_dem_path)
    assert mask.shape == (278, 209)
    assert mask.dtype == "bool"
    assert mask.sum() == 23782
