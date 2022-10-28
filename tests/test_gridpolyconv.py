import logging
import numpy as np
import pytest

from .conftest import datadir, has_pkg, requires_pkg, set_env
from gridit import Grid, GridPolyConv

if not (has_pkg("fiona") and has_pkg("rasterio")):
    pytest.skip(
        "skipping tests that require fiona and rasterio",
        allow_module_level=True)

if has_pkg("fiona"):
    import fiona


waitaku2_shp = datadir / "waitaku2.shp"
waitaku2_nc = datadir / "waitaku2.nc"


@pytest.fixture
def waitaku2_gpc_rid_2():
    grid = Grid.from_bbox(1811435, 5866204, 1815226, 5871934, 250)
    gpc = GridPolyConv.from_grid_vector(
        grid, waitaku2_shp, "rid", refine=2, caching=0)
    return gpc


@pytest.fixture
def waitaku2_gpc_rid_5():
    grid = Grid.from_bbox(1806827, 5865749, 1817941, 5875560, 1000)
    gpc = GridPolyConv.from_grid_vector(
        grid, waitaku2_shp, "rid", refine=5, max_levels=6, caching=0)
    return gpc


def test_init(caplog):
    # test errors first
    ar = np.array(
        [[[0, 1],
          [2, 1]],
         [[0, 2],
          [0, 2]]])
    with pytest.raises(ValueError, match="poly_idx must be a tuple or list-"):
        GridPolyConv(1, ar[0])
    with pytest.raises(ValueError, match="poly_idx values are not unique"):
        GridPolyConv([101, 101], ar[0])
    with pytest.raises(ValueError, match="idx_ar must be a numpy array"):
        GridPolyConv([101, 102], 1)
    with pytest.raises(ValueError, match="idx_ar dtype must integer-based"):
        GridPolyConv([101, 102], ar[0].astype(float))
    with pytest.raises(ValueError, match="idx_ar ndim must be 2 or 3"):
        GridPolyConv([101, 102], ar[0, 0])
    with caplog.at_level(logging.ERROR):
        _ = GridPolyConv([101, 102], ar[0], ar)
        assert "ignoring ar_count" in caplog.messages[-1]
    with pytest.raises(ValueError, match="ar_count must be specified"):
        GridPolyConv([101, 102], ar)
    with pytest.raises(ValueError, match="ar_count must be a numpy array"):
        GridPolyConv([101, 102], ar, 1)
    with pytest.raises(ValueError, match="ar_count dtype must integer-based"):
        GridPolyConv([101, 102], ar, ar.astype(float))
    with pytest.raises(ValueError, match="ar_count shape must match idx_ar"):
        GridPolyConv([101, 102], ar, ar[0])
    # test success
    gpc = GridPolyConv([102, 101], ar[0])
    assert gpc.poly_idx == (102, 101)
    np.testing.assert_array_equal(gpc.idx_ar, ar[0])
    assert gpc.ar_count is None
    assert gpc.weight is None
    np.testing.assert_array_equal(gpc.mask, ar[0] == 0)
    ar_count = np.array(
        [[[0, 3],
          [4, 1]],
         [[0, 1],
          [0, 3]]])
    gpc = GridPolyConv([102, 101], ar, ar_count)
    assert gpc.poly_idx == (102, 101)
    np.testing.assert_array_equal(gpc.idx_ar, ar)
    np.testing.assert_array_equal(gpc.ar_count, ar_count)
    np.testing.assert_array_equal(gpc.weight, ar_count / 4)
    np.testing.assert_array_equal(gpc.mask, ar[0] == 0)


def test_from_grid_vector_errors():
    grid = Grid.from_bbox(1811435, 5866204, 1815226, 5871934, 250)
    with pytest.raises(ValueError, match="grid must be an instance of Grid"):
        GridPolyConv.from_grid_vector(
            1, waitaku2_shp, "rid", caching=0)
    with pytest.raises(ValueError, match="refine must be int"):
        GridPolyConv.from_grid_vector(
            grid, waitaku2_shp, "rid", refine=1.0, caching=0)
    with pytest.raises(ValueError, match="refine must be >= 1"):
        GridPolyConv.from_grid_vector(
            grid, waitaku2_shp, "rid", refine=0, caching=0)
    with pytest.raises(ValueError, match="max_levels must be int"):
        GridPolyConv.from_grid_vector(
            grid, waitaku2_shp, "rid", max_levels=1.0, caching=0)
    with pytest.raises(ValueError, match="max_levels must be >= 1"):
        GridPolyConv.from_grid_vector(
            grid, waitaku2_shp, "rid", max_levels=0, caching=0)
    with pytest.raises(KeyError, match=r"could not find 'nope' in \['rid'\]"):
        GridPolyConv.from_grid_vector(grid, waitaku2_shp, "nope", caching=0)
    outgrid = Grid.from_bbox(0, 0, 1000, 2000, 250)
    with pytest.raises(ValueError, match="no features were found in grid ext"):
        GridPolyConv.from_grid_vector(outgrid, waitaku2_shp, "rid", caching=0)


def test_caching(tmp_path):
    with set_env(GRID_CACHE_DIR=str(tmp_path)):
        assert len(list(tmp_path.iterdir())) == 0
        grid = Grid.from_bbox(1811435, 5866204, 1815226, 5871934, 250)
        gpc0 = GridPolyConv.from_grid_vector(
            grid, waitaku2_shp, "rid", caching=0)
        assert len(list(tmp_path.iterdir())) == 0
        gpc1 = GridPolyConv.from_grid_vector(
            grid, waitaku2_shp, "rid", caching=1)
        assert len(list(tmp_path.iterdir())) == 1
        gpc2 = GridPolyConv.from_grid_vector(
            grid, waitaku2_shp, "rid", caching=1)
        assert len(list(tmp_path.iterdir())) == 1
        gpc3 = GridPolyConv.from_grid_vector(
            grid, waitaku2_shp, "rid", caching=2)
        assert len(list(tmp_path.iterdir())) == 2
        assert gpc1.idx_ar.shape == (5, 24, 16)
        assert gpc1 == gpc0
        assert gpc1 == gpc2
        assert gpc1 == gpc3
        np.testing.assert_array_equal(gpc1.weight, gpc2.weight)
        grid2 = Grid.from_bbox(1811435, 5866204, 1815226, 5871934, 300)
        gpc4 = GridPolyConv.from_grid_vector(
            grid2, waitaku2_shp, "rid", caching=1)
        assert len(list(tmp_path.iterdir())) == 3
        assert gpc1 != gpc4


@pytest.fixture
def waitaku2_index_values():
    index = []
    with fiona.open(waitaku2_shp) as ds:
        for f in ds:
            index.append(f["properties"]["rid"])
    values = np.array(index) / 1000. - 3040
    return index, values


def test_array_from_values_errors(waitaku2_gpc_rid_2, waitaku2_index_values):
    gpc = waitaku2_gpc_rid_2
    index, values = waitaku2_index_values
    values2 = np.stack([values, values + 1])
    with pytest.raises(ValueError, match="expected index be a list"):
        gpc.array_from_values(object(), values)
    with pytest.raises(ValueError, match="expected values to have 1 or 2 dim"):
        gpc.array_from_values(index, np.float64(4))
    with pytest.raises(ValueError, match="expected values to have 1 or 2 dim"):
        gpc.array_from_values(index, np.float64(4).reshape((1, 1, 1)))
    with pytest.raises(ValueError, match="length of last dimension of values"):
        gpc.array_from_values(index, values.reshape((67, 1)))
    with pytest.raises(ValueError, match="length of last dimension of values"):
        gpc.array_from_values(index, values2.T)
    with pytest.raises(ValueError, match="values must have one dimension"):
        gpc.array_from_values(index, values2, enforce1d=True)
    with pytest.raises(ValueError, match="index is disjoint from poly_idx"):
        gpc.array_from_values(list(np.array(index) + 10000), values)
    with pytest.raises(ValueError, match="index is not a superset of poly_id"):
        gpc.array_from_values(list(np.array(index) + 1), values)


def min_max_sum(ar):
    return np.array([ar.min(), ar.max(), ar.sum()])


def test_waitaku2_200(waitaku2_index_values):
    expected_fill_value = np.ma.masked_array(0.0).fill_value
    # resolution 200 captures all catchment polygons
    grid = Grid.from_vector(waitaku2_shp, 200)
    gpc = GridPolyConv.from_grid_vector(
        grid, waitaku2_shp, "rid", refine=5, max_levels=3, caching=0)
    assert len(gpc.poly_idx) == 67
    assert gpc.poly_idx[:4] == (3046539, 3046727, 3046736, 3046737)
    assert gpc.poly_idx[-4:] == (3049674, 3049683, 3050099, 3050100)
    assert gpc.idx_ar.shape == (3, 88, 83)
    assert list(gpc.idx_ar.min((1, 2))) == [0, 0, 0]
    assert list(gpc.idx_ar.max((1, 2))) == [67, 67, 67]
    assert list(gpc.idx_ar.sum((1, 2))) == [141009, 38182, 3412]
    assert gpc.ar_count.shape == (3, 88, 83)
    assert list(gpc.ar_count.min((1, 2))) == [0, 0, 0]
    assert list(gpc.ar_count.max((1, 2))) == [25, 12, 7]
    assert list(gpc.ar_count.sum((1, 2))) == [96852, 6307, 277]
    assert gpc.weight.shape == (3, 88, 83)
    assert gpc.weight.min() == 0.0
    assert gpc.weight.max() == 1.0
    assert gpc.weight.sum(0).max() == 1.0
    index, values = waitaku2_index_values
    # 1D values
    ar1 = gpc.array_from_values(index, values)
    assert ar1.shape == (88, 83)
    np.testing.assert_approx_equal(ar1.min(), 6.539)
    np.testing.assert_approx_equal(ar1.data.min(), 0.0)
    np.testing.assert_approx_equal(ar1.max(), 10.1)
    np.testing.assert_approx_equal(ar1.sum(), 34795.77622153665)
    assert ar1.fill_value == expected_fill_value
    # with fill
    ar1f = gpc.array_from_values(index, values, 7.0)
    assert ar1f.shape == (88, 83)
    np.testing.assert_approx_equal(ar1f.min(), 6.539)
    np.testing.assert_approx_equal(ar1f.max(), 10.1)
    np.testing.assert_approx_equal(ar1f.sum(), 34795.77622153665)
    np.testing.assert_approx_equal(ar1f.data.sum(), 55851.77622153665)
    assert ar1f.fill_value == expected_fill_value
    # 2D values
    values2 = np.stack([values, values + 1])
    assert values2.shape == (2, 67)
    ar2 = gpc.array_from_values(index, values2)
    assert ar2.shape == (2, 88, 83)
    np.testing.assert_approx_equal(ar2.min(), 6.539)
    np.testing.assert_approx_equal(ar2.data.min(), 0.0)
    np.testing.assert_approx_equal(ar2.max(), 11.1)
    np.testing.assert_approx_equal(ar2.sum(), 73887.53389089942)
    np.testing.assert_array_almost_equal(ar1, ar2[0])
    assert ar2.fill_value == expected_fill_value
    # with fill
    ar2f = gpc.array_from_values(index, values2, 7.0)
    assert ar2f.shape == (2, 88, 83)
    np.testing.assert_approx_equal(ar2f.min(), 6.539)
    np.testing.assert_approx_equal(ar2f.max(), 11.1)
    np.testing.assert_approx_equal(ar2f.sum(), 73887.53389089942)
    np.testing.assert_approx_equal(ar2f.data.sum(), 115999.5524430733)
    np.testing.assert_array_almost_equal(ar1f, ar2f[0])
    assert ar2f.fill_value == expected_fill_value


def test_waitaku2_500(caplog, waitaku2_index_values):
    # resolution 500 captures a subset of catchment polygons
    grid = Grid.from_vector(waitaku2_shp, 500)
    exp_l = "missing idx values: [2, 4, 37] or " \
        "'rid' values: [3046727, 3046737, 3048351]"
    with caplog.at_level(logging.INFO):
        gpc = GridPolyConv.from_grid_vector(
            grid, waitaku2_shp, "rid", caching=0)
        assert exp_l in caplog.messages[-1]
    assert len(gpc.poly_idx) == 67
    assert gpc.poly_idx[:4] == (3046539, 3046727, 3046736, 3046737)
    assert gpc.poly_idx[-4:] == (3049674, 3049683, 3050099, 3050100)
    assert gpc.idx_ar.shape == (4, 35, 33)
    assert list(gpc.idx_ar.min((1, 2))) == [0, 0, 0, 0]
    assert list(gpc.idx_ar.max((1, 2))) == [67, 67, 67, 50]
    assert list(gpc.idx_ar.sum((1, 2))) == [23802, 13106, 3213, 263]
    assert gpc.weight.min() == 0.0
    assert gpc.weight.max() == 1.0
    assert gpc.weight.sum(0).max() == 1.0
    index, values = waitaku2_index_values
    # 1D values
    ar1 = gpc.array_from_values(index, values)
    assert ar1.shape == (35, 33)
    np.testing.assert_approx_equal(ar1.min(), 6.539)
    np.testing.assert_approx_equal(ar1.max(), 10.1)
    np.testing.assert_approx_equal(ar1.sum(), 5864.9361446043295)
    # 2D values
    values2 = np.stack([values, values + 1])
    assert values2.shape == (2, 67)
    ar2 = gpc.array_from_values(index, values2)
    assert ar2.shape == (2, 35, 33)
    np.testing.assert_approx_equal(ar2.min(), 6.539)
    np.testing.assert_approx_equal(ar2.max(), 11.1)
    np.testing.assert_approx_equal(ar2.sum(), 12453.872289208659)
    np.testing.assert_array_almost_equal(ar1, ar2[0])


def test_waitaku2_500_no_refine(waitaku2_index_values):
    # resolution 500 captures a subset of catchment polygons
    grid = Grid.from_vector(waitaku2_shp, 500)
    gpc = GridPolyConv.from_grid_vector(
        grid, waitaku2_shp, "rid", refine=1, caching=0)
    assert len(gpc.poly_idx) == 67
    assert gpc.poly_idx[:4] == (3046539, 3046727, 3046736, 3046737)
    assert gpc.poly_idx[-4:] == (3049674, 3049683, 3050099, 3050100)
    assert gpc.idx_ar.shape == (35, 33)
    assert gpc.weight is None
    assert gpc.idx_ar.min() == 0
    assert gpc.idx_ar.max() == 67
    assert gpc.idx_ar.sum() == 21724
    index, values = waitaku2_index_values
    # 1D values
    ar1 = gpc.array_from_values(index, values)
    assert ar1.shape == (35, 33)
    assert ar1.dtype == np.float64
    np.testing.assert_approx_equal(ar1.min(), 6.539)
    np.testing.assert_approx_equal(ar1.data.min(), 0.0)
    np.testing.assert_approx_equal(ar1.max(), 10.1)
    np.testing.assert_approx_equal(ar1.sum(), 5373.454)
    # with fill
    ar1f = gpc.array_from_values(index, values, 7.0)
    assert ar1f.shape == (35, 33)
    np.testing.assert_approx_equal(ar1f.min(), 6.539)
    np.testing.assert_approx_equal(ar1f.max(), 10.1)
    np.testing.assert_approx_equal(ar1f.data.sum(), 8810.454)
    np.testing.assert_approx_equal(ar1.sum(), 5373.454)
    # 2D values
    values2 = np.stack([values, values + 1])
    assert values2.shape == (2, 67)
    ar2 = gpc.array_from_values(index, values2)
    assert ar2.shape == (2, 35, 33)
    np.testing.assert_approx_equal(ar2.min(), 6.539)
    np.testing.assert_approx_equal(ar2.data.min(), 0.0)
    np.testing.assert_approx_equal(ar2.max(), 11.1)
    np.testing.assert_approx_equal(ar2.sum(), 11410.908)
    np.testing.assert_array_almost_equal(ar1, ar2[0])
    # with fill
    ar2f = gpc.array_from_values(index, values2, 7.0)
    assert ar2f.shape == (2, 35, 33)
    np.testing.assert_approx_equal(ar2f.min(), 6.539)
    np.testing.assert_approx_equal(ar2f.max(), 11.1)
    np.testing.assert_approx_equal(ar2f.sum(), 11410.908)
    np.testing.assert_approx_equal(ar2f.data.sum(), 18284.908)
    np.testing.assert_array_almost_equal(ar1f, ar2f[0])


def test_waitaku2_different_projection(waitaku2_index_values):
    grid = Grid.from_bbox(
        19526452, -4480887, 19530406, -4476213, 250, projection="EPSG:3857")
    gpc = GridPolyConv.from_grid_vector(grid, waitaku2_shp, "rid", caching=0)
    assert len(gpc.poly_idx) == 12
    assert gpc.poly_idx[:4] == (3047451, 3047452, 3047648, 3047649)
    assert gpc.poly_idx[-4:] == (3048351, 3048363, 3048532, 3048533)
    assert gpc.idx_ar.shape == (4, 20, 17)
    assert list(gpc.idx_ar.min((1, 2))) == [1, 0, 0, 0]
    assert list(gpc.idx_ar.max((1, 2))) == [12, 12, 11, 10]
    assert list(gpc.idx_ar.sum((1, 2))) == [2250, 525, 39, 10]
    assert gpc.weight.min() == 0.0
    assert gpc.weight.max() == 1.0
    assert gpc.weight.sum(0).max() == 1.0
    index, values = waitaku2_index_values
    # 1D values
    ar1 = gpc.array_from_values(index, values)
    assert ar1.shape == (20, 17)
    np.testing.assert_approx_equal(ar1.min(), 7.451)
    np.testing.assert_approx_equal(ar1.max(), 8.533)
    np.testing.assert_approx_equal(ar1.sum(), 2745.88476)
    # 2D values
    values2 = np.stack([values, values + 1])
    assert values2.shape == (2, 67)
    ar2 = gpc.array_from_values(index, values2)
    assert ar2.shape == (2, 20, 17)
    np.testing.assert_approx_equal(ar2.min(), 7.451)
    np.testing.assert_approx_equal(ar2.max(), 9.533)
    np.testing.assert_approx_equal(ar2.sum(), 5831.76952)
    np.testing.assert_array_almost_equal(ar1, ar2[0])


@requires_pkg("netcdf4", "xarray")
def test_array_from_netcdf_errors(caplog, waitaku2_gpc_rid_2):
    gpc = waitaku2_gpc_rid_2
    # works without error/warning
    with caplog.at_level(logging.WARNING):
        res = gpc.array_from_netcdf(waitaku2_nc, "rid", "myvar", xidx=0)
        assert isinstance(res, dict)
        assert len(caplog.messages) == 0
        # now generate one warning
        res = gpc.array_from_netcdf(waitaku2_nc, "rid", "myvar")
        assert isinstance(res, dict)
        assert len(caplog.messages) == 1
        assert "dataset has extra dimension 'run'" in caplog.messages[-1]
    with pytest.raises(AttributeError, match="cannot find 'novar' in variabl"):
        gpc.array_from_netcdf(waitaku2_nc, "rid", "novar")
    with pytest.raises(AttributeError, match="cannot find 'noidx' in variabl"):
        gpc.array_from_netcdf(waitaku2_nc, "noidx", "myvar")
    with pytest.raises(ValueError, match="expected 1-d myvar index dimension"):
        gpc.array_from_netcdf(waitaku2_nc, "myvar", "myvar")
    with pytest.raises(IndexError, match="index 2 is out of bounds for axis "):
        gpc.array_from_netcdf(waitaku2_nc, "rid", "myvar", xidx=2)
    with pytest.raises(KeyError):  # xidx should be int
        gpc.array_from_netcdf(waitaku2_nc, "rid", "myvar", xidx="0")
    args = (waitaku2_nc, "rid", "myvar")
    with pytest.raises(ValueError, match="expected one ':' in time stats"):
        gpc.array_from_netcdf(*args, time_stats="Jan:mean:max")
    with pytest.raises(ValueError, match="too many '-' for time_window"):
        gpc.array_from_netcdf(*args, time_stats="Jan-Feb-Mar:mean")
    with pytest.raises(ValueError, match=r"error reading quantile\(N\)"):
        gpc.array_from_netcdf(*args, time_stats="quantile(N)")
    with pytest.raises(ValueError, match="unhandled time stats 'quantile'"):
        gpc.array_from_netcdf(*args, time_stats="quantile")
    with pytest.raises(ValueError, match="time stats window 'Nov' not suppor"):
        gpc.array_from_netcdf(*args, time_stats="Nov:mean")


@requires_pkg("netcdf4", "xarray")
def test_array_from_netcdf(waitaku2_gpc_rid_2):
    gpc = waitaku2_gpc_rid_2
    args = (waitaku2_nc, "rid", "myvar")
    kwargs = {"xidx": 0}
    # 1D values with default time_stats="mean"
    r_default = gpc.array_from_netcdf(*args, **kwargs)
    assert isinstance(r_default, dict)
    assert list(r_default.keys()) == ["mean"]
    ar_default = r_default["mean"]
    assert ar_default.shape == (24, 16)
    assert ar_default.dtype == np.float64
    r_mean = gpc.array_from_netcdf(*args, **kwargs, time_stats="mean")
    assert list(r_mean.keys()) == ["mean"]
    ar_mean = r_mean["mean"]
    np.testing.assert_equal(ar_default, ar_mean)
    np.testing.assert_array_almost_equal(
        min_max_sum(ar_mean),
        [0.000814885541331023, 0.004716134164482355, 1.643364535892033])
    # process several time stats
    r3 = gpc.array_from_netcdf(
        *args, **kwargs, time_stats="median,min,max,quantile(0.25)")
    assert list(r3.keys()) == ["median", "min", "max", "quantile(0.25)"]
    np.testing.assert_array_almost_equal(
        min_max_sum(r3["median"]),
        [0.0007635842193849385, 0.00459129037335515, 1.5956306542357197])
    np.testing.assert_array_almost_equal(
        min_max_sum(r3["min"]),
        [8.118862751871347e-05, 0.0009615062735974789, 0.33079623253433965])
    np.testing.assert_array_almost_equal(
        min_max_sum(r3["max"]),
        [0.0017967323074117303, 0.009081936441361904, 3.1762626652780455])
    np.testing.assert_array_almost_equal(
        min_max_sum(r3["quantile(0.25)"]),
        [0.00031803673482500017, 0.002559370594099164, 0.8835400953612407])
    # 2D values, all times returned
    r_none = gpc.array_from_netcdf(*args, **kwargs, time_stats=None)
    assert list(r_none.keys()) == [None]
    ar_none = r_none[None]
    assert ar_none.shape == (371, 24, 16)
    np.testing.assert_array_almost_equal(
        min_max_sum(ar_none),
        [8.118862751871347e-05, 0.009081936441361904, 609.6882373648059])
    # define time_window
    rt = gpc.array_from_netcdf(
        *args, **kwargs, time_stats="annual:mean,min,max")
    assert list(rt.keys()) == ["mean", "min", "max"]
    np.testing.assert_array_almost_equal(
        min_max_sum(rt["mean"]),
        [0.000814885541331023, 0.004716134164482355, 1.643364535892033])
    np.testing.assert_array_almost_equal(
        min_max_sum(rt["min"]),
        [0.0007734778919257224, 0.004534569568932056, 1.5795951503532706])
    np.testing.assert_array_almost_equal(
        min_max_sum(rt["max"]),
        [0.0016269355546683073, 0.008276794105768204, 2.893949011049699])


@requires_pkg("xarray")
@pytest.mark.parametrize("idx_ar,ar_count", [
    (np.uint8(1).reshape((1,) * 2), None),
    (np.uint8(1).reshape((1,) * 3), np.uint8(1).reshape((1,) * 3)),
], ids=["2D", "3D"])
def test_array_from_netcdf_time_stats(idx_ar, ar_count):
    # netcdf not required for this test
    import pandas as pd
    import xarray

    # 1x1 grid with 1 catchment with ID 100
    gpc = GridPolyConv([100], idx_ar, ar_count)
    time = pd.date_range("2013-01-01", periods=365 * 3)
    data = (
        time.year - 2010 +
        time.month / 100 +
        time.day / 10000
    ).to_numpy().reshape((1, -1))
    ds = xarray.DataArray(
        data=data,
        coords={"rid": [100], "time": time},
        dims=["rid", "time"],
    ).to_dataset(name="dat")

    # bypass timestats
    res = gpc.array_from_netcdf(ds, "rid", "dat", time_stats=None)
    expected = ds.dat.to_numpy().reshape((-1, 1, 1))
    assert list(res.keys()) == [None]
    np.testing.assert_allclose(res[None], expected)

    def check_result(res, expected):
        assert list(res.keys()) == list(expected.keys())
        for key, array in expected.items():
            np.testing.assert_allclose(
                res[key], np.ma.atleast_2d(array), err_msg=f"key is {key!r}")

    args = (ds, "rid", "dat")

    # default is "mean"
    res = gpc.array_from_netcdf(*args)
    check_result(res, {"mean": 4.0668323287671235})

    time_stats = "min,mean,median,quantile(0.5),max"

    # no time-window
    res = gpc.array_from_netcdf(*args, time_stats=time_stats)
    check_result(
        res,
        {"min": 3.0101,
         "mean": 4.0668323287671235,
         "median": 4.0702,
         "quantile(0.5)": 4.0702,
         "max": 5.1231})

    # full year, Jan-Dec
    res = gpc.array_from_netcdf(*args, time_stats="annual:" + time_stats)
    check_result(
        res,
        {"min": 3.0668323287671235,
         "mean": 4.0668323287671235,
         "median": 4.066832328767123,
         "quantile(0.5)": 4.0702,
         "max": 5.066832328767123})

    # NZ water year
    res = gpc.array_from_netcdf(*args, time_stats="July-June:" + time_stats)
    check_result(
        res,
        {"min": 3.0366430939226516,
         "mean": 4.0668323287671235,
         "median": 4.562722739726028,
         "quantile(0.5)": 4.0702,
         "max": 5.096529347826087})

    # US water year
    res = gpc.array_from_netcdf(*args, time_stats="Oct-Sep:" + time_stats)
    check_result(
        res,
        {"min": 3.051751282051282,
         "mean": 4.0668323287671235,
         "median": 4.814777534246575,
         "quantile(0.5)": 4.0702,
         "max": 5.1115836956521745})

    # two months
    res = gpc.array_from_netcdf(*args, time_stats="Jan-feb:" + time_stats)
    check_result(
        res,
        {"min": 3.016274576271186,
         "mean": 4.0162745762711864,
         "median": 4.016274576271186,
         "quantile(0.5)": 4.0130,
         "max": 5.0162745762711864})

    # one month
    res = gpc.array_from_netcdf(*args, time_stats="Nov-november:" + time_stats)
    check_result(
        res,
        {"min": 3.11155,
         "mean": 4.11155,
         "median": 4.11155,
         "quantile(0.5)": 4.11155,
         "max": 5.11155})
