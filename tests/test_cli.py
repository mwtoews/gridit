import pytest

from .conftest import datadir, requires_pkg, run_cli, set_env


module_name = "gridit"
print(f"{module_name=}")
mana_tif = datadir / "Mana.tif"
mana_shp = datadir / "Mana_polygons.shp"
waitaku2_nc = datadir / "waitaku2.nc"
waitaku2_shp = datadir / "waitaku2.shp"


def test_usage():
    stdout, stderr, returncode = run_cli([module_name])
    assert 'usage' in stdout
    assert len(stderr) == 0
    assert returncode == 0


@pytest.fixture
def grid_from_bbox_args():
    return [
        module_name,
        "--grid-from-bbox", "1748762.8", "5448908.9", "1749509", "5449749",
        "--resolution", "25"]


def test_grid_from_bbox(grid_from_bbox_args):
    stdout, stderr, returncode = run_cli(grid_from_bbox_args)
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0


@requires_pkg("rasterio")
def test_grid_from_bbox_array_from_raster(grid_from_bbox_args):
    stdout, stderr, returncode = run_cli(
        grid_from_bbox_args +
        ["--array-from-raster", mana_tif])
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0


@requires_pkg("fiona")
def test_grid_from_bbox_array_from_vector(grid_from_bbox_args):
    stdout, stderr, returncode = run_cli(
        grid_from_bbox_args +
        ["--array-from-vector", mana_shp])
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0


@requires_pkg("fiona")
def test_grid_from_bbox_array_from_vector_attribute(grid_from_bbox_args):
    stdout, stderr, returncode = run_cli(
        grid_from_bbox_args +
        ["--array-from-vector", mana_shp,
         "--array-from-vector-attribute", "K_m_d"])
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0


@requires_pkg("fiona", "netcdf4", "xarray")
def test_grid_from_vector_array_from_netcdf(tmp_path):
    vn = "__xarray_dataarray_variable__"
    args = ([
        module_name,
        "--grid-from-vector", waitaku2_shp,
        "--resolution", "250",
        "--array-from-vector", f"{datadir}:waitaku2",
        "--array-from-vector-attribute", "rid",
        "--array-from-netcdf", f"{waitaku2_nc}:rid:{vn}",
        "--time-stats", "quantile(0.75),max",
    ])
    with set_env(GRID_CACHE_DIR=str(tmp_path)):
        stdout, stderr, returncode = run_cli(args)
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0
