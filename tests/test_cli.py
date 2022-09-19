import pytest

from .conftest import datadir, requires_pkg, run_cli, set_env


module_name = "gridit"
mana_tif = datadir / "Mana.tif"
mana_shp = datadir / "Mana_polygons.shp"
modflow_dir = datadir / "modflow"
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
def test_grid_from_bbox_array_from_raster(tmp_path, grid_from_bbox_args):
    out_path = tmp_path / "out.tif"
    stdout, stderr, returncode = run_cli(
        grid_from_bbox_args +
        ["--array-from-raster", str(mana_tif) + ":1",
         "--write-raster", str(out_path)])
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0
    assert out_path.exists()


@requires_pkg("fiona")
def test_grid_from_bbox_array_from_vector(grid_from_bbox_args):
    stdout, stderr, returncode = run_cli(
        grid_from_bbox_args +
        ["--array-from-vector", mana_shp])
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0


@requires_pkg("fiona", "matplotlib")
def test_grid_from_bbox_array_from_vector_attribute(
        tmp_path, grid_from_bbox_args):
    out_path = tmp_path / "out.png"
    stdout, stderr, returncode = run_cli(
        grid_from_bbox_args +
        ["--array-from-vector", mana_shp,
         "--array-from-vector-attribute", "K_m_d",
         "--write-image", str(out_path)])
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0
    assert out_path.exists()


@requires_pkg("fiona", "netcdf4", "xarray")
def test_grid_from_vector_array_from_netcdf(tmp_path):
    out_path = tmp_path / "out.txt"
    args = ([
        module_name,
        "--grid-from-vector", waitaku2_shp,
        "--resolution", "250",
        "--array-from-vector", f"{datadir}:waitaku2",
        "--array-from-vector-attribute", "rid",
        "--array-from-netcdf", f"{waitaku2_nc}:rid:myvar:0",
        "--time-stats", "quantile(0.75),max",
        "--write-text", str(out_path) + ":%12.7E",
    ])
    with set_env(GRID_CACHE_DIR=str(tmp_path)):
        stdout, stderr, returncode = run_cli(args)
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0
    assert not out_path.exists()
    assert set(pth.name for pth in out_path.parent.iterdir()).issuperset(
        {"out_max.txt", "out_quantile(0.75).txt"})


@requires_pkg("flopy")
def test_grid_from_modflow_classic(grid_from_bbox_args):
    stdout, stderr, returncode = run_cli([
        module_name,
        "--grid-from-modflow", modflow_dir / "h.nam",
    ])
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0


@requires_pkg("fiona", "flopy")
def test_grid_from_modflow_array_from_vector_attribute(
        tmp_path, grid_from_bbox_args):
    out_path = tmp_path / "out.txt"
    stdout, stderr, returncode = run_cli([
        module_name,
        "--grid-from-modflow", str(modflow_dir / "mfsim.nam") + ":h6",
        "--array-from-vector", waitaku2_shp,
        "--array-from-vector-attribute", "rid",
        "--write-text", str(out_path),
    ])
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0
    assert out_path.exists()
