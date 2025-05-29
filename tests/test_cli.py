import pytest

from .common import datadir, requires_pkg, run_cli, set_env

module_name = "gridit"
mana_tif = datadir / "Mana.tif"
mana_shp = datadir / "Mana_polygons.shp"
modflow_dir = datadir / "modflow"
waitaku2_nc = datadir / "waitaku2.nc"
waitaku2_shp = datadir / "waitaku2.shp"


def test_usage():
    stdout, stderr, returncode = run_cli([module_name])
    assert "usage" in stdout
    assert len(stderr) == 0
    assert returncode == 0


@pytest.fixture
def grid_from_bbox_args():
    # fmt: off
    return [
        module_name,
        "--grid-from-bbox", "1748762.8", "5448908.9", "1749509", "5449749",
        "--resolution", "25",
    ]
    # fmt: on


def test_grid_from_bbox(grid_from_bbox_args):
    stdout, stderr, returncode = run_cli(grid_from_bbox_args)
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0


@pytest.mark.parametrize(
    "more",
    [
        ["--snap", "half"],
        ["--snap", "(1.1 -2.9)"],
        ["--buffer", "1,-1"],
        ["--buffer", "1,-1,2,2"],
    ],
)
def test_grid_from_bbox_more_args(grid_from_bbox_args, more):
    stdout, stderr, returncode = run_cli(grid_from_bbox_args + more)
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0


@pytest.mark.parametrize(
    "more, msg",
    [
        [["--snap", "2x2"], "--snap: cannot convert '2x2' into"],
        [["--snap", "HALF"], "--snap: must be one of"],
        [["--snap", "(1 2 3 4)"], "--snap: must have two values"],
        [["--buffer", "half"], "--buffer: cannot convert 'half' into"],
        [["--buffer", "1,2,3"], "--buffer: must have two or four values"],
    ],
)
def test_grid_from_bbox_more_args_error(grid_from_bbox_args, more, msg):
    stdout, stderr, returncode = run_cli(grid_from_bbox_args + more)
    assert len(stderr) > 0
    assert len(stdout) > 0
    assert msg in stdout
    assert returncode != 0


@pytest.mark.parametrize(
    "args, msg",
    [
        [["--resolution", "25"], "options must be specified"],
        [["--grid-from-bbox"] + list("0011"), "-grid-from-bbox: requires --resolution"],
    ],
)
def test_grid_from_bbox_bad_args(args, msg):
    stdout, stderr, returncode = run_cli([module_name] + args)
    assert len(stderr) > 0
    assert len(stdout) > 0
    assert msg in stdout
    assert returncode != 0


@requires_pkg("rasterio")
def test_grid_from_bbox_array_from_raster(tmp_path, grid_from_bbox_args):
    out_path = tmp_path / "out.tif"
    # fmt: off
    stdout, stderr, returncode = run_cli(
        grid_from_bbox_args
        + [
            "--array-from-raster", str(mana_tif) + ":1",
            "--array-from-raster-resampling", "nearest",
            "--write-raster", str(out_path),
            "--write-creation-option", "COMPRESS=deflate",
        ]
    )
    # fmt: on
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0
    assert out_path.exists()


@requires_pkg("fiona", "rasterio")
def test_grid_from_bbox_array_from_vector(grid_from_bbox_args):
    stdout, stderr, returncode = run_cli(
        grid_from_bbox_args + ["--array-from-vector", mana_shp]
    )
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0


@requires_pkg("fiona", "matplotlib")
def test_grid_from_bbox_array_from_vector_attribute(tmp_path, grid_from_bbox_args):
    import fiona

    out_png = tmp_path / "out.png"
    out_shp = tmp_path / "out.shp"
    # fmt: off
    stdout, stderr, returncode = run_cli(
        grid_from_bbox_args
        + [
            "--array-from-vector", mana_shp,
            "--array-from-vector-attribute", "K_m_d",
            "--write-image", str(out_png),
            "--write-vector", str(out_shp),
            "--write-vector-attribute", "kmd",
        ]
    )
    # fmt: on
    # assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0
    assert out_png.exists()
    assert {pth.name for pth in tmp_path.iterdir()}.issuperset(
        {"out.shp", "out.shx", "out.dbf"}
    )
    assert not (tmp_path / "out.prj").exists()
    with fiona.open(out_shp) as ds:
        resulting_properties = {
            k: v.replace("int32", "int")
            for k, v in dict(ds.schema["properties"]).items()
        }
    expected_properties = {
        "idx": "int:4",
        "row": "int:2",
        "col": "int:2",
        "kmd": "float:22.20",
    }
    if expected_properties != resulting_properties:
        expected_properties["kmd"] = "float:18.16"
    assert expected_properties == resulting_properties


@requires_pkg("fiona")
def test_write_csv(tmp_path, grid_from_bbox_args):
    out_csv = tmp_path / "out.csv"
    # fmt: off
    stdout, stderr, returncode = run_cli(
        grid_from_bbox_args
        + [
            "--array-from-vector", mana_shp,
            "--array-from-vector-attribute", "K_m_d",
            "--write-vector", str(out_csv),
            "--write-creation-option", "GEOMETRY=AS_WKT",
        ]
    )
    # fmt: on
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0
    assert out_csv.read_text().splitlines()[0] == "WKT,idx,row,col,value"


@requires_pkg("fiona", "netcdf4", "xarray")
def test_grid_from_vector_array_from_netcdf(tmp_path):
    import fiona

    out_txt = tmp_path / "out.txt"
    out_shp = tmp_path / "out.shp"
    # fmt: off
    args = [
        module_name,
        "--grid-from-vector", waitaku2_shp,
        "--resolution", "250",
        "--array-from-vector", f"{datadir}:waitaku2",
        "--array-from-vector-attribute", "rid",
        "--array-from-netcdf", f"{waitaku2_nc}:rid:myvar:0",
        "--time-stats", "quantile(0.75),max",
        "--write-text", str(out_txt) + ":%12.7E",
        "--write-vector",
        str(out_shp),
    ]
    # fmt: on
    with set_env(GRID_CACHE_DIR=str(tmp_path)):
        stdout, stderr, returncode = run_cli(args)
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0
    assert not out_txt.exists()
    assert {pth.name for pth in tmp_path.iterdir()}.issuperset(
        {
            "out_max.txt",
            "out_quantile(0.75).txt",
            "out.shp",
            "out.shx",
            "out.dbf",
            "out.prj",
        }
    )
    with fiona.open(out_shp) as ds:
        assert len(ds) == 4620
        resulting_properties = {
            k: v.replace("int32", "int")
            for k, v in dict(ds.schema["properties"]).items()
        }
    assert resulting_properties == {
        "idx": "int:4",
        "row": "int:2",
        "col": "int:2",
        "quantile(0": "float:22.20",
        "max": "float:22.20",
    }


@requires_pkg("flopy")
def test_grid_from_modflow_classic(grid_from_bbox_args):
    stdout, stderr, returncode = run_cli(
        [
            module_name,
            "--grid-from-modflow",
            modflow_dir / "h.nam",
        ]
    )
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0


@requires_pkg("fiona", "flopy")
def test_grid_from_modflow_array_from_vector_attribute(tmp_path, grid_from_bbox_args):
    out_path = tmp_path / "out.txt"
    # fmt: off
    stdout, stderr, returncode = run_cli(
        [
            module_name,
            "--grid-from-modflow", str(modflow_dir / "mfsim.nam") + ":h6",
            "--array-from-vector", waitaku2_shp,
            "--array-from-vector-attribute", "rid",
            "--write-text", str(out_path),
        ]
    )
    # fmt: on
    assert len(stderr) == 0
    assert len(stdout) > 0
    assert returncode == 0
    assert out_path.exists()
