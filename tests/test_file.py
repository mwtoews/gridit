"""Test file methods."""

import numpy as np
import pytest

from gridit import Grid
from gridit.file import (
    fiona_filter_collection,
    fiona_property_type,
    float32_is_also_float64,
)

from .conftest import datadir

points_path = datadir / "waitaku2_points.shp"


def test_float32_is_also_float64():
    assert float32_is_also_float64(np.float32(1.0))
    assert float32_is_also_float64(np.float32(3.28e9))
    assert float32_is_also_float64(np.float32(np.nan))
    assert not float32_is_also_float64(np.float32(0.1))
    assert not float32_is_also_float64(np.float32(3.29e9))


def test_fiona_property_type():
    assert fiona_property_type(np.array([0], np.uint8)) == "int:1"
    assert fiona_property_type(np.array([-1, 5])) == "int:2"
    assert fiona_property_type(np.array([12000])) == "int:5"

    assert fiona_property_type(np.array([True])) == "int:1"

    assert fiona_property_type(np.array([0.0])) == "float:1"
    assert fiona_property_type(np.array([0.1], np.float32)) == "float:3.1"
    assert fiona_property_type(np.array([-0.1])) == "float:4.1"
    assert fiona_property_type(np.array([123.0])) == "float:3"
    assert fiona_property_type(np.array([1e-12])) == "float:14.12"
    assert fiona_property_type(np.array([123.45])) == "float:6.2"
    assert fiona_property_type(np.array([np.nan, 123.45])) == "float:6.2"
    assert fiona_property_type(np.array([np.nan])) == "float"

    assert fiona_property_type(np.array(["x"])) == "str:1"
    assert fiona_property_type(np.array([b"x"])) == "str:1"
    assert fiona_property_type(np.array(["x", "z z"])) == "str:3"
    assert fiona_property_type(np.array([set()])) == "str"


@pytest.fixture
def grid_basic():
    return Grid(10, (20, 30), (1000.0, 2000.0))


@pytest.fixture
def grid_projection():
    return Grid(10, (20, 30), (1000.0, 2000.0), "EPSG:2193")


@pytest.mark.parametrize("masked", [False, True])
def test_write_raster(tmp_path, grid_basic, grid_projection, masked):
    rasterio = pytest.importorskip("rasterio")

    ar_float32 = np.arange(20 * 30, dtype=np.float32).reshape((20, 30))
    if masked:
        ar_float32 = np.ma.array(ar_float32)
    fname = tmp_path / "float32.tif"
    grid_basic.write_raster(ar_float32, fname)
    with rasterio.open(fname) as ds:
        assert not ds.crs
        if masked:
            assert ds.nodata == 3.28e9
        else:
            assert ds.nodata is None
        ar = ds.read()
        assert ar.dtype == np.float32
        assert ar.shape == (1, 20, 30)
        np.testing.assert_array_almost_equal(ar[0], ar_float32)

    ar_float64 = np.arange(20 * 30, dtype=np.float64).reshape((20, 30))
    if masked:
        ar_float64 = np.ma.array(ar_float64)
        ar_float64.fill_value = -1.0
    fname = tmp_path / "float64.tif"
    grid_basic.write_raster(ar_float64, fname)
    with rasterio.open(fname) as ds:
        assert not ds.crs
        if masked:
            assert ds.nodata == -1.0
        else:
            assert ds.nodata is None
        ar = ds.read()
        assert ar.dtype == np.float64
        assert ar.shape == (1, 20, 30)
        np.testing.assert_array_almost_equal(ar[0], ar_float64)

    ar_uint16 = np.arange(20 * 30, dtype=np.uint16).reshape((20, 30)) + 1
    if masked:
        ar_uint16 = np.ma.array(ar_uint16)
    fname = tmp_path / "uint8.tif"
    grid_basic.write_raster(ar_uint16, fname)
    with rasterio.open(fname) as ds:
        assert ds.crs is None
        if masked:
            assert ds.nodata == 0
        else:
            assert ds.nodata is None
        ar = ds.read()
        assert ar.dtype == np.uint16
        assert ar.shape == (1, 20, 30)
        np.testing.assert_array_almost_equal(ar[0], ar_uint16)

    ar_uint16 = np.arange(20 * 30, dtype=np.uint16).reshape((20, 30))
    if masked:
        ar_uint16 = np.ma.masked_greater(ar_uint16, 450)
    fname = tmp_path / "uint16.tif"
    grid_projection.write_raster(ar_uint16, fname)
    with rasterio.open(fname) as ds:
        assert ds.crs is not None
        if masked:
            assert ds.nodata == 451
        else:
            assert ds.nodata is None
        ar = ds.read()
        assert ar.dtype == np.uint16
        assert ar.shape == (1, 20, 30)
        np.testing.assert_array_equal(ar[0], ar_uint16)

    ar_bool = np.eye(20, 30, dtype=bool)
    if masked:
        ar_bool = np.ma.array(ar_bool)
    fname = tmp_path / "bool.tif"
    grid_basic.write_raster(ar_bool, fname)
    with rasterio.open(fname) as ds:
        assert ds.crs is None
        if masked:
            assert ds.nodata == 2
        else:
            assert ds.nodata is None
        assert ds.tags(1, ns="IMAGE_STRUCTURE")["NBITS"] == "2"
        ar = ds.read()
        assert ar.dtype == np.uint8
        assert ar.shape == (1, 20, 30)
        np.testing.assert_array_almost_equal(ar[0], ar_bool)

    # errors
    ar1d = np.ones(20 * 30)
    if masked:
        ar1d = np.ma.array(ar1d)
    with pytest.raises(ValueError, match="array must have two-dimensions"):
        grid_basic.write_raster(ar1d, "out.tif")
    with pytest.raises(ValueError, match="array must have two-dimensions"):
        grid_basic.write_raster(ar1d.reshape((1, 20, 30)), "out.tif")


def test_write_vector(tmp_path, grid_basic, grid_projection):
    fiona = pytest.importorskip("fiona")

    ar_float32 = np.arange(20 * 30, dtype=np.float32).reshape((20, 30)) / 10
    fname_float32 = tmp_path / "float32.shp"
    grid_basic.write_vector(ar_float32, fname_float32, "flt")
    with fiona.open(fname_float32) as ds:
        assert not ds.crs
        schema = ds.schema
        assert schema["geometry"] == "Polygon"
        assert dict(ds.schema["properties"]) == {
            "idx": "int:3",
            "row": "int:2",
            "col": "int:2",
            "flt": "float:4.1",
        }
        assert len(ds) == 600
        rec = ds[0]
        geom = dict(rec["geometry"])
        if "geometries" in geom:
            del geom["geometries"]
        assert geom == {
            "coordinates": [
                [
                    (1000.0, 2000.0),
                    (1010.0, 2000.0),
                    (1010.0, 1990.0),
                    (1000.0, 1990.0),
                    (1000.0, 2000.0),
                ]
            ],
            "type": "Polygon",
        }
        assert dict(rec["properties"]) == {
            "idx": 0,
            "row": 0,
            "col": 0,
            "flt": 0.0,
        }

    ar_uint16 = (
        np.ma.masked_greater(np.arange(20 * 30, dtype=np.uint16).reshape((20, 30)), 450)
        * 5
    )
    fname_uint16 = tmp_path / "uint16.shp"
    grid_projection.write_vector(ar_uint16, fname_uint16, ["nums"])
    with fiona.open(fname_uint16) as ds:
        assert ds.crs is not None
        schema = ds.schema
        assert schema["geometry"] == "Polygon"
        assert dict(ds.schema["properties"]) == {
            "idx": "int:3",
            "row": "int:2",
            "col": "int:2",
            "nums": "int:4",
        }
        assert len(ds) == 451
        rec = ds[-1]
        geom = dict(rec["geometry"])
        if "geometries" in geom:
            del geom["geometries"]
        assert geom == {
            "coordinates": [
                [
                    (1000.0, 1850.0),
                    (1010.0, 1850.0),
                    (1010.0, 1840.0),
                    (1000.0, 1840.0),
                    (1000.0, 1850.0),
                ]
            ],
            "type": "Polygon",
        }
        assert dict(rec["properties"]) == {
            "idx": 450,
            "row": 15,
            "col": 0,
            "nums": 2250,
        }

    ar_int = np.ma.arange(2 * 20 * 30).reshape((2, 20, 30)) * 5
    ar_int.mask = False
    ar_int.mask[0, 2:10, 3:8] = True
    ar_int.mask[1, 5:14, 3:20] = True
    fname_int = tmp_path / "int.gpkg"
    grid_basic.write_vector(ar_int, fname_int, ["nums1", "nums2"], layer="foo")
    with fiona.open(fname_int) as ds:
        assert ds.crs is not None
        schema = ds.schema
        assert schema["geometry"] == "Polygon"
        assert dict(ds.schema["properties"]) == {
            "idx": "int",
            "row": "int",
            "col": "int",
            "nums1": "int",
            "nums2": "int",
        }
        assert len(ds) == 575
        rec = ds[575]
        assert dict(rec["properties"]) == {
            "idx": 599,
            "row": 19,
            "col": 29,
            "nums1": 2995,
            "nums2": 5995,
        }

    # errors
    ar2d = np.ones(20 * 30).reshape((20, 30))
    ar3d = np.ones(2 * 20 * 30).reshape((2, 20, 30))
    with pytest.raises(ValueError, match="array must have 2 or 3 dimensions"):
        grid_basic.write_vector(ar2d.ravel(), "out.shp", "val")
    with pytest.raises(ValueError, match="must be a str or a 1 item str list"):
        grid_basic.write_vector(ar2d, "out.shp", ["val1", "val2"])
    with pytest.raises(ValueError, match="must list of str with length 2"):
        grid_basic.write_vector(ar3d, "out.shp", ["val"])
    with pytest.raises(ValueError, match="last two dimensions of array shape"):
        grid_basic.write_vector(ar2d.T, "out.shp", "val")
    with pytest.raises(ValueError, match="Unable to detect driver"):
        grid_basic.write_vector(ar2d, "out.nope", "val")


def test_fiona_filter_collection():
    fiona = pytest.importorskip("fiona")
    expected_schema = {
        "geometry": "Point",
        "properties": {"id": "int:10"},
    }
    with fiona.open(points_path) as ds:
        flt = fiona_filter_collection(ds, filter={"id": 0})
        assert flt.schema == expected_schema
        assert len(flt) == 0
        assert flt.bounds == (0.0, 0.0, 0.0, 0.0)

        flt = fiona_filter_collection(ds, filter={"id": 1})
        assert flt.schema == expected_schema
        assert len(flt) == 1
        np.testing.assert_array_almost_equal(
            flt.bounds,
            (1814758.4763, 5871013.6156, 1814758.4763, 5871013.6156),
            decimal=4,
        )

        flt = fiona_filter_collection(ds, filter={"id": [1, 8]})
        assert flt.schema == expected_schema
        assert len(flt) == 2
        np.testing.assert_array_almost_equal(
            flt.bounds,
            (1812243.7372, 5871013.6156, 1814758.4763, 5876813.8657),
            decimal=4,
        )

        if fiona.__version__[0:3] >= "1.9":
            flt = fiona_filter_collection(ds, filter="id=2")
            assert len(flt) == 1
            np.testing.assert_array_almost_equal(
                flt.bounds,
                (1812459.1405, 5875971.5429, 1812459.1405, 5875971.5429),
                decimal=4,
            )

            # errors
            with pytest.raises(ValueError, match="SQL Expression Parsing Err"):
                fiona_filter_collection(ds, filter="id==2")

        else:
            with pytest.raises(ValueError, match="filter str as SQL WHERE"):
                fiona_filter_collection(ds, filter="id=2")

        with pytest.raises(ValueError, match=r"ds must be fiona\.Collection"):
            fiona_filter_collection(None, filter={"id": 1})

        with pytest.raises(KeyError, match="cannot find filter keys"):
            fiona_filter_collection(ds, filter={"ID": 1})

    with pytest.raises(ValueError, match="ds is closed"):
        fiona_filter_collection(ds, filter={"id": 1})
