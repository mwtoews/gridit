import numpy as np
import pytest

from gridit import Grid

from .conftest import requires_pkg


@requires_pkg("shapely")
def test_cell_geoms():
    grid = Grid(50.0, (2, 3), (1000.0, 2000.0))
    first_poly_coords = [
        (1000.0, 2000.0),
        (1050.0, 2000.0),
        (1050.0, 1950.0),
        (1000.0, 1950.0),
        (1000.0, 2000.0),
    ]
    first_centroid = (1025.0, 1975.0)
    one_right_centroid = (1075.0, 1975.0)
    one_down_centroid = (1025.0, 1925.0)

    # Default
    cg = grid.cell_geoms()
    assert isinstance(cg, np.ndarray)
    assert np.issubdtype(cg.dtype, np.object_)
    assert cg.shape == (6,)
    assert {g.geom_type for g in cg} == {"Polygon"}
    assert list(map(lambda g: g.is_valid, cg)) == [True] * 6
    assert list(map(lambda g: g.area, cg)) == [2500.0] * 6
    assert cg[0].exterior.coords[:] == first_poly_coords
    centroids = list(map(lambda g: g.centroid.coords[0], cg))
    assert first_centroid == centroids[0]
    assert one_right_centroid == centroids[1]
    assert one_down_centroid == centroids[3]

    # Fortran-style order
    cg = grid.cell_geoms(order="F")
    assert cg.shape == (6,)
    assert list(map(lambda g: g.is_valid, cg)) == [True] * 6
    assert list(map(lambda g: g.area, cg)) == [2500.0] * 6
    assert cg[0].exterior.coords[:] == first_poly_coords
    centroids = list(map(lambda g: g.centroid.coords[0], cg))
    assert first_centroid == centroids[0]
    assert one_down_centroid == centroids[1]
    assert one_right_centroid == centroids[2]

    # Mask with order options
    for order in ["C", "F"]:
        cg = grid.cell_geoms(order=order, mask=np.ones(grid.shape))
        assert isinstance(cg, np.ndarray)
        assert np.issubdtype(cg.dtype, np.object_)
        assert cg.shape == (0,)
        cg = grid.cell_geoms(mask=np.zeros(grid.shape), order=order)
        assert cg.shape == (6,)
        assert list(map(lambda g: g.is_valid, cg)) == [True] * 6
        assert list(map(lambda g: g.area, cg)) == [2500.0] * 6
        centroids = list(map(lambda g: g.centroid.coords[0], cg))
        assert first_centroid == centroids[0]
        if order == "C":
            assert one_right_centroid == centroids[1]
            assert one_down_centroid == centroids[3]
        elif order == "F":
            assert one_down_centroid == centroids[1]
            assert one_right_centroid == centroids[2]
        cg = grid.cell_geoms(mask=np.eye(2, 3, 1), order=order)
        assert cg.shape == (4,)
        centroids = list(map(lambda g: g.centroid.coords[0], cg))
        assert first_centroid == centroids[0]
        assert one_right_centroid not in centroids
        if order == "C":
            assert one_down_centroid == centroids[2]
        elif order == "F":
            assert one_down_centroid == centroids[1]
        cg = grid.cell_geoms(mask=~np.eye(2, 3, -1, bool), order=order)
        assert cg.shape == (1,)
        centroids = list(map(lambda g: g.centroid.coords[0], cg))
        assert centroids == [one_down_centroid]

    # errors
    with pytest.raises(ValueError, match='order must be "C" or "F"'):
        grid.cell_geoms(order="f")
    with pytest.raises(ValueError, match="mask must be an array the same sha"):
        grid.cell_geoms(mask=False)
    with pytest.raises(ValueError, match="mask must be an array the same sha"):
        grid.cell_geoms(mask=np.ones((3, 2)))


@requires_pkg("geopandas")
def test_cell_geoseries():
    import geopandas
    import pandas as pd

    grid = Grid(50.0, (2, 3), (1000.0, 2000.0), projection="EPSG:3857")
    one_right_centroid = (1075.0, 1975.0)
    one_down_centroid = (1025.0, 1925.0)

    gs = grid.cell_geoseries()
    assert isinstance(gs, geopandas.GeoSeries)
    assert gs.crs.to_epsg() == 3857
    assert gs.shape == (6,)
    assert gs.area.min() == 2500.0
    pd.testing.assert_index_equal(gs.index, pd.RangeIndex(6))
    assert gs[1].centroid.coords[0] == one_right_centroid

    grid = Grid(50.0, (2, 3), (1000.0, 2000.0))
    gs = grid.cell_geoseries(order="F")
    assert gs.crs is None
    assert gs.shape == (6,)
    pd.testing.assert_index_equal(gs.index, pd.RangeIndex(6))
    assert gs[1].centroid.coords[0] == one_down_centroid

    # Mask with order options
    for order in ["C", "F"]:
        gs = grid.cell_geoseries(order=order, mask=np.ones(grid.shape))
        assert gs.shape == (0,)
        gs = grid.cell_geoseries(order=order, mask=np.zeros(grid.shape))
        assert gs.shape == (6,)
        pd.testing.assert_index_equal(gs.index, pd.RangeIndex(6))
        if order == "C":
            assert gs[1].centroid.coords[0] == one_right_centroid
        elif order == "F":
            assert gs[1].centroid.coords[0] == one_down_centroid
        gs = grid.cell_geoseries(order=order, mask=np.eye(2, 3, 1))
        assert gs.shape == (4,)
        centroids = list(gs.centroid.apply(lambda g: g.coords[0]))
        if order == "C":
            assert one_right_centroid not in centroids
            assert one_down_centroid == centroids[2]
            pd.testing.assert_index_equal(gs.index, pd.Index([0, 2, 3, 4]))
        elif order == "F":
            assert one_right_centroid not in centroids
            assert one_down_centroid == centroids[1]
            pd.testing.assert_index_equal(gs.index, pd.Index([0, 1, 3, 4]))


@requires_pkg("geopandas")
def test_cell_geodataframe():
    import geopandas
    import pandas as pd

    grid = Grid(50.0, (2, 3), (1000.0, 2000.0), projection="EPSG:3857")

    gdf = grid.cell_geodataframe()
    assert isinstance(gdf, geopandas.GeoDataFrame)
    assert gdf.crs.to_epsg() == 3857
    assert gdf.shape == (6, 3)
    assert gdf.area.min() == 2500.0
    assert list(gdf.columns) == ["geometry", "row", "col"]
    pd.testing.assert_index_equal(gdf.index, pd.RangeIndex(6))
    ra_row = pd.Series(np.repeat(np.arange(2), 3).astype(np.int64), name="row")
    ta_col = pd.Series(np.tile(np.arange(3), 2).astype(np.int64), name="col")
    pd.testing.assert_series_equal(gdf["row"], ra_row)
    pd.testing.assert_series_equal(gdf["col"], ta_col)

    grid = Grid(50.0, (2, 3), (1000.0, 2000.0))
    gdf = grid.cell_geodataframe(order="F")
    assert gdf.crs is None
    assert gdf.shape == (6, 3)
    assert list(gdf.columns) == ["geometry", "row", "col"]
    pd.testing.assert_index_equal(gdf.index, pd.RangeIndex(6))
    ta_row = pd.Series(np.tile(np.arange(2), 3).astype(np.int64), name="row")
    ra_col = pd.Series(np.repeat(np.arange(3), 2).astype(np.int64), name="col")
    pd.testing.assert_series_equal(gdf["row"], ta_row)
    pd.testing.assert_series_equal(gdf["col"], ra_col)

    ar = np.arange(6).reshape(grid.shape) * 2.0 + 1
    # Values, mask with order options
    for order in ["C", "F"]:
        gdf = grid.cell_geodataframe(
            order=order, mask=np.ones(grid.shape), values={"a": ar}
        )
        assert gdf.shape == (0, 4)
        assert list(gdf.columns) == ["geometry", "row", "col", "a"]
        gdf = grid.cell_geodataframe(
            order=order, mask=np.zeros(grid.shape), values={"a": ar}
        )
        assert gdf.shape == (6, 4)
        assert list(gdf.columns) == ["geometry", "row", "col", "a"]
        pd.testing.assert_index_equal(gdf.index, pd.RangeIndex(6))
        pd.testing.assert_series_equal(
            gdf["a"], pd.Series(ar.ravel(order=order), name="a")
        )
        if order == "C":
            pd.testing.assert_series_equal(gdf["row"], ra_row)
            pd.testing.assert_series_equal(gdf["col"], ta_col)
        elif order == "F":
            pd.testing.assert_series_equal(gdf["row"], ta_row)
            pd.testing.assert_series_equal(gdf["col"], ra_col)
        gdf = grid.cell_geodataframe(
            order=order, mask=np.eye(2, 3, 1), values={"a": ar}
        )
        assert gdf.shape == (4, 4)
        assert list(gdf.columns) == ["geometry", "row", "col", "a"]
        if order == "C":
            idx = pd.Index([0, 2, 3, 4])
            pd.testing.assert_index_equal(gdf.index, idx)
            pd.testing.assert_series_equal(
                gdf["row"], pd.Series([0, 0, 1, 1], name="row", index=idx)
            )
            pd.testing.assert_series_equal(
                gdf["col"], pd.Series([0, 2, 0, 1], name="col", index=idx)
            )
            pd.testing.assert_series_equal(
                gdf["a"], pd.Series([1.0, 5.0, 7.0, 9.0], name="a", index=idx)
            )
        elif order == "F":
            idx = pd.Index([0, 1, 3, 4])
            pd.testing.assert_index_equal(gdf.index, idx)
            pd.testing.assert_series_equal(
                gdf["row"], pd.Series([0, 1, 1, 0], name="row", index=idx)
            )
            pd.testing.assert_series_equal(
                gdf["col"], pd.Series([0, 0, 1, 2], name="col", index=idx)
            )
            pd.testing.assert_series_equal(
                gdf["a"], pd.Series([1.0, 7.0, 9.0, 5.0], name="a", index=idx)
            )

    # errors
    with pytest.raises(ValueError, match="values must be dict"):
        grid.cell_geodataframe(values=False)
    with pytest.raises(ValueError, match="key for values must be str"):
        grid.cell_geodataframe(values={False: np.ones(grid.shape)})
    with pytest.raises(ValueError, match="key for values must be str"):
        grid.cell_geodataframe(values={False: np.ones(grid.shape)})
    with pytest.raises(ValueError, match="array 'a' in values must have the"):
        grid.cell_geodataframe(values={"a": np.ones((3, 2))})
