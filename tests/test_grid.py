import pytest

from gridit import Grid

from .conftest import requires_pkg


@pytest.fixture
def grid_basic():
    return Grid(10, (20, 30), (1000.0, 2000.0))


def test_grid_basic(grid_basic):
    grid = grid_basic
    assert isinstance(grid, Grid)
    assert grid.resolution == 10.0
    assert grid.shape == (20, 30)
    assert grid.top_left == (1000.0, 2000.0)
    assert grid.projection is None


def test_grid_dict(grid_basic):
    grid_d = dict(grid_basic)
    assert list(grid_d.keys()) == ["resolution", "shape", "top_left", "projection"]
    assert grid_d["resolution"] == 10.0
    assert grid_d["shape"] == (20, 30)
    assert grid_d["top_left"] == (1000.0, 2000.0)
    assert grid_d["projection"] is None


def test_grid_repr(grid_basic):
    expected = "<Grid: resolution=10.0, shape=(20, 30), " "top_left=(1000.0, 2000.0) />"
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

    assert grid_basic.transform == Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
