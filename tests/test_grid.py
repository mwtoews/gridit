import pickle

import pytest

from gridit import Grid

from .common import requires_pkg


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


@pytest.mark.parametrize(
    "grid, expected_str",
    [
        pytest.param(
            Grid(10, (20, 30), (1000.0, 2000.0)),
            "<Grid: resolution=10.0, shape=(20, 30), top_left=(1000.0, 2000.0) />",
            id="without projection",
        ),
        pytest.param(
            Grid(10.0, (20, 30), (1000.0, 2000.0), "EPSG:2193"),
            "<Grid: resolution=10.0, shape=(20, 30), top_left=(1000.0, 2000.0) />",
            id="with projection",
        ),
    ],
)
def test_grid_repr(grid, expected_str):
    assert repr(grid) == expected_str
    assert str(grid) == expected_str


def test_grid_eq_hash():
    grid1 = Grid(10, (20, 30), (1000.0, 2000.0))
    grid2 = Grid(10, (20, 30), (1001.0, 2000.0))
    grid3 = Grid(10, (20, 30), (1000.0, 2000.0), "EPSG:2193")
    grid1r = Grid(10, (20, 30), (1000.0, 2000.0))
    assert grid1 != grid2
    assert grid1 != grid3, "projection is different"
    assert grid1 == grid1r
    hash1 = hash(grid1)
    hash2 = hash(grid2)
    hash3 = hash(grid3)
    hash1r = hash(grid1r)
    assert hash1 != hash2
    assert hash1 != hash3, "projection is different"
    assert hash1 == hash1r


def test_grid_bounds(grid_basic):
    assert grid_basic.bounds == (1000.0, 1800.0, 1300.0, 2000.0)


def test_grid_corner_coords(grid_basic):
    assert grid_basic.corner_coords == [
        (1000.0, 2000.0),
        (1000.0, 1800.0),
        (1300.0, 1800.0),
        (1300.0, 2000.0),
    ]


@requires_pkg("affine")
def test_grid_transform(grid_basic):
    from affine import Affine

    assert grid_basic.transform == Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)


@pytest.mark.parametrize(
    "grid, expected_bytes",
    [
        pytest.param(
            Grid(25.0, (36, 33), (1748725.0, 5449775.0)),
            b"\x80\x04\x95c\x00\x00\x00\x00\x00\x00\x00\x8c\x0b"
            b"gridit.grid\x94\x8c\x04Grid\x94\x93\x94)\x81\x94}\x94(\x8c\n"
            b"resolution\x94G@9\x00\x00\x00\x00\x00\x00\x8c\x05"
            b"shape\x94K$K!\x86\x94\x8c\x08"
            b"top_left\x94GA:\xae\xf5\x00\x00\x00\x00GAT\xca\x0b\xc0\x00\x00\x00\x86\x94ub.",
            id="without projection",
        ),
        pytest.param(
            Grid(25.0, (36, 33), (1748725.0, 5449775.0), projection="EPSG:2193"),
            b"\x80\x04\x95|\x00\x00\x00\x00\x00\x00\x00\x8c\x0b"
            b"gridit.grid\x94\x8c\x04Grid\x94\x93\x94)\x81\x94}\x94(\x8c\n"
            b"resolution\x94G@9\x00\x00\x00\x00\x00\x00\x8c\x05"
            b"shape\x94K$K!\x86\x94\x8c\x08"
            b"top_left\x94GA:\xae\xf5\x00\x00\x00\x00GAT\xca\x0b\xc0\x00\x00\x00\x86"
            b"\x94\x8c\nprojection\x94\x8c\tEPSG:2193\x94ub.",
            id="with projection",
        ),
    ],
)
def test_pickle(grid, expected_bytes):
    assert pickle.loads(expected_bytes) == grid, "failed loading previous serialization"
    assert pickle.loads(pickle.dumps(grid)) == grid, "failed round-trip"
