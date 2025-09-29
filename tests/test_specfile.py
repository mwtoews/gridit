import logging
from io import StringIO
from textwrap import dedent

import pytest

from gridit import Grid

from .common import datadir


def test_from_specfile(caplog):
    caplog.set_level(logging.WARNING)
    caplog.clear()
    pth = datadir / "pest.spc"

    grid = Grid.from_specfile(pth)
    assert grid == Grid(50.0, (8, 10), (300000.0, 7800000.0))

    grid = Grid.from_specfile(str(pth), projection="EPSG:2193")
    assert grid == Grid(50.0, (8, 10), (300000.0, 7800000.0), "EPSG:2193")

    obj = StringIO(
        dedent("""\
        1, 2, ignored
        200, 300, 0e-0, ignored
        10, 10
        10
        ignored
    """)
    )
    grid = Grid.from_specfile(obj)
    assert grid == Grid(10.0, (1, 2), (200.0, 300.0))
    assert len(caplog.messages) == 0

    obj = StringIO(
        dedent("""\
        2\t2
        200\t300\t0
        10\t10
        10
    """)
    )
    with pytest.raises(ValueError, match=r"expected 2 item\(s\) but found 1"):
        Grid.from_specfile(obj)
    assert len(caplog.messages) == 0

    obj = StringIO(
        dedent("""\
        2 3
        200. 300. 4.
        10 10
        10 10
        10 20
    """)
    )
    grid = Grid.from_specfile(obj)
    assert grid == Grid(10.0, (2, 3), (200.0, 300.0))
    assert caplog.messages == [
        "too many items gathered; trimming",
        "rotated specfile grids are not supported",
        "specfile delc is not constant 10.0",
    ]


def test_write_specfile(tmp_path):
    grid = Grid(50.0, (8, 10), (300000.0, 7800000.0))
    assert grid.write_specfile(None) == dedent("""\
        8 10
        300000.0 7800000.0 0.0
        10*50.0
        8*50.0
    """)
    pth = tmp_path / "out.spc"
    grid.write_specfile(pth)
    assert grid == Grid.from_specfile(pth)
    grid = Grid(10.0, (1, 2), (200.0, 300.0))
    obj = StringIO()
    grid.write_specfile(obj)
    obj.seek(0)
    assert obj.read() == dedent("""\
        1 2
        200.0 300.0 0.0
        2*10.0
        10.0
    """)
