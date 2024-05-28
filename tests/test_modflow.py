import logging

import pytest

from gridit import Grid

from .conftest import datadir, requires_pkg

modflow_dir = datadir / "modflow"


@requires_pkg("flopy")
def test_get_modflow_model():
    import flopy

    from gridit.modflow import get_modflow_model

    m = get_modflow_model(modflow_dir / "h.nam")
    assert isinstance(m, flopy.modflow.Modflow)
    assert m.get_package_list() == ["DIS", "BAS6"]

    m = get_modflow_model(m)
    assert isinstance(m, flopy.modflow.Modflow)

    with pytest.warns(UserWarning):
        m = get_modflow_model(modflow_dir)
        assert isinstance(m, flopy.mf6.MFModel)

    with pytest.warns(UserWarning):
        m = get_modflow_model(modflow_dir / "mfsim.nam")
        assert isinstance(m, flopy.mf6.MFModel)

    m = get_modflow_model(modflow_dir / "mfsim.nam", "h6")
    assert isinstance(m, flopy.mf6.MFModel)
    assert m.get_package_list() == ["DIS"]
    assert hasattr(m, "tdis")  # check hack

    m = get_modflow_model(m)
    assert isinstance(m, flopy.mf6.MFModel)


@requires_pkg("flopy")
def test_grid_from_modflow_classic():
    grid = Grid.from_modflow(modflow_dir / "h.nam")
    expected = Grid(1000.0, (18, 17), (1802000.0, 5879000.0), "EPSG:2193")
    assert grid == expected
    assert grid.projection == "EPSG:2193"


@requires_pkg("flopy")
def test_grid_from_modflow_6(caplog):
    from gridit.modflow import get_modflow_model

    expected = Grid(1000.0, (18, 17), (1802000.0, 5879000.0))

    with caplog.at_level(logging.WARNING):
        grid = Grid.from_modflow(modflow_dir / "mfsim.nam", "h6")
        assert len(caplog.messages) == 0
        assert grid == expected
        assert grid.projection is None

    grid = Grid.from_modflow(modflow_dir / "mfsim.nam", "h6", "EPSG:2193")
    # assert grid == expected
    assert grid.projection == "EPSG:2193"

    with caplog.at_level(logging.WARNING):
        grid = Grid.from_modflow(modflow_dir / "mfsim.nam")
        assert "a model name should be specified" in caplog.messages[-1]
        assert grid == expected
        assert grid.projection is None

    # also rasises logger warning
    grid = Grid.from_modflow(modflow_dir)
    assert grid == expected

    # test with modelgrid object
    with pytest.warns(UserWarning, match="model name should be specified"):
        model = get_modflow_model(modflow_dir / "mfsim.nam")
    grid = Grid.from_modflow(model.modelgrid)
    assert grid.projection is None
    assert grid.shape == (18, 17)
    assert grid.resolution == 1000.0


@requires_pkg("flopy")
def test_mask_from_modflow_classic():
    grid = Grid.from_modflow(modflow_dir)
    mask = grid.mask_from_modflow(modflow_dir)
    assert mask.sum() == 128
    mask = grid.mask_from_modflow(modflow_dir / "mfsim.nam")
    assert mask.sum() == 128
    grid = Grid.from_modflow(modflow_dir)
    mask = grid.mask_from_modflow(modflow_dir / "mfsim.nam", "h6")
    assert mask.sum() == 128


@requires_pkg("flopy")
def test_mask_from_modflow_6():
    grid = Grid.from_modflow(modflow_dir / "h.nam")
    mask = grid.mask_from_modflow(modflow_dir / "h.nam")
    assert mask.sum() == 128
