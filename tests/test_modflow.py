import logging
from types import SimpleNamespace

import numpy as np
import pytest

from gridit import Grid

from .common import datadir, get_str_list_count, requires_pkg

modflow_dir = datadir / "modflow"


@requires_pkg("flopy")
def test_get_modflow_model():
    import flopy

    from gridit.modflow import get_modflow_model

    # classic MODFLOW
    m = get_modflow_model(modflow_dir / "h.nam")
    assert isinstance(m, flopy.modflow.Modflow)
    assert m.get_package_list() == ["DIS", "BAS6"]

    with pytest.deprecated_call():
        m = get_modflow_model(m)
    assert isinstance(m, flopy.modflow.Modflow)

    # MODFLOW 6
    with pytest.warns(UserWarning, match="model_name should be specified"):
        m = get_modflow_model(modflow_dir)
    assert isinstance(m, flopy.mf6.MFModel)

    with pytest.warns(UserWarning, match="model_name should be specified"):
        m = get_modflow_model(modflow_dir / "mfsim.nam")
    assert isinstance(m, flopy.mf6.MFModel)

    m = get_modflow_model(modflow_dir / "mfsim.nam", "h6")
    assert isinstance(m, flopy.mf6.MFModel)
    assert m.get_package_list() == ["DIS"]
    assert hasattr(m, "tdis")  # check hack

    with pytest.deprecated_call():
        m = get_modflow_model(m)
    assert isinstance(m, flopy.mf6.MFModel)

    m = get_modflow_model(modflow_dir / "h6.dis.grb")
    assert isinstance(m, SimpleNamespace)
    assert hasattr(m, "modelgrid")

    with pytest.warns(UserWarning, match="ignoring model_name"):
        m = get_modflow_model(modflow_dir / "h6.dis.grb", "unused")
    assert isinstance(m, SimpleNamespace)
    assert hasattr(m, "modelgrid")

    # Check exceptions
    with pytest.raises(TypeError, match="expected str or PathLike object"):
        get_modflow_model(1)
    with pytest.raises(FileNotFoundError, match="cannot read path"):
        get_modflow_model("not-a-path")
    with pytest.raises(KeyError, match="model name h not found"):
        get_modflow_model(modflow_dir / "mfsim.nam", "h")
    with pytest.raises(ValueError, match="cannot open MODFLOW file"):
        get_modflow_model(modflow_dir / "h.dis")
    with pytest.raises(ValueError, match="cannot determine how to read MODFLOW model"):
        get_modflow_model(datadir)


@requires_pkg("flopy")
def test_grid_from_modflow_classic():
    grid = Grid.from_modflow(modflow_dir / "h.nam")
    expected = Grid(1000.0, (18, 17), (1802000.0, 5879000.0), "EPSG:2193")
    assert grid == expected
    assert grid.projection == "EPSG:2193"


@requires_pkg("flopy")
def test_grid_from_modflow_6(caplog):
    caplog.set_level(logging.WARNING)
    expected = Grid(1000.0, (18, 17), (1802000.0, 5879000.0))
    expected_wp = Grid(1000.0, (18, 17), (1802000.0, 5879000.0), projection="EPSG:2193")

    grid = Grid.from_modflow(modflow_dir, "h6")
    assert len(caplog.messages) == 0
    assert grid == expected

    grid = Grid.from_modflow(modflow_dir / "mfsim.nam", "h6", "EPSG:2193")
    assert len(caplog.messages) == 0
    assert grid == expected_wp

    caplog.clear()
    grid = Grid.from_modflow(modflow_dir)
    assert get_str_list_count("model_name should be specified", caplog.messages) == 1
    assert grid == expected

    caplog.clear()
    grid = Grid.from_modflow(modflow_dir / "h6.dis.grb")
    assert len(caplog.messages) == 0
    assert grid == expected

    caplog.clear()
    grid = Grid.from_modflow(modflow_dir / "h6.dis.grb", "unused", "EPSG:2193")
    assert get_str_list_count("ignoring model_name", caplog.messages) == 1
    assert grid == expected_wp


@requires_pkg("flopy")
def test_mask_from_modflow_classic():
    grid = Grid.from_modflow(modflow_dir / "h.nam")
    mask = grid.mask_from_modflow(modflow_dir / "h.nam")
    assert mask.sum() == 128


@requires_pkg("flopy")
def test_mask_from_modflow_6():
    grid = Grid.from_modflow(modflow_dir)
    mask = grid.mask_from_modflow(modflow_dir)
    assert mask.sum() == 128
    mask = grid.mask_from_modflow(modflow_dir / "mfsim.nam")
    assert mask.sum() == 128
    grid = Grid.from_modflow(modflow_dir)
    mask = grid.mask_from_modflow(modflow_dir / "mfsim.nam", "h6")
    assert mask.sum() == 128
    mask = grid.mask_from_modflow(modflow_dir / "h6.dis.grb")
    assert mask.sum() == 128


def test_modelgrid():
    from gridit.modflow import ModelGrid

    # 2D grid
    mg = ModelGrid(
        [10, 10],
        [10, 10, 10],
        np.ones((2, 3)),
        xoffset=100,
        yoffset=200,
        projection="EPSG:2193",
    )
    assert mg.delr.shape == (2,)
    assert mg.delc.shape == (3,)
    assert mg.domain.shape == (2, 3)
    assert mg.to_grid() == Grid(10, (2, 3), (100.0, 230.0), "EPSG:2193")
    # 3D grid, no projection
    mg = ModelGrid([10, 10], [10, 10, 10], np.ones((4, 2, 3)), xoffset=100, yoffset=200)
    assert mg.delr.shape == (2,)
    assert mg.delc.shape == (3,)
    assert mg.domain.shape == (4, 2, 3)
    assert mg.to_grid() == Grid(10, (2, 3), (100.0, 230.0))


@requires_pkg("flopy")
def test_modelgrid_flopy():
    import flopy

    from gridit.modflow import ModelGrid

    expected = Grid(1000.0, (18, 17), (1802000.0, 5879000.0))
    expected_wp = Grid(1000.0, (18, 17), (1802000.0, 5879000.0), projection="EPSG:2193")

    m = flopy.modflow.Modflow.load("h.nam", model_ws=modflow_dir, check=False)
    mg = ModelGrid.from_modelgrid(m.modelgrid)
    assert mg.to_grid() == expected_wp

    sim = flopy.mf6.MFSimulation.load(sim_ws=modflow_dir)
    gwf = sim.get_model()
    mg = ModelGrid.from_modelgrid(gwf.modelgrid)
    assert mg.to_grid() == expected

    mg = ModelGrid.from_modelgrid(gwf.modelgrid, projection="EPSG:2193")
    assert mg.to_grid() == expected_wp

    mg = ModelGrid.from_modflow(modflow_dir / "h.nam")
    assert mg.to_grid() == expected_wp

    mg = ModelGrid.from_modflow(modflow_dir, "h6")
    assert mg.to_grid() == expected

    mg = ModelGrid.from_modflow(modflow_dir / "mfsim.nam", "h6")
    assert mg.to_grid() == expected

    mg = ModelGrid.from_modflow(modflow_dir / "h6.dis.grb")
    assert mg.to_grid() == expected

    mg = ModelGrid.from_modflow(modflow_dir / "mfsim.nam", "h6", "EPSG:2193")
    assert mg.to_grid() == expected_wp
