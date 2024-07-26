"""Modflow methods."""

from dataclasses import dataclass, field
from importlib.util import find_spec
from os import PathLike
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union
from warnings import catch_warnings, filterwarnings, warn

import numpy as np
import numpy.typing as npt

from gridit.grid import mask_cache
from gridit.logger import get_logger, logger_factory


@dataclass
class ModelGrid:
    """Light-weight modelgrid object."""

    delr: npt.NDArray[np.floating] = field(repr=False)
    delc: npt.NDArray[np.floating] = field(repr=False)
    domain: npt.NDArray[np.integer] = field(repr=False)
    xoffset: float = 0.0
    yoffset: float = 0.0
    rotation: float = 0.0
    projection: Optional[str] = None
    logger: npt.NDArray[np.integer] = field(default_factory=logger_factory, repr=False)

    def __post_init__(self):
        self.logger = get_logger(self.__class__.__name__)

        self.delr = np.asarray(self.delr)
        if self.delr.ndim != 1:
            raise ValueError("model delr is not a 1-d array")
        delr = self.delr[0]
        if not (self.delr == delr).all():
            raise ValueError("model delr is not constant")

        self.delc = np.asarray(self.delc)
        if self.delc.ndim != 1:
            raise ValueError("model delc is not a 1-d array")
        delc = self.delc[0]
        if not (self.delc == delc).all():
            raise ValueError("model delc is not constant")
        elif delr != delc:
            raise ValueError("model delr and delc are different")

        if self.rotation != 0:
            self.logger.error("rotated model grids are not supported")

        self.domain = np.asarray(self.domain)
        if self.domain.ndim == 1:
            nrow = self.delc.size
            ncol = self.delr.size
            nlay = self.domain.size // (nrow * ncol)
            self.domain.shape = (nlay, nrow, ncol)
        elif self.domain.ndim not in (2, 3):
            raise ValueError("model domain is not a 2- or 3-d array")

    @classmethod
    def from_modflow(
        cls, file_or_dir: Union[str, PathLike], model_name=None, logger=None
    ):
        """Create from MODFLOW file or directory."""
        m = get_modflow_model(file_or_dir, model_name=model_name)
        return cls.from_modelgrid(m.modelgrid, logger=logger)

    @classmethod
    def from_modelgrid(cls, mg, logger=None):
        """Create from flopy's modelgrid object."""
        attrs = ["delr", "delc", "idomain", "xoffset", "yoffset", "angrot", "crs"]
        mg_name = {"idomain": "domain", "angrot": "rotation", "crs": "projection"}
        kwargs = {mg_name.get(attr, attr): getattr(mg, attr) for attr in attrs}
        if kwargs["projection"] is None:
            epsg = getattr(mg, "epsg")
            if isinstance(epsg, int):
                kwargs["projection"] = f"EPSG:{epsg}"
            else:
                kwargs["projection"] = getattr(mg, "proj4")
        return cls(**kwargs, logger=logger)

    @property
    def top_left(self):
        return (self.xoffset, self.yoffset + np.sum(self.delc))

    @property
    def shape(self):
        return self.domain.shape[-2:]

    def to_grid(self):
        """Returns Grid object."""
        from gridit.grid import Grid

        return Grid(
            resolution=self.delr[0].item(),
            shape=self.shape,
            top_left=self.top_left,
            projection=self.projection,
            logger=self.logger,
        )


def get_modflow_model(model, model_name=None, logger=None):
    """Return model object from str or Path.

    Parameters
    ----------
    model : str or PathLike
        Path to a MODFLOW 6 file (mfsim.nam or TODO)
    """
    import flopy

    if hasattr(model, "xoffset"):
        tmpobj = SimpleNamespace()
        tmpobj.modelgrid = model
        return tmpobj  # dummy object with a modelgrid atrib
    if hasattr(model, "modelgrid"):
        return model
    elif not isinstance(model, (str, Path)):
        raise TypeError("expected str, Path or model instance")
    pth = Path(model).resolve()
    if not pth.exists():
        raise ValueError(f"cannot read path '{pth}'")
    elif pth.suffixes[-2:] == [".dis", ".grb"]:
        # Binary grid file
        if logger is not None:
            logger.info("reading grid from a binary grid file: %s", pth)
        if model_name:
            if logger is not None:
                logger.warning("ignoring model_name '%s'", model_name)
        return flopy.mf6.utils.MfGrdFile(pth)
    elif (pth.is_dir() and (pth / "mfsim.nam").is_file()) or pth.name == "mfsim.nam":
        # MODFLOW 6
        sim_ws = str(pth) if pth.is_dir() else str(pth.parent)
        if logger is not None:
            logger.info("reading mf6 simulation from '%s'", sim_ws)
        sim = flopy.mf6.MFSimulation.load(
            sim_ws=sim_ws, strict=False, verbosity_level=0, load_only=["dis", "tdis"]
        )
        model_names = list(sim.model_names)
        if model_name is None:
            model_name = model_names[0]
            msg = "a model name should be specified, "
            if len(model_names) > 1:
                msg += "mfsim.nam has %d models (%s); selecting first"
                args = (len(model_names), model_names)
            else:
                msg += "selecting %r from mfsim.nam"
                args = (model_name,)
            if logger is None:
                warn(msg % args, UserWarning, stacklevel=2)
            else:
                logger.warning(msg, *args)
        elif model_name not in model_names:
            raise KeyError(f"model name {model_name} not found in {model_names}")
        model = sim.get_model(model_name)
        model.tdis = sim.tdis  # this is a bit of a hack
        return model
    elif pth.is_file():  # assume 'classic' MOFLOW file
        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            try:
                model = flopy.modflow.Modflow.load(
                    pth.name,
                    model_ws=str(pth.parent),
                    load_only=["dis", "bas6"],
                    check=False,
                    verbose=False,
                    forgive=True,
                )
            except UnicodeDecodeError:
                raise ValueError(f"cannot open MODFLOW file '{pth}'")
        return model
    raise ValueError(f"cannot determine how to read MODFLOW model '{pth}'")


@classmethod
def from_modflow(cls, model, model_name=None, projection=None, logger=None):
    """Create grid information from a MODFLOW model.

    Parameters
    ----------
    model : str, PathLike, flopy.modflow.Modflow, flopy.mf6.mfmodel.MFModel or flopy.discretization.grid
        MODFLOW model specified either as a FloPy object, or path to
        a MODFLOW file (either TODO).
    model_name : str or None (default)
        Needed if MODFLOW 6 simulation has more than one model.
    projection : str, default None
        Coordinate reference system described as a string either as (e.g.)
        EPSG:2193 or a WKT string. Default None will attempt to obtain
        this from the model, otherwise it will be "".
    logger : logging.Logger, optional
        Logger to show messages.

    Raises
    ------
    ModuleNotFoundError
        If flopy is not installed.

    """  # noqa
    if find_spec("flopy") is None:
        raise ModuleNotFoundError("from_modflow requires flopy")
    if logger is None:
        logger = get_logger(cls.__name__)
    logger.info(
        "creating from a MODFLOW model: %s",
        model if isinstance(model, (str, PathLike)) else type(model),
    )
    mask_cache_key = (repr(model), model_name)
    model = get_modflow_model(model, model_name, logger)
    modelgrid = ModelGrid.from_modelgrid(model.modelgrid)
    modelgrid = model.modelgrid
    delr = modelgrid.delr[0]
    delc = modelgrid.delc[0]
    if not (modelgrid.delr == delr).all():
        raise ValueError("model delr is not constant")
    elif not (modelgrid.delc == delc).all():
        raise ValueError("model delc is not constant")
    elif delr != delc:
        raise ValueError("model delr and delc are different")
    if modelgrid.angrot != 0:
        logger.error("rotated model grids are not supported")
    top_left = (modelgrid.xoffset, modelgrid.yoffset + modelgrid.delc.sum())
    if projection is None:
        if isinstance(modelgrid.epsg, int):
            projection = f"EPSG:{modelgrid.epsg}"
        elif modelgrid.proj4 is not None:
            projection = modelgrid.proj4
        else:
            projection = ""
    # also cache mask while we are here
    domain = modelgrid.idomain
    mask = (domain == 0).all(0)
    mask_cache[mask_cache_key] = mask
    return cls(
        resolution=delr,
        shape=modelgrid.top.shape,
        top_left=top_left,
        projection=projection,
        logger=logger,
    )


def mask_from_modflow(self, model, model_name=None):
    """Return a mask array from a MODFLOW model, based on IBOUND/IDOMAIN.

    Parameters
    ----------
    model : str, Path, flopy.modflow.Modflow, or flopy.mf6.mfmodel.MFModel
        MODFLOW model specified either as a FloPy object, or path to
        a MODFLOW file.
    model_name : str or None (default)
        Needed if MODFLOW 6 simulation has more than one model.

    Returns
    -------
    np.array

    """
    mask_cache_key = (repr(model), model_name)
    if mask_cache_key in mask_cache:
        self.logger.info("using mask from cache")
        return mask_cache[mask_cache_key]
    model = get_modflow_model(model, model_name, self.logger)
    if hasattr(model, "bas6"):
        domain = model.bas6.ibound.array
    else:
        domain = model.dis.idomain.array
    mask = (domain == 0).all(0)
    mask_cache[mask_cache_key] = mask
    return mask
