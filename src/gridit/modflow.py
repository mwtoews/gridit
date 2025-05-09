"""Modflow methods."""

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import SimpleNamespace
from typing import Any
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
    projection: str | None = None
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
        if delr != delc:
            raise ValueError("model delr and delc are different")

        if self.rotation != 0:
            self.logger.error("rotated model grids are not supported")

        self.domain = np.asarray(self.domain)
        if self.domain.ndim == 1:
            # reshape into 3-d array
            nrow = self.delc.size
            ncol = self.delr.size
            nlay = self.domain.size // (nrow * ncol)
            self.domain.shape = (nlay, nrow, ncol)
        elif self.domain.ndim not in (2, 3):
            raise ValueError("model domain is not a 2- or 3-d array")

    @classmethod
    def from_modflow(
        cls,
        file_or_dir: str | PathLike,
        model_name: str | None = None,
        projection: str | None = None,
        logger=None,
    ):
        """Create from MODFLOW file or directory.

        Parameters
        ----------
        file_or_dir : str or PathLike
            Path to a MODFLOW 6 directory, mfsim.nam, binary grid file
            (with .dis.grb suffix), or classic MODFLOW NAM file.
        model_name : str, optional
            Model name; ignored with binary grid or classic MODFLOW NAM files.
        projection : optional str, default None
            WKT coordinate reference system string. If not provided, then try to
            obtain from model data.
        logger : logging.Logger, optional
            Logger to show messages.
        """
        model = get_modflow_model(file_or_dir, model_name=model_name)
        return cls.from_modelgrid(model.modelgrid, projection=projection, logger=logger)

    @classmethod
    def from_modelgrid(cls, mg: Any, projection: str | None = None, logger=None):
        """Create from a modelgrid-like object.

        Parameters
        ----------
        mg : flopy.discretization.StructuredGrid or similar
            Object with attributes "delr", "delc", "idomain", "xoffset",
            "yoffset", "angrot", and "crs".
        projection : optional str, default None
            WKT coordinate reference system string. If not provided, then try to
            obtain from model data.
        logger : logging.Logger, optional
            Logger to show messages.
        """
        attrs = ["delr", "delc", "idomain", "xoffset", "yoffset", "angrot", "crs"]
        mg_name = {"idomain": "domain", "angrot": "rotation", "crs": "projection"}
        kwargs = {mg_name.get(attr, attr): getattr(mg, attr) for attr in attrs}
        if projection is not None:
            kwargs["projection"] = projection
        elif kwargs["projection"] is None:
            epsg = getattr(mg, "epsg")
            if isinstance(epsg, int):
                kwargs["projection"] = f"EPSG:{epsg}"
            else:
                kwargs["projection"] = getattr(mg, "proj4")
        return cls(**kwargs, logger=logger)

    def _grid_kwargs(self):
        """Returns dict of keyword parameters for Grid.__init__."""
        return dict(
            resolution=self.delr[0].item(),
            shape=self.domain.shape[-2:],
            top_left=(self.xoffset, self.yoffset + np.sum(self.delc)),
            projection=self.projection,
            logger=self.logger,
        )

    def to_grid(self):
        """Returns Grid object."""
        from gridit.grid import Grid

        return Grid(**self._grid_kwargs())


def get_modflow_model(
    file_or_dir: str | PathLike, model_name: str | None = None, logger=None
):
    """Return model object from a str or Path to MODFLOW files.

    Parameters
    ----------
    file_or_dir : str or PathLike
        Path to a MODFLOW 6 directory, mfsim.nam, binary grid file
        (with .dis.grb suffix), or classic MODFLOW NAM file.
    model_name : str, optional
        Model name; ignored with binary grid or classic MODFLOW NAM files.
    logger : logging.Logger, optional
        Logger to show messages.

    Raises
    ------
    ModuleNotFoundError
        If flopy is not installed.

    Returns
    -------
    Several types of objects, but should have a "modelgrid" attribute.

    """
    import flopy

    if hasattr(file_or_dir, "xoffset"):  # this is a modelgrid-like object
        warn(
            "getting modflow model from a modelgrid-like object is deprecated; "
            "use ModelGrid.from_modelgrid(obj) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return SimpleNamespace(modelgrid=file_or_dir)
    if hasattr(file_or_dir, "modelgrid"):  # this is a flopy-like object
        warn(
            "getting modflow model from a flopy-like object is deprecated; "
            "use ModelGrid.from_modelgrid(obj.modelgrid) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return file_or_dir
    if not isinstance(file_or_dir, str | PathLike):
        raise TypeError(f"expected str or PathLike object; found {type(file_or_dir)}")

    pth = Path(file_or_dir).resolve()
    if not pth.exists():
        raise FileNotFoundError(f"cannot read path '{pth}'")
    if pth.suffixes[-2:] == [".dis", ".grb"]:
        # Binary grid file
        if logger is not None:
            logger.info("reading grid from a binary grid file: %s", pth)
        if model_name:
            msg = "ignoring model_name '%s'"
            args = (model_name,)
            if logger is None:
                warn(msg % args, UserWarning, stacklevel=2)
            else:
                logger.warning(msg, *args)
        grb = flopy.mf6.utils.MfGrdFile(pth)
        return SimpleNamespace(modelgrid=grb.modelgrid)
    if (pth.is_dir() and (pth / "mfsim.nam").is_file()) or pth.name == "mfsim.nam":
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
            msg = "model_name should be specified, "
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
    if pth.is_file():  # assume 'classic' MOFLOW file
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
            except (ValueError, UnicodeDecodeError):
                raise ValueError(f"cannot open MODFLOW file '{pth}'")
        return model
    raise ValueError(f"cannot determine how to read MODFLOW model '{pth}'")


# imported from Grid class
@classmethod
def from_modflow(cls, model, model_name=None, projection=None, logger=None):
    """Create grid information from a MODFLOW model.

    Parameters
    ----------
    model : str, PathLike, flopy.modflow.Modflow, flopy.mf6.mfmodel.MFModel or flopy.discretization.StructuredGrid
        Path to a MODFLOW 6 directory, mfsim.nam, binary grid file
        (with .dis.grb suffix), classic MODFLOW NAM file,
        or certain FloPy objects.
    model_name : str, optional
        Model name; ignored with binary grid or classic MODFLOW NAM files.
    projection : str, optional
        Coordinate reference system described as a string either as (e.g.)
        EPSG:2193 or a WKT string. Default None will attempt to obtain
        this from the model, otherwise it will be None.
    logger : logging.Logger, optional
        Logger to show messages.

    Raises
    ------
    ModuleNotFoundError
        Reading from str or PathLike requires flopy.

    """  # noqa
    if logger is None:
        logger = get_logger(cls.__name__)
    if isinstance(model, str | PathLike):
        pth = Path(model)
        logger.info(
            "creating grid from a MODFLOW model %s: %s",
            "directory" if pth.is_dir() else "file",
            model,
        )
        # also cache mask while we are here
        mask_cache_key = (repr(model), model_name)
        model = get_modflow_model(model, model_name, logger)
        mg = ModelGrid.from_modelgrid(model.modelgrid)
        domain = mg.domain
        mask = (domain == 0).all(0)
        mask_cache[mask_cache_key] = mask
    else:
        logger.info(
            "creating grid from a MODFLOW model object: %s",
            type(model),
        )
        if hasattr(model, "modelgrid"):
            modelgrid = model.modelgrid
        else:  # assume this is a modelgrid object
            modelgrid = model
        mg = ModelGrid.from_modelgrid(modelgrid)
    grid_kwargs = mg._grid_kwargs()
    if projection is not None:
        grid_kwargs["projection"] = projection
    return cls(**grid_kwargs)


# imported from Grid class
def mask_from_modflow(self, model, model_name=None):
    """Return a mask array from a MODFLOW model, based on IBOUND/IDOMAIN.

    Parameters
    ----------
    model : str, PathLike, flopy.modflow.Modflow, flopy.mf6.mfmodel.MFModel or flopy.discretization.StructuredGrid
        Path to a MODFLOW 6 directory, mfsim.nam, binary grid file
        (with .dis.grb suffix), classic MODFLOW NAM file,
        or certain FloPy objects.
    model_name : str or None (default)
        Needed if MODFLOW 6 simulation has more than one model.

    Returns
    -------
    np.array

    """  # noqa
    mask_cache_key = (repr(model), model_name)
    if mask_cache_key in mask_cache:
        self.logger.info("using mask from cache")
        return mask_cache[mask_cache_key]
    model = get_modflow_model(model, model_name, self.logger)
    mg = model.modelgrid
    # reshape idomain into a 3-d array
    domain = mg.idomain
    if domain.ndim == 1:
        domain = domain.reshape(mg.shape)
    mask = (domain == 0).all(0)
    mask_cache[mask_cache_key] = mask
    return mask
