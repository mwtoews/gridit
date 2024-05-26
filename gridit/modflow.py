"""Modflow methods."""

from importlib.util import find_spec
from pathlib import Path
from warnings import catch_warnings, filterwarnings, warn

from gridit.grid import mask_cache
from gridit.logger import get_logger


def get_modflow_model(model, model_name=None, logger=None):
    """Return model object from str or Path."""
    import flopy

    if hasattr(model, "xoffset"):
        from types import SimpleNamespace

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
            model = flopy.modflow.Modflow.load(
                pth.name,
                model_ws=str(pth.parent),
                load_only=["dis", "bas6"],
                check=False,
                verbose=False,
                forgive=True,
            )
        return model
    raise ValueError(f"cannot determine how to read MODFLOW model '{pth}'")


@classmethod
def from_modflow(cls, model, model_name=None, projection=None, logger=None):
    """Create grid information from a MODFLOW model.

    Parameters
    ----------
    model : str, Path, flopy.modflow.Modflow, flopy.mf6.mfmodel.MFModel or flopy.discretization.grid
        MODFLOW model specified either as a FloPy object, or path to
        a MODFLOW file.
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
        model if isinstance(model, str) else type(model),
    )
    mask_cache_key = (repr(model), model_name)
    model = get_modflow_model(model, model_name, logger)
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
