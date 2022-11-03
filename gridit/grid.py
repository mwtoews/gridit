"""Grid class and spatial tools to read array datasets."""

__all__ = ["Grid"]

import numpy as np
from importlib.util import find_spec
from math import floor, ceil
from pathlib import Path
from warnings import warn

from .display import shorten
from .logger import get_logger
from .spatial import is_same_crs

mask_cache = {}


def float32_is_also_float64(val):
    """Return True if float32 and float64 values are the same."""
    val32 = np.float32(val)
    val64 = np.float64(val)
    if np.isnan([val32, val64]).all():
        return True
    val6432 = np.float64(val32)
    return val64 == val6432


def get_modflow_model(model, model_name=None, logger=None):
    """Return model object from str or Path."""
    import flopy

    if hasattr(model, "modelgrid"):
        return model
    elif not isinstance(model, (str, Path)):
        raise TypeError("expected str, Path or model instance")
    pth = Path(model).resolve()
    if not pth.exists():
        raise ValueError(f"cannot read path '{pth}'")
    elif (
            (pth.is_dir() and (pth / "mfsim.nam").is_file()) or
            pth.name == "mfsim.nam"):
        # MODFLOW 6
        if pth.is_dir():
            sim_ws = str(pth)
        else:
            sim_ws = str(pth.parent)
        if logger is not None:
            logger.info("reading mf6 simulation from '%s'", sim_ws)
        sim = flopy.mf6.MFSimulation.load(
            sim_ws=sim_ws, strict=False, verbosity_level=0,
            load_only=["dis", "tdis"])
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
            raise KeyError(
                f"model name {model_name} not found in {model_names}")
        model = sim.get_model(model_name)
        setattr(model, "tdis", sim.tdis)  # this is a bit of a hack
        return model
    elif pth.is_file():  # assume 'classic' MOFLOW file
        model = flopy.modflow.Modflow.load(
            pth.name, model_ws=str(pth.parent), load_only=["dis", "bas6"],
            check=False, verbose=False)
        return model
    raise ValueError(
        f"cannot determine how to read MODFLOW model '{pth}'")


class Grid:
    """Grid information class to discritize a spatial domain on a grid.

    Parameters
    ----------
    resolution : float
        Raster resolution along X and Y directions.
    shape : tuple
        2D array shape (nrow, ncol).
    top_left : tuple, default (0.0, 0.0)
        Top left corner coordinate.
    projection : str, default ""
        WKT coordinate reference system string.
    logger : logging.Logger, optional
        Logger to show messages.

    Attributes
    ----------
    transform : Affine
        Affine transformation object; requires affine.

    """

    def __init__(
            self, resolution: float, shape: tuple,
            top_left: tuple = (0.0, 0.0), projection: str = "", logger=None):

        if logger is None:
            self.logger = get_logger(self.__class__.__name__)
        else:
            self.logger = logger
        self.resolution = float(resolution)
        if len(shape) != 2:
            raise ValueError("expected shape to contain two values")
        self.shape = tuple(int(v) for v in shape)
        if len(top_left) != 2:
            raise ValueError("expected top_left to contain two values")
        self.top_left = tuple(float(v) for v in top_left)
        self.projection = str(projection)

    def __iter__(self):
        """Return object datasets with an iterator."""
        yield "resolution", self.resolution
        yield "shape", self.shape
        yield "top_left", self.top_left
        yield "projection", self.projection

    def __hash__(self):
        """Return unique hash based on content."""
        return hash(tuple(self))

    def __eq__(self, other):
        """Return True if objects are equal."""
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        try:
            return dict(self) == dict(other)
        except (AssertionError, TypeError, ValueError):
            return False

    def __repr__(self):
        """Return string representation of object."""
        content = ", ".join(
            f"{k}={v}" for k, v in self if k != "projection")
        return f"<{self.__class__.__name__}: {content} />"

    @property
    def bounds(self):
        """Return bounds tuple of (xmin, ymin, xmax, ymax)."""
        xmin, ymax = self.top_left
        nrow, ncol = self.shape
        xmax = xmin + ncol * self.resolution
        ymin = ymax - nrow * self.resolution
        return xmin, ymin, xmax, ymax

    @property
    def transform(self):
        """Return Affine transform; requires affine."""
        try:
            from affine import Affine
        except ModuleNotFoundError:
            raise ModuleNotFoundError("transform requires affine")
        if self.top_left is None:
            raise AttributeError("top_left is not set")
        c, f = self.top_left
        if self.resolution is None:
            raise AttributeError("resolution is not set")
        a = self.resolution
        e = -a
        b = d = 0.0
        return Affine(a, b, c, d, e, f)

    @classmethod
    def from_bbox(
            cls, minx: float, miny: float, maxx: float, maxy: float,
            resolution: float, buffer: float = 0.0, projection: str = "",
            logger=None):
        """Create grid information from a bounding box and resolution.

        Bounds are "snapped" to a multiple of the resolution.

        Parameters
        ----------
        fname : str
            Input file, such as a shapefile.
        minx, miny, maxx, maxy : float
            Extents of a bounding box.
        resolution : float
            A grid resolution, e.g. 250.0 for 250m x 250m
        buffer : float, default 0.0
            Add buffer to extents of bounding box.
        projection : str, default ""
            Coordinate reference system described as a string either as (e.g.)
            EPSG:2193 or a WKT string.
        logger : logging.Logger, optional
            Logger to show messages.
        """
        if logger is None:
            logger = get_logger(cls.__class__.__name__)
        logger.info("creating grid info a bounding box")
        bounds = minx, miny, maxx, maxy
        shape, top_left = _grid_from_bbox(bounds, resolution, buffer)
        return cls(resolution=resolution, shape=shape, top_left=top_left,
                   projection=projection, logger=logger)

    @classmethod
    def from_raster(
            cls, fname: str, resolution: float = None, buffer: float = 0.0,
            logger=None):
        """Fetch grid information from a raster.

        Parameters
        ----------
        fname : str
            Input file, such as a shapefile.
        resolution : float, optional
            An optional grid resolution. If not specified, the default
            resolution is from the raster. If specified, the bounds may be
            expanded and "snapped" to a multiple of the resolution.
        buffer : float, default 0.0.
            Add buffer to extents of raster.
        logger : logging.Logger, optional
            Logger to show messages.

        Raises
        ------
        ModuleNotFoundError
            If rasterio is not installed.
        """
        try:
            import rasterio
        except ModuleNotFoundError:
            raise ModuleNotFoundError("from_raster requires rasterio")
        if logger is None:
            logger = get_logger(cls.__class__.__name__)
        logger.info("reading grid info from raster: %s", fname)
        with rasterio.open(fname, "r") as ds:
            t = ds.transform
            shape = ds.shape
            projection = ds.crs.to_wkt()
        if t.e != -t.a:
            logger.error("expected e == -a, but %r != %r", t.e, t.a)
        if t.b != 0 or t.d != 0:
            logger.error("expected b == d == 0.0, but %r and %r", t.b, t.d)
        if resolution is not None or resolution != t.a or buffer > 0:
            if resolution is None:
                resolution = t.a
            ny, nx = shape
            bounds = t.c, t.f + ny * t.e, t.c + nx * t.a, t.f
            shape, top_left = _grid_from_bbox(bounds, resolution, buffer)
        else:
            resolution = t.a
            top_left = t.c, t.f
        return cls(resolution=resolution, shape=shape, top_left=top_left,
                   projection=projection, logger=logger)

    @classmethod
    def from_vector(
            cls, fname: str, resolution: float, filter: dict = None,
            buffer: float = 0.0, layer=None, logger=None):
        """Create grid information from a vector source.

        Bounds are "snapped" to a multiple of the resolution.

        Parameters
        ----------
        fname : str
            Input file, such as a shapefile.
        resolution : float
            A grid resolution, e.g. 250.0 for 250m x 250m
        filter : dict, optional
            Property filter criteria.
        buffer : float, default 0.0
            Add buffer to extents of polygon.
        layer : int or str, default None
            The integer index or name of a layer in a multi-layer dataset.
        logger : logging.Logger, optional
            Logger to show messages.

        Raises
        ------
        ModuleNotFoundError
            If fiona is not installed.
        """
        try:
            import fiona
        except ModuleNotFoundError:
            raise ModuleNotFoundError("from_vector requires fiona")
        if logger is None:
            logger = get_logger(cls.__class__.__name__)
        logger.info("creating grid info a bounding box")
        logger.info("reading grid info from vector source: %s", fname)
        if layer is None:
            layers = fiona.listlayers(fname)
            if len(layers) > 1:
                logger.warning(
                    "choosing the first of %d layers: %s", len(layers), layers)
                layer = layers[0]
        with fiona.open(fname, "r", layer=layer) as ds:
            projection = ds.crs_wkt
            if filter:
                found = False
                for f in ds:
                    r = []
                    for k in filter.keys():
                        r.append(f["properties"].get(k, "") == filter[k])
                    if len(r) > 0 and all(r):
                        found = True
                        break
                if not found:
                    raise ValueError(
                        f"could not find {filter} in {fname}")
                geom_type = f["geometry"]["type"]
                if geom_type == "Polygon":
                    ar = np.array(f["geometry"]["coordinates"])
                    assert ar.ndim == 3
                    assert ar.shape[2] in (2, 3)
                    xcs = ar[:, :, 0]
                    ycs = ar[:, :, 1]
                else:
                    raise NotImplementedError(
                        f"unexpected geometry type {geom_type}")
                bounds = xcs.min(), ycs.min(), xcs.max(), ycs.max()
            else:  # full shapefile bounds
                bounds = ds.bounds
        shape, top_left = _grid_from_bbox(bounds, resolution, buffer)
        return cls(resolution=resolution, shape=shape, top_left=top_left,
                   projection=projection, logger=logger)

    @classmethod
    def from_modflow(
            cls, model, model_name=None, projection=None, logger=None):
        """Create grid information from a MODFLOW model.

        Parameters
        ----------
        model : str, Path, flopy.modflow.Modflow, or flopy.mf6.mfmodel.MFModel
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
        """
        if find_spec("flopy") is None:
            raise ModuleNotFoundError("from_modflow requires flopy")
        if logger is None:
            logger = get_logger(cls.__class__.__name__)
        logger.info("creating grid info from a MODFLOW model: %s", type(model))
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
        top_left = (
            modelgrid.xoffset,
            modelgrid.yoffset + modelgrid.delc.sum())
        if projection is None:
            if isinstance(modelgrid.epsg, int):
                projection = f"EPSG:{modelgrid.epsg}"
            elif modelgrid.proj4 is not None:
                projection = modelgrid.proj4
            else:
                projection = ""
        # also cache mask while we are here
        if hasattr(model, "bas6"):
            domain = model.bas6.ibound.array
        else:
            domain = model.dis.idomain.array
        mask = (domain == 0).all(0)
        mask_cache[mask_cache_key] = mask
        return cls(
            resolution=delr, shape=modelgrid.top.shape, top_left=top_left,
            projection=projection, logger=logger)

    def array_from_array(self, grid, array, resampling=None):
        """Return array from a different grid and array.

        Parameters
        ----------
        grid : Grid
            Grid for array input.
        array : array_like
            Array data to regrid. If 3D, the first dimension is the band.
        resampling : rasterio.enums.Resampling, optional
            Choose one from rasterio.enums.Resampling; default (None)
            automatically selects the best method based on the relative grid
            resolutions and data type.

        Returns
        -------
        np.ma.array

        Raises
        ------
        ModuleNotFoundError
            If rasterio is not installed.
        """
        try:
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.warp import reproject
        except ModuleNotFoundError:
            raise ModuleNotFoundError("array_from_array requires rasterio")
        if not isinstance(grid, Grid):
            raise TypeError(
                f"expected grid to be a Grid; found {type(grid)!r}")
        elif not (hasattr(array, "ndim") and hasattr(array, "shape")):
            raise TypeError(
                f"expected array to be array_like; found {type(array)!r}")
        elif not (array.ndim in (2, 3) and array.shape[-2:] == grid.shape):
            raise ValueError("array has different shape than grid")

        rel_res_diff = abs(
            (grid.resolution - self.resolution) / self.resolution) * 100
        if rel_res_diff == 0.0:
            self.logger.info(
                "source and destination have the same resolution %s",
                self.resolution)
        else:
            self.logger.info(
                "source resolution %s vs destination resolution "
                "%s, which is a relative difference of %s%%",
                grid.resolution, self.resolution, rel_res_diff)
        if resampling is None:
            is_floating = np.issubdtype(array.dtype, np.floating)
            if rel_res_diff <= 10.0:
                resampling = Resampling.nearest
            elif grid.resolution > self.resolution:
                if is_floating:
                    resampling = Resampling.bilinear
                else:
                    resampling = Resampling.nearest
            elif grid.resolution < self.resolution:
                if is_floating:
                    resampling = Resampling.average
                else:
                    resampling = Resampling.mode
            else:
                raise ValueError()
        self.logger.info("using %s resampling method", resampling)
        if not rasterio.dtypes.check_dtype(array.dtype):
            dtype = rasterio.dtypes.get_minimum_dtype(array)
            self.logger.debug(
                "changing array dtype from '%s' to '%s'",
                array.dtype, dtype)
            array = array.astype(dtype)
        kwds = {}
        nodata = None
        if np.ma.isMA(array):
            if array.dtype == np.float32:
                fill_value = array.fill_value
                if not float32_is_also_float64(fill_value):
                    # TODO: any better way to find fill_value?
                    fill_value = 3.28e9
                    assert float32_is_also_float64(fill_value)
                    assert fill_value not in array
                    array = array.copy()
                    array.fill_value = fill_value
            kwds["src_nodata"] = nodata = array.fill_value
            array = array.filled()
        src_crs = self.projection
        dst_crs = grid.projection
        if not src_crs and not dst_crs:
            # TODO: is there a better catch-all projection?
            src_crs = "EPSG:3857"
            dst_crs = "EPSG:3857"
        elif not src_crs:
            src_crs = dst_crs
        elif not dst_crs:
            dst_crs = src_crs
        if array.ndim == 3:
            dst_shape = array.shape[0:1] + self.shape
        else:
            dst_shape = self.shape
        dst_array = np.ma.empty(dst_shape, array.dtype)
        dst_array.mask = False
        _ = reproject(
            array, dst_array.data,
            src_transform=grid.transform, dst_transform=self.transform,
            src_crs=src_crs, dst_crs=dst_crs,
            resampling=resampling, **kwds)
        if nodata is not None:
            dst_array.mask = dst_array.data == nodata
        return dst_array

    def array_from_raster(self, fname: str, bidx: int = 1, resampling=None):
        """Return array from a raster source aligned to grid info.

        Parameters
        ----------
        fname : str
            Source raster data to regrid.
        bidx : int, optional
            Band index, default is 1 (the first).
        resampling : rasterio.enums.Resampling, optional
            Choose one from rasterio.enums.Resampling; default (None)
            automatically selects the best method based on the relative grid
            resolutions and data type.

        Returns
        -------
        np.ma.array

        Raises
        ------
        ModuleNotFoundError
            If rasterio is not installed.
        """
        try:
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.warp import reproject
        except ModuleNotFoundError:
            raise ModuleNotFoundError("array_from_raster requires rasterio")
        self.logger.info("reading array from raster: %s, band %s", fname, bidx)
        with rasterio.open(fname, "r") as ds:
            ds_crs = ds.crs.to_wkt()
            if ds.transform == self.transform and ds.shape == self.shape:
                self.logger.info(
                    "source raster matches grid info; reading full array")
                if ds_crs != self.projection:
                    self.logger.error(
                        "source projection is different than grid")
                ar = ds.read(bidx, masked=True)
                return ar

            band = rasterio.band(ds, bidx)
            ar = np.ma.zeros(self.shape, band.dtype)
            ds_mean_res = np.mean(ds.res)
            rel_res_diff = abs(
                (ds_mean_res - self.resolution) / self.resolution) * 100
            if rel_res_diff == 0.0:
                self.logger.info(
                    "source and destination have the same mean resolution %s",
                    ds_mean_res)
            else:
                self.logger.info(
                    "source mean resolution %s vs destination mean resolution "
                    "%s, which is a relative difference of %s%%",
                    ds_mean_res, self.resolution, rel_res_diff)
            if resampling is None:
                is_floating = np.issubdtype(ar.dtype, np.floating)
                if rel_res_diff <= 10.0:
                    resampling = Resampling.nearest
                elif ds_mean_res > self.resolution:
                    if is_floating:
                        resampling = Resampling.bilinear
                    else:
                        resampling = Resampling.nearest
                elif ds_mean_res < self.resolution:
                    if is_floating:
                        resampling = Resampling.average
                    else:
                        resampling = Resampling.mode
                else:
                    raise ValueError()
            self.logger.info("using %s resampling method", resampling)
            grid_crs = self.projection
            if not grid_crs:
                grid_crs = ds_crs
                self.logger.info(
                    "assuming same projection: %s", shorten(grid_crs, 60))
            _ = reproject(
                band, ar.data,
                src_transform=ds.transform, dst_transform=self.transform,
                src_crs=ds.crs, dst_crs=grid_crs,
                dst_nodata=ds.nodata,
                resampling=resampling)
            if ds.nodata is not None:
                if np.isnan(ds.nodata):
                    ar.mask = np.isnan(ar.data)
                    ar.fill_value = np.nan
                else:
                    ar.mask = ar.data == ds.nodata
        return ar

    def mask_from_raster(self, fname: str, bidx: int = 1):
        """Return a mask array from a raster source aligned to grid info.

        Parameters
        ----------
        fname : str
            Source raster data to extract mask.
        bidx : int, optional
            Band index, default is 1 (the first).

        Returns
        -------
        np.array
        """
        return self.array_from_raster(fname, bidx).mask

    def array_from_vector(
            self, fname: str, attribute: str, *, fill=0, refine: int = 1,
            layer=None, all_touched=False):
        """Return array from vector source data aligned to grid info.

        The datatype is inferred from the attribute values.

        Parameters
        ----------
        fname : str
            Polygon vector source data to create array from.
        attribute : str or None
            Name of attribute to rasterize. If None, a boolean result where
            polygon features are located is returned.
        fill : float or int, default 0
            Fill value, only used where polygon does not cover unmasked grid.
        refine : int, default 1
            If greater than 1, refine each dimension by a factor as a
            pre-processing step to approximate more details from the vector
            file to the gridded result.
        layer : int or str, default None
            The integer index or name of a layer in a multi-layer dataset.
        all_touched : bool, defalt False
            If True, all grid cells touched by geometries will be burned in.
            Default False will only burn in cells whose center is within the
            polygon.

        Returns
        -------
        np.ma.array

        Raises
        ------
        ModuleNotFoundError
            If fiona and/or rasterio is not installed.
        """
        try:
            import fiona
            from affine import Affine
            from rasterio import features
            from rasterio.dtypes import get_minimum_dtype
            from rasterio.enums import Resampling
            from rasterio.warp import reproject
            from shapely.geometry import shape
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "array_from_vector requires fiona and rasterio")
        grid_transform = self.transform
        grid_crs = self.projection
        use_refine = refine > 1
        if refine < 1:
            raise ValueError("refine must be >= 1")
        self.logger.info("reading array from vector datasource: %s", fname)
        if layer is None:
            layers = fiona.listlayers(fname)
            if len(layers) > 1:
                self.logger.warning(
                    "choosing the first of %d layers: %s", len(layers), layers)
                layer = layers[0]
        with fiona.open(fname, "r", layer=layer) as ds:
            if ds.schema["geometry"] != "Polygon":
                self.logger.warning(
                    "expected 'Polygon', found %r", ds.schema["geometry"])
            ds_crs = ds.crs_wkt
            do_transform = False
            if not grid_crs:
                grid_crs = ds_crs
                self.logger.info(
                    "assuming same projection: %s", shorten(grid_crs, 60))
            elif is_same_crs(grid_crs, ds_crs):
                grid_crs = ds_crs
                self.logger.info(
                    "same projection: %s", shorten(grid_crs, 60))
            else:
                do_transform = True
                from fiona.transform import transform_geom
                from shapely.geometry import box, mapping
                self.logger.info(
                    "geometries will be transformed from %s to %s",
                    shorten(ds_crs, 60), shorten(grid_crs, 60))
            geom_vals = []
            grid_bounds = self.bounds
            if attribute is None:
                val = 1
                nodata = 0
                dtype = "uint8"
                resampling = Resampling.mode
            else:
                vals = []
                attributes = list(ds.schema["properties"].keys())
                if attribute not in attributes:
                    raise KeyError(
                        f"could not find {attribute} in {attributes}")
                vdtype = ds.schema["properties"][attribute]
            if do_transform:
                grid_box = box(*grid_bounds)
                # TODO: does this make sense?
                buf = self.resolution * np.average(self.shape) * 0.15
                grid_box_t = shape(transform_geom(
                    grid_crs, ds_crs,
                    mapping(grid_box.buffer(buf)))).buffer(buf)
                kwargs = {"bbox": grid_box_t.bounds}
                self.logger.info(
                    "transforming features in bbox %s", grid_box_t.bounds)
                for _, feat in ds.items(**kwargs):
                    if attribute is not None:
                        val = feat["properties"][attribute]
                        if val is None:
                            continue
                    geom = transform_geom(ds_crs, grid_crs, feat["geometry"])
                    geom_obj = shape(geom)
                    if geom_obj.is_empty or not grid_box.intersects(geom_obj):
                        continue
                    if attribute is not None:
                        vals.append(val)
                    geom_vals.append((geom, val))
            else:
                for _, feat in ds.items(bbox=grid_bounds):
                    if attribute is not None:
                        val = feat["properties"][attribute]
                        if val is None:
                            continue
                    geom = feat["geometry"]
                    if shape(geom).is_empty:
                        continue
                    if attribute is not None:
                        vals.append(val)
                    geom_vals.append((geom, val))

        if len(geom_vals) == 0:
            self.logger.warning("no valid geometry objects found")
            nodata = 0
            if attribute is not None and vdtype.startswith("float"):
                nodata = 0.0
            dtype = get_minimum_dtype(nodata)
            ar = np.ma.zeros(self.shape, dtype=dtype)
            ar.mask = True
            ar.fill_value = fill
            return ar

        if attribute is not None:
            vals = np.array(vals)
            if vdtype.startswith("float"):
                nodata = vals.max() * 10.0
                resampling = Resampling.average
            elif vdtype.startswith("int"):
                if vals.min() > 0:
                    nodata = 0
                else:
                    nodata = vals.max() + 1
                resampling = Resampling.mode
            else:
                raise ValueError(
                    f"attribute {attribute} is neither float or int")
            dtype = get_minimum_dtype(np.append(vals, nodata))
            if dtype == "float32":
                dtype = "float64"
        dtype_conv = np.dtype(dtype).type
        nodata = dtype_conv(nodata)
        fill = dtype_conv(fill)
        ar = np.ma.empty(self.shape, dtype=dtype)
        ar.fill(nodata)
        if use_refine:
            fine_transform = grid_transform * Affine.scale(1. / refine)
            fine_shape = tuple(n * refine for n in self.shape)
            self.logger.info("rasterizing features to %s fine array", dtype)
            fine_ar = features.rasterize(
                geom_vals, fine_shape, transform=fine_transform,
                fill=nodata, dtype=dtype, all_touched=all_touched)
            self.logger.info(
                "reprojecting from fine to coarse array using "
                "%s resampling method and all_touched=%s",
                resampling, all_touched)
            _ = reproject(
                fine_ar, ar.data,
                src_transform=fine_transform, dst_transform=grid_transform,
                src_crs=ds_crs, dst_crs=ds_crs,
                src_nodata=nodata, dst_nodata=nodata,
                resampling=resampling)
        else:
            self.logger.info(
                "rasterizing features to %s array with all_touched=%s",
                dtype, all_touched)
            _ = features.rasterize(
                geom_vals, self.shape, out=ar.data, transform=grid_transform,
                dtype=dtype, all_touched=all_touched)
        is_nodata = ar.data == nodata
        if is_nodata.any():
            ar.data[is_nodata] = fill
            ar.mask |= is_nodata
        ar.fill_value = fill
        return ar

    def mask_from_vector(self, fname, layer=None):
        """Return a mask array from a vector source aligned to grid info.

        Parameters
        ----------
        fname : str
            Polygon vector source data to evaluate mask.
        layer : int or str, default None
            The integer index or name of a layer in a multi-layer dataset.

        Returns
        -------
        np.array
        """
        ar = self.array_from_vector(fname, None, layer=layer)
        return ~ar.data.astype(bool)

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

    def write_raster(self, array, fname: str, driver=None):
        """Write array to a raster file format.

        Parameters
        ----------
        array : array_like
            Array to write; must have 2-dimensions that match shape.
        fname : str
            Output file to write.
        driver : str or None (default)
            Raster driver. Default None will determine driver from fname.

        Raises
        ------
        ModuleNotFoundError
            If rasterio is not installed.
        """
        try:
            import rasterio
        except ModuleNotFoundError:
            raise ModuleNotFoundError("array_from_vector requires rasterio")
        if array.ndim != 2:
            raise ValueError("array must have two-dimensions")
        elif array.shape != self.shape:
            raise ValueError("array must have same shape " + str(self.shape))
        self.logger.info("writing raster file: %s", fname)
        if driver is None:
            from rasterio.drivers import driver_from_extension
            driver = driver_from_extension(fname)
        kwds = {
            "driver": driver,
            "width": self.shape[1],
            "height": self.shape[0],
            "count": 1,
            "dtype": array.dtype,
            "crs": self.projection,
            "transform": self.transform,
        }
        if np.ma.isMA(array):
            if array.dtype == np.float32:
                fill_value = array.fill_value
                if not float32_is_also_float64(fill_value):
                    # TODO: any better way to find fill_value?
                    fill_value = 3.28e9
                    assert float32_is_also_float64(fill_value)
                    assert fill_value not in array
                    array = array.copy()
                    array.fill_value = fill_value
            kwds["nodata"] = array.fill_value
            array = array.filled()
        with rasterio.open(fname, "w", **kwds) as ds:
            ds.write(array, 1)


def _grid_from_bbox(bounds, resolution, buffer=0.0):
    minx, miny, maxx, maxy = bounds
    if not (minx < maxx):
        raise ValueError("'minx' must be less than 'maxx'")
    elif not (miny < maxy):
        raise ValueError("'miny' must be less than 'maxy'")
    elif resolution <= 0:
        raise ValueError("'resolution' must be greater than zero")
    elif buffer < 0:
        raise ValueError("'buffer' must be zero or greater")
    if buffer > 0.0:
        minx -= buffer
        miny -= buffer
        maxx += buffer
        maxy += buffer
    dx = dy = resolution
    if buffer > 0.0:
        minx = dx * round(minx / dx)
        miny = dy * round(miny / dy)
        maxx = dx * round(maxx / dx)
        maxy = dy * round(maxy / dy)
    else:
        minx = dx * floor(minx / dx)
        miny = dy * floor(miny / dy)
        maxx = dx * ceil(maxx / dx)
        maxy = dy * ceil(maxy / dy)
    lenx = maxx - minx
    leny = maxy - miny
    assert lenx % dx == 0.0
    assert leny % dy == 0.0
    nx = int(lenx / dx)
    ny = int(leny / dy)
    shape = ny, nx
    top_left = (minx, maxy)
    return shape, top_left
