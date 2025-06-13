"""Grid.array_from_* and mask_from_* methods."""

import inspect

import numpy as np

from gridit.display import shorten
from gridit.file import float32_is_also_float64
from gridit.spatial import is_same_crs


def array_from_array(self, grid, array, resampling=None):
    """Return array from a different grid and array.

    Parameters
    ----------
    grid : Grid
        Grid for array input.
    array : array_like
        Array data to regrid. If 3D, the first dimension is the band.
    resampling : str, int or rasterio.enums.Resampling, optional
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
        from rasterio.crs import CRS
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ModuleNotFoundError:
        raise ModuleNotFoundError("array_from_array requires rasterio")
    from gridit import Grid

    if not isinstance(grid, Grid):
        raise TypeError(f"expected grid to be a Grid; found {type(grid)!r}")
    if not (hasattr(array, "ndim") and hasattr(array, "shape")):
        raise TypeError(f"expected array to be array_like; found {type(array)!r}")
    if not (array.ndim in (2, 3) and array.shape[-2:] == grid.shape):
        raise ValueError("array has different shape than grid")

    rel_res_diff = abs((grid.resolution - self.resolution) / self.resolution) * 100
    if rel_res_diff == 0.0:
        self.logger.info(
            "source and destination have the same resolution %s", self.resolution
        )
    else:
        self.logger.info(
            "source resolution %s vs destination resolution "
            "%s, which is a relative difference of %s%%",
            grid.resolution,
            self.resolution,
            rel_res_diff,
        )
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
    elif isinstance(resampling, str):
        resampling = rasterio.enums.Resampling[resampling]
    self.logger.info("using %s resampling method", resampling.name)
    if not rasterio.dtypes.check_dtype(array.dtype):
        dtype = rasterio.dtypes.get_minimum_dtype(array)
        self.logger.debug("changing array dtype from '%s' to '%s'", array.dtype, dtype)
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
        src_crs = dst_crs = CRS.from_epsg(3857)
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
        array,
        dst_array.data,
        src_transform=grid.transform,
        dst_transform=self.transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=resampling,
        **kwds,
    )
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
    resampling : str, int or rasterio.enums.Resampling, optional
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
        from rasterio.crs import CRS
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ModuleNotFoundError:
        raise ModuleNotFoundError("array_from_raster requires rasterio")
    self.logger.info("reading array from raster: %s, band %s", fname, bidx)
    with rasterio.open(fname, "r") as ds:
        ds_nodata = ds.nodata
        unset_nodata = ds_nodata is None
        ds_crs = None if ds.crs is None else ds.crs.to_wkt()
        if ds.transform == self.transform and ds.shape == self.shape:
            self.logger.info("source raster matches grid info; reading full array")
            if ds_crs != self.projection:
                self.logger.error("source projection is different than grid")
            ar = ds.read(bidx, masked=True)
            if unset_nodata:
                ar_is_nan = np.isnan(ar.data)
                if np.isnan(ar.data).any():
                    self.logger.warning("implicitly setting nodata as nan")
                    ar.mask |= ar_is_nan
                    ar.fill_value = np.nan
            return ar

        grid_crs = self.projection
        if not grid_crs:
            if ds_crs:
                grid_crs = ds_crs
                self.logger.info("assuming same projection: %s", shorten(grid_crs, 60))
            else:
                # TODO: is there a better catch-all projection?
                grid_crs = ds_crs = CRS.from_epsg(3857)

        band = rasterio.band(ds, bidx)
        if ds_nodata is None:
            val = np.zeros(1, band.dtype)
            ds_nodata = np.ma.default_fill_value(val)
            self.logger.debug("implicitly setting nodata as %r", ds_nodata)
        else:
            self.logger.debug("filling array with nodata %r", ds_nodata)
        ar = np.ma.array(np.full(self.shape, ds_nodata, band.dtype))
        ds_mean_res = np.mean(ds.res)
        rel_res_diff = abs((ds_mean_res - self.resolution) / self.resolution) * 100
        if rel_res_diff == 0.0:
            self.logger.info(
                "source and destination have the same mean resolution %s", ds_mean_res
            )
        else:
            self.logger.info(
                "source mean resolution %s vs destination mean resolution "
                "%s, which is a relative difference of %s%%",
                ds_mean_res,
                self.resolution,
                rel_res_diff,
            )
        if resampling is None:
            is_floating = np.issubdtype(ar.dtype, np.floating)
            if rel_res_diff <= 10.0:
                resampling = Resampling.nearest
            elif ds_mean_res > self.resolution:
                resampling = Resampling.bilinear if is_floating else Resampling.nearest
            elif ds_mean_res < self.resolution:
                resampling = Resampling.average if is_floating else Resampling.mode
            else:
                raise ValueError()
        elif isinstance(resampling, str):
            resampling = rasterio.enums.Resampling[resampling]
        self.logger.info("using %s resampling method", resampling.name)

        reproject_kwds = dict(
            src_transform=ds.transform,
            dst_transform=self.transform,
            src_crs=ds_crs,
            dst_crs=grid_crs,
            dst_nodata=ds_nodata,
            resampling=resampling,
        )
        _ = reproject(band, ar.data, **reproject_kwds)
        ar_is_nan = np.isnan(ar.data)
        if unset_nodata and ar_is_nan.any():
            self.logger.debug("2nd reproject, implicitly setting nodata as %r", np.nan)
            ar.fill(np.nan)
            reproject_kwds.update(dict(dst_nodata=np.nan, src_nodata=np.nan))
            _ = reproject(band, ar.data, **reproject_kwds)
            ar_is_nan = np.isnan(ar.data)
        if (np.isnan(ds_nodata) or unset_nodata) and ar_is_nan.any():
            ar.mask |= ar_is_nan
            ar.fill_value = np.nan
        else:
            new_mask = ar.data == ds_nodata
            if new_mask.any():
                ar.mask |= new_mask
            ar.fill_value = ds_nodata
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
    ar = self.array_from_raster(fname, bidx)
    # return ar.mask
    if ar.mask.shape:
        return ar.mask
    return np.full(ar.shape, ar.mask)


def array_from_geom(
    self,
    geom,
    *,
    refine=None,
    all_touched=False,
):
    """Return array from a Shapely geometry.

    Returned values are float values between 0 and 1, depending on overlap.

    Parameters
    ----------
    geom : shapely geometry, list or array-like of geometries
        One or more input geometries.
    refine : int, optional
        Controls level of pre-processing used to approximate details
        from polygon vector sources. Ignored for points.
        If one, use default (coarse) rasterizing at grid resolution.
        If greater than 1, refine each dimension by a factor.
        Default will determine an appropriate refine value based on
        geometry type.
    all_touched : bool, default False
        If True, all grid cells touched by polygon or line geometries will be
        updated. Default False will only update cells whose center is within
        the polygon or line is on the render path. Ignored for points.

    Returns
    -------
    np.ma.array

    Raises
    ------
    ModuleNotFoundError
        If rasterio is not installed.

    """
    vd = GridVectorData.from_geom(self, geom)
    return vd.rasterize_array(refine=refine, all_touched=all_touched)


def array_from_geopandas(
    self,
    gpd,
    *,
    attribute=None,
    fill=0,
    refine=None,
    all_touched=False,
):
    """Return array from a geopandas GeoSeries, GeoDataFrame or GeometryArray.

    Parameters
    ----------
    gpd : geopandas GeoSeries, GeoDataFrame, or GeometryArray object
    attribute : str, optional
        Column name of GeoDataFrame object to rasterize.
    fill : float or int, default 0
        Fill value, only used where geometries do not cover unmasked grid.
    refine : int, optional
        Controls level of pre-processing used to approximate details
        from polygon vector sources. Ignored for points.
        If one, use default (coarse) rasterizing at grid resolution.
        If greater than 1, refine each dimension by a factor.
        Default will determine an appropriate refine value based on
        geometry type.
    all_touched : bool, default False
        If True, all grid cells touched by polygon or line geometries will be
        updated. Default False will only update cells whose center is within
        the polygon or line is on the render path. Ignored for points.

    Returns
    -------
    np.ma.array

    Raises
    ------
    ModuleNotFoundError
        If rasterio is not installed.

    """
    vd = GridVectorData.from_geopandas(self, gpd, attribute)
    return vd.rasterize_array(fill=fill, refine=refine, all_touched=all_touched)


def array_from_vector(
    self,
    fname: str,
    *,
    layer=None,
    attribute=None,
    fill=0,
    refine=None,
    all_touched=False,
):
    """Return array from vector source data aligned to grid info.

    The datatype is inferred from the attribute values.

    Parameters
    ----------
    fname : str or PathLike
        Vector source data to create array from.
    layer : int or str, default None
        The integer index or name of a layer in a multi-layer dataset.
    attribute : str, optional
        Name of attribute to rasterize.
    fill : float or int, default 0
        Fill value, only used where geometries do not cover unmasked grid.
    refine : int, optional
        Controls level of pre-processing used to approximate details
        from polygon vector sources. Ignored for points.
        If one, use default (coarse) rasterizing at grid resolution.
        If greater than 1, refine each dimension by a factor.
        Default will determine an appropriate refine value.
    all_touched : bool, default False
        If True, all grid cells touched by polygon or line geometries will be
        updated. Default False will only update cells whose center is within
        the polygon or line is on the render path. Ignored for points.

    Returns
    -------
    np.ma.array

    Raises
    ------
    ModuleNotFoundError
        If fiona and/or rasterio is not installed.

    """
    self.logger.info("reading array from vector datasource: %s", fname)
    vd = GridVectorData.from_vector_file(self, fname, layer, attribute)
    return vd.rasterize_array(fill=fill, refine=refine, all_touched=all_touched)


class GridVectorData:
    def __init__(self, grid):
        """Create a GridVectorData instance from a Grid object."""
        self.grid = grid
        self.logger = grid.logger
        self.shape = grid.shape
        self.resolution = grid.resolution
        self.transform = grid.transform
        self.bounds = grid.bounds

    @classmethod
    def from_geom(cls, grid, geom):
        """Create grid vector data from a shapely-like geometry."""
        if hasattr(geom, "tolist"):
            # convert array-like to list
            geom = geom.tolist()
        if isinstance(geom, list):
            # check each item is a geometry
            geom_type_s = set()
            for item in geom:
                geom_type = getattr(item, "geom_type", None)
                geom_type_s.add(geom_type)
                if not isinstance(geom_type, str):
                    stack1 = inspect.stack()[1]
                    raise TypeError(
                        f"'geom' for {stack1.function} must be a list of "
                        "shapely-like geometry objects"
                    )
            if len(geom_type_s) == 1:
                geom_type = geom_type_s.pop()
            else:
                geom_type = "GeometryCollection"
        else:
            # single geometry
            geom_type = getattr(geom, "geom_type", None)
            if not isinstance(geom_type, str):
                stack1 = inspect.stack()[1]
                msg = f"'geom' for {stack1.function} must be shapely-like geometry"
                raise TypeError(msg)
            geom = [geom]
        obj = cls(grid)
        obj.attribute = None
        obj.geoms = geom
        obj.geom_type = geom_type
        return obj

    @classmethod
    def from_geopandas(cls, grid, gpd, attribute=None):
        """Create grid vector data from a geopandas object."""
        obj = cls(grid)
        if hasattr(gpd, "geometry"):
            obj.geoms = gpd.geometry
        elif hasattr(gpd, "to_numpy"):  # GeometryArray
            obj.geoms = gpd.to_numpy()
        else:
            stack1 = inspect.stack()[1]
            msg = f"'gpd' for {stack1.function} does not seem to be a geopandas object"
            raise TypeError(msg)

        if hasattr(gpd, "crs") and hasattr(gpd, "to_crs"):
            # Check if it needs to be re-projected to match grid
            gpd_crs = gpd.crs
            grid_crs = grid.projection
            if not grid_crs:
                obj.logger.info("geometries not transformed: grid has no projection")
            elif not gpd_crs:
                obj.logger.info("geometries not transformed: vector has no projection")
            elif is_same_crs(grid_crs, gpd_crs):
                obj.logger.debug(
                    "geometries not transformed: same projection: %s",
                    shorten(grid_crs, 60),
                )
            else:
                obj.logger.info(
                    "geometries will be transformed from %s to %s",
                    shorten(gpd_crs, 60),
                    shorten(grid_crs, 60),
                )
                gpd = gpd.to_crs(grid_crs)
        obj.attribute = attribute
        if attribute is not None:
            try:
                obj.vals = gpd[attribute]
            except (IndexError, KeyError):
                raise IndexError(f"'gdb' does not have attribute '{attribute}'")
            obj.vdtype = str(gpd[attribute].dtype)
        if len(geom_type := np.unique(gpd.geom_type)) == 1:
            obj.geom_type = geom_type[0]
        else:
            obj.geom_type = "GeometryCollection"
        return obj

    @classmethod
    def from_vector_file(cls, grid, fname, layer, attribute=None):
        """Read vector data source."""
        try:
            import fiona
        except ModuleNotFoundError:
            stack1 = inspect.stack()[1]
            raise ModuleNotFoundError(f"{stack1.function} requires fiona")
        from shapely.geometry import shape

        obj = cls(grid)
        obj.attribute = attribute
        if attribute is not None:
            obj.vals = []
        obj.geoms = []
        layers = fiona.listlayers(fname)
        if layer is None:
            if len(layers) > 1:
                obj.logger.warning(
                    "choosing the first of %d layers: %s", len(layers), layers
                )
                layer = layers[0]
        elif layer not in layers:
            # show error message before fiona/GDAL raise error with "Null layer"
            if len(layers) == 1:
                obj.logger.error(
                    "layer %r does not match vector source layer %r", layer, layers[0]
                )
            else:
                obj.logger.error(
                    "layer %r not found in source layers: %r", layer, layers
                )
        with fiona.open(fname, "r", layer=layer) as ds:
            obj.geom_type = ds.schema["geometry"]
            obj.logger.info("processing %s geometry data", obj.geom_type)
            ds_crs = ds.crs_wkt
            grid_crs = grid.projection
            do_transform = False
            if not grid_crs:
                obj.logger.info("geometries not transformed: grid has no projection")
            elif not ds_crs:
                obj.logger.info("geometries not transformed: vector has no projection")
            elif is_same_crs(grid_crs, ds_crs):
                obj.logger.debug(
                    "geometries not transformed: same projection: %s",
                    shorten(grid_crs, 60),
                )
            else:
                do_transform = True
                obj.logger.info(
                    "geometries will be transformed from %s to %s",
                    shorten(ds_crs, 60),
                    shorten(grid_crs, 60),
                )
            if attribute is not None:
                attributes = list(ds.schema["properties"].keys())
                if attribute not in attributes:
                    raise KeyError(f"could not find {attribute!r} in {attributes}")
                obj.vdtype = ds.schema["properties"][attribute]

            if do_transform:
                from fiona.transform import transform_geom
                from shapely.geometry import box, mapping

                grid_box = box(*obj.bounds)
                # expand box slightly, because it might rotate with transform
                buf = obj.resolution * np.average(obj.shape) * 0.15
                grid_box_t = shape(
                    transform_geom(grid_crs, ds_crs, mapping(grid_box.buffer(buf)))
                ).buffer(buf)
                kwargs = {"bbox": grid_box_t.bounds}
                obj.logger.info("transforming features in bbox %s", grid_box_t.bounds)
                for _, feat in ds.items(**kwargs):
                    if attribute is not None:
                        val = feat["properties"][attribute]
                        if val is None:
                            continue
                    geom = shape(transform_geom(ds_crs, grid_crs, feat["geometry"]))
                    if geom.is_empty or not grid_box.intersects(geom):
                        continue
                    obj.geoms.append(geom)
                    if attribute is not None:
                        obj.vals.append(val)
            else:
                for _, feat in ds.items(bbox=obj.bounds):
                    if attribute is not None:
                        val = feat["properties"][attribute]
                        if val is None:
                            continue
                    geom = shape(feat["geometry"])
                    if geom.is_empty:
                        continue
                    obj.geoms.append(geom)
                    if attribute is not None:
                        obj.vals.append(val)
        return obj

    def __len__(self):
        return len(self.geoms)

    def empty_array(self, fill=0):
        # nodata = 0
        if self.attribute is None:
            dtype = "uint8"
        elif self.vdtype.startswith("float"):
            dtype = "float64"
        elif self.vdtype.startswith("int"):
            dtype = "uint8"
        else:
            raise ValueError(f"attribute {self.attribute} is neither float or int")
        ar = np.ma.zeros(self.shape, dtype=dtype)
        ar.mask = True
        ar.fill_value = fill
        return ar

    def rasterize_array(self, refine=None, fill=0, all_touched=False):
        try:
            from affine import Affine
            from rasterio import features
            from rasterio.dtypes import get_minimum_dtype
            from rasterio.enums import Resampling
        except ModuleNotFoundError:
            stack1 = inspect.stack()[1]
            raise ModuleNotFoundError(f"{stack1.function} requires rasterio")

        if len(self) == 0:
            self.logger.warning("no valid geometry objects found")
            return self.empty_array(fill=fill)

        if refine is None:
            # Auto-evaluate refine and dtype based on geometry type
            if self.attribute is None:
                # If "attribute" is not provided:
                #  - Point: refine == 1
                #  - LineString: refine == 1
                #  - Polygon: refine > 1
                if "Polygon" in self.geom_type:
                    refine = 5
                else:
                    refine = 1
            else:
                # If "attribute" is provided:
                #  - Point: refine > 1
                #  - LineString: refine > 1
                #  - Polygon: refine > 1
                refine = 5
            msg = "selecting default"
        else:
            msg = "using"
        self.logger.info("%s refine=%d for %s", msg, refine, self.geom_type)

        if self.attribute is None:
            # Rasterize 0 and 1 values
            line_or_point = "Point" in self.geom_type or "Line" in self.geom_type
            rasterize_fill = 0
            geom_vals = [(g, 1) for g in self.geoms]
            if refine == 1 or line_or_point:
                dtype = "uint8"
                resampling = Resampling.mode
            else:
                dtype = "float32"
                resampling = Resampling.average
            if line_or_point:
                nodata = 0
            else:
                nodata = 2  # neither 0 or 1
        else:
            geom_vals = zip(self.geoms, self.vals)
            vals = np.array(self.vals)
            if self.vdtype.startswith("float"):
                nodata = vals.max() * 10.0
                resampling = Resampling.average
            elif self.vdtype.startswith("int"):
                nodata = 0 if vals.min() > 0 else vals.max() + 1
                resampling = Resampling.mode
            else:
                raise ValueError(f"attribute {self.attribute} is neither float or int")
            dtype = get_minimum_dtype(np.append(vals, nodata))
            if dtype == "float32":
                dtype = "float64"
            rasterize_fill = nodata

        ar = np.ma.empty(self.shape, dtype=dtype)
        ar.fill(rasterize_fill)

        if refine > 1:
            from rasterio.crs import CRS
            from rasterio.warp import reproject

            fine_transform = self.transform * Affine.scale(1.0 / refine)
            fine_shape = tuple(n * refine for n in self.shape)
            self.logger.info(
                "rasterizing features to %s fine array with fill=%s and all_touched=%s",
                dtype,
                rasterize_fill,
                all_touched,
            )
            fine_ar = features.rasterize(
                geom_vals,
                fine_shape,
                transform=fine_transform,
                fill=rasterize_fill,
                dtype=dtype,
                all_touched=all_touched,
            )
            dtype_conv = np.dtype(dtype).type
            nodata = dtype_conv(nodata)
            # TODO: is there a better catch-all projection?
            ds_crs = CRS.from_epsg(3857)
            self.logger.info(
                "reprojecting from fine to coarse array using %s resampling method "
                "and nodata=%s",
                resampling.name,
                nodata,
            )
            _ = reproject(
                fine_ar,
                ar.data,
                src_transform=fine_transform,
                dst_transform=self.transform,
                src_crs=ds_crs,
                dst_crs=ds_crs,
                src_nodata=nodata,
                dst_nodata=nodata,
                resampling=resampling,
            )
        else:
            self.logger.info(
                "rasterizing features to %s array with fill=%s and all_touched=%s",
                dtype,
                rasterize_fill,
                all_touched,
            )
            _ = features.rasterize(
                geom_vals,
                self.shape,
                out=ar.data,
                transform=self.transform,
                fill=rasterize_fill,
                dtype=dtype,
                all_touched=all_touched,
            )
        is_nodata = ar.data == rasterize_fill
        if is_nodata.any():
            ar.data[is_nodata] = fill
            ar.mask |= is_nodata
        ar.fill_value = fill
        return ar


def mask_from_vector(self, fname, *, layer=None):
    """Return a mask array from a vector source aligned to grid info.

    Parameters
    ----------
    fname : str
        Vector source data to evaluate mask.
    layer : int or str, default None
        The integer index or name of a layer in a multi-layer dataset.

    Returns
    -------
    np.array

    """
    ar = self.array_from_vector(fname, layer=layer, refine=1, all_touched=True)
    return ~ar.data.astype(bool)
