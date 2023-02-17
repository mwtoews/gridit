"""Grid.array_from_* and mask_from_* methods."""
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
    from gridit import Grid

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
