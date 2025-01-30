"""File methods."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np

__all__ = [
    "fiona_filter_collection",
    "fiona_property_type",
    "float32_is_also_float64",
    "write_raster",
    "write_vector",
]


def float32_is_also_float64(val):
    """Return True if float32 and float64 values are the same."""
    val_str = str(val)
    val32 = np.float32(val_str)
    val64 = np.float64(val_str)
    if np.isnan([val32, val64]).all():
        return True
    val6432 = np.float64(val32)
    return val64 == val6432


def write_raster(grid, array, fname, driver=None, **kwargs):
    """Write array to a raster file format.

    Parameters
    ----------
    grid : Grid
    array : array_like
        Array to write; must have 2-dimensions that match shape.
    fname : str or PathLike
        Output file to write.
    driver : str, optional
        Raster driver. Default None will determine driver from fname.
    **kwargs : dict, optional
        Other driver-specific parameters that will be interpreted by the
        GDAL library as raster creation options.

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
    if array.shape != grid.shape:
        raise ValueError("array must have same shape " + str(grid.shape))
    grid.logger.info("writing raster file: %s", fname)
    if driver is None:
        from rasterio.drivers import driver_from_extension

        driver = driver_from_extension(fname)
    # GDAL treats these as case-insensitive
    driver_lname = driver.lower()
    if kwargs:
        kwargs_lnames = [key.lower() for key in kwargs]
    else:
        kwargs = {}
        kwargs_lnames = []
    kwargs.update(
        {
            "driver": driver,
            "width": grid.shape[1],
            "height": grid.shape[0],
            "count": 1,
            "crs": grid.projection,
            "transform": grid.transform,
        }
    )
    if np.issubdtype(array.dtype, np.bool_):
        grid.logger.debug("changing dtype from %s to uint8", array.dtype)
        array = array.astype(np.uint8, copy=True)
        if driver_lname == "gtiff" and "nbits" not in kwargs_lnames:
            # write a smaller file
            kwargs["NBITS"] = "2"
    kwargs["dtype"] = array.dtype
    if np.ma.isMA(array) and "nodata" not in kwargs.keys():
        if array.mask.all():
            nodata = 0
        elif np.issubdtype(array.dtype, np.integer):
            nodata = 0 if array.min() > 0 else array.max() + 1
        else:
            nodata = array.fill_value
            if array.dtype == np.float32 and not float32_is_also_float64(nodata):
                # TODO: any better way to find fill_value?
                nodata = 3.28e9
                assert nodata not in array
                array = array.copy()
                array.nodata = nodata
        kwargs["nodata"] = nodata = array.dtype.type(nodata)
        array = array.filled(nodata)
    with rasterio.open(fname, "w", **kwargs) as ds:
        ds.write(array, 1)


def fiona_property_type(ar):
    """Returns Fiona property type from array."""
    if np.issubdtype(ar.dtype, np.floating):
        ar_str = np.array([np.format_float_positional(x) for x in ar.ravel()])
        ar_len = np.char.str_len(ar_str)
        ar_len[np.char.endswith(ar_str, ".")] -= 1
        ar_len[np.char.endswith(ar_str, ".0")] -= 2
        precision = ar_len.max()
        frac = np.modf(ar)[0]
        if (frac == 0.0).all():
            return f"float:{precision}"
        if (has_dec := np.char.count(ar_str, ".") > 0).any():
            ndc = ar_len[has_dec] - np.char.index(ar_str[has_dec], ".")
            scale = ndc.max() - 1
            return f"float:{precision}.{scale}"
        return "float"
    if np.issubdtype(ar.dtype, np.integer):
        scale = max(len(str(ar.min())), len(str(ar.max())))
        return f"int:{scale}"
    if np.issubdtype(ar.dtype, np.bool_):
        return "int:1"
    if np.issubdtype(ar.dtype, np.str_) or np.issubdtype(ar.dtype, np.bytes_):
        scale = np.char.str_len(ar).max()
        return f"str:{scale}"
    return "str"


def write_vector(grid, array, fname, attribute, layer=None, driver=None, **kwargs):
    """Write array to a vector file format.

    Parameters
    ----------
    grid : Grid
    array : array_like
        Array to write; must have either 2-dimensions that match shape, or
        3-dimensions where the first is the same length as attribute.
    fname : path-like or str
        Output file to write.
    attribute : str or list
        Attribute name. If array is 2D, this is either a str or list with
        length 1. If array is 3D, this is a list with the same length as
        the first dimension.
    layer : str or None (default)
        Vector layer, if implemented by vector driver.
    driver : str or None (default)
        Vector driver. Default None will try to determine driver from
        fname.
    **kwargs : dict, optional
        Other driver-specific parameters that will be interpreted by the
        OGR library as layer creation options.

    Raises
    ------
    ModuleNotFoundError
        If fiona and/or shapely is not installed.

    """
    try:
        import fiona
    except ModuleNotFoundError:
        raise ModuleNotFoundError("write_vector requires fiona")
    try:
        from shapely.geometry import mapping
    except ModuleNotFoundError:
        raise ModuleNotFoundError("write_vector requires shapely")
    if array.ndim == 2:
        array = array.reshape((-1,) + array.shape)
        if isinstance(attribute, str):
            attribute = [attribute]
    elif array.ndim != 3:
        raise ValueError("array must have 2 or 3 dimensions")
    if not isinstance(attribute, list) or len(attribute) != array.shape[0]:
        if array.shape[0] == 1:
            raise ValueError("attribute must be a str or a 1 item str list")
        raise ValueError(f"attribute must list of str with length {array.shape[0]}")
    if array.shape[-2:] != grid.shape:
        raise ValueError(f"last two dimensions of array shape must be {grid.shape}")
    grid.logger.info("writing vector file: %s with layer: %s", fname, layer)
    if driver is None:
        try:
            from fiona.drvsupport import driver_from_extension
        except (AttributeError, ImportError):

            def driver_from_extension(path):
                try:
                    return {
                        "csv": "CSV",
                        "gpkg": "GPKG",
                        "shp": "ESRI Shapefile",
                    }[Path(path).suffix.lstrip(".").lower()]
                except KeyError:
                    raise ValueError("Unable to detect driver. Please specify driver.")

        driver = driver_from_extension(fname)
        grid.logger.debug("driver from extension: %s", driver)

    geoms = grid.cell_geoms()
    idxs = np.arange(geoms.size)
    if np.ma.isMA(array) and array.mask.any():
        sel2d = (~array.mask).any(0)
        sel1d = sel2d.ravel()
        geoms = geoms[sel1d]
        idxs = idxs[sel1d]
        vals = array.data[:, sel2d]
    else:
        vals = array.reshape(array.shape[0:1] + geoms.shape)
    rows, cols = np.unravel_index(idxs, grid.shape)
    # build records
    recs = []
    for items in zip(geoms, idxs, rows, cols, *vals):
        geom, idx, row, col = items[:4]
        rec = {
            "geometry": mapping(geom),
            "properties": {
                "idx": idx.item(),
                "row": row.item(),
                "col": col.item(),
            },
        }
        for attr_name, attr_val in zip(attribute, items[4:]):
            rec["properties"][attr_name] = attr_val.item()
        recs.append(rec)
    # build schema
    schema = {
        "geometry": "Polygon",
        "properties": {
            "idx": fiona_property_type(idxs),
            "row": fiona_property_type(rows),
            "col": fiona_property_type(cols),
        },
    }
    for attr_name, attr_val in zip(attribute, vals):
        schema["properties"][attr_name] = fiona_property_type(attr_val)
    grid.logger.debug("schema: %s", schema)
    if not kwargs:
        kwargs = {}
    kwargs.update(
        {
            "driver": driver,
            "schema": schema,
            "crs": grid.projection,
        }
    )
    if layer:
        kwargs["layer"] = layer
    with fiona.open(fname, "w", **kwargs) as ds:
        ds.writerecords(recs)
    grid.logger.info("wrote %d features", idxs.size)


def fiona_filter_collection(ds, filter):
    """Returns Fiona collection with applied filter.

    Parameters
    ----------
    ds : fiona.Collection
        Input data source
    filter : dict, str
        Property filter criteria. For example ``{"id": 4}`` to select one
        feature with attribute "id" value 4. Or ``{"id": [4, 7, 19]}`` to
        select features with several values. A SQL WHERE statement can also be
        used if Fiona 1.9 or later is installed.

    Returns
    -------
    fiona.Collection

    Raises
    ------
    ModuleNotFoundError
        If fiona is not installed.

    """
    try:
        import fiona
    except ModuleNotFoundError:
        raise ModuleNotFoundError("fiona_filter_collection requires fiona")
    if not isinstance(ds, fiona.Collection):
        raise ValueError(f"ds must be fiona.Collection; found {type(ds)}")
    if ds.closed:
        raise ValueError("ds is closed")
    flt = fiona.io.MemoryFile().open(driver=ds.driver, schema=ds.schema, crs=ds.crs)
    if isinstance(filter, dict):
        # check that keys are found in datasource
        filter_keys = list(filter.keys())
        ds_attrs = list(ds.schema["properties"].keys())
        if not set(filter_keys).issubset(ds_attrs):
            not_found = set(filter_keys).difference(ds_attrs)
            raise KeyError(
                f"cannot find filter keys: {not_found}; "
                f"choose from data source attributes: {ds_attrs}"
            )
        found = 0
        for feat in ds:
            for attr, filt_val in filter.items():
                feat_val = feat["properties"][attr]
                if isinstance(filt_val, Iterable) and not isinstance(filt_val, str):
                    for fv in filt_val:
                        if feat_val == fv:
                            found += 1
                            flt.write(feat)
                else:
                    if feat_val == filt_val:
                        found += 1
                        flt.write(feat)
    elif isinstance(filter, str):
        if tuple(map(int, fiona.__version__.split(".", maxsplit=2)[0:2])) < (1, 9):
            raise ValueError(
                "Fiona 1.9 or later required to use filter str as SQL WHERE"
            )
        for feat in ds.filter(where=filter):
            flt.write(feat)
    else:
        raise ValueError("filter must be a dict or str")
    return flt
