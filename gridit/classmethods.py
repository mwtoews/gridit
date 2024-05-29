"""Grid from_* classmethods."""

from math import ceil, floor
from typing import Optional

from gridit.logger import get_logger


def get_shape_top_left(bounds, resolution, buffer=0.0):
    minx, miny, maxx, maxy = bounds
    if not (minx <= maxx):
        raise ValueError("'minx' must be less than 'maxx'")
    elif not (miny <= maxy):
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


@classmethod
def from_bbox(
    cls,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    resolution: float,
    buffer: float = 0.0,
    projection: Optional[str] = None,
    logger=None,
):
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
    projection : optional str, default None
        Coordinate reference system described as a string either as (e.g.)
        EPSG:2193 or a WKT string.
    logger : logging.Logger, optional
        Logger to show messages.

    Examples
    --------
    From user-supplied bounds:

    >>> from gridit import Grid
    >>> grid1 = Grid.from_bbox(1620000, 5324000, 1685000, 5360000,
    ...                       200, projection="EPSG:2193")
    >>> grid1
    <Grid: resolution=200.0, shape=(180, 325), top_left=(1620000.0, 5360000.0) />

    From shapely geometry:
    >>> from shapely import wkt
    >>> domain = wkt.loads("POLYGON ((1685000 5359000, 1665000 5324000, "
    ...                              "1620000 5360000, 1685000 5359000))")
    >>> grid2 = Grid.from_bbox(*domain.bounds, 200, projection="EPSG:2193")
    >>> assert grid1 == grid2

    """
    if logger is None:
        logger = get_logger(cls.__name__)
    logger.info("creating from a bounding box")
    bounds = minx, miny, maxx, maxy
    shape, top_left = get_shape_top_left(bounds, resolution, buffer)
    return cls(
        resolution=resolution,
        shape=shape,
        top_left=top_left,
        projection=projection,
        logger=logger,
    )


@classmethod
def from_raster(
    cls, fname: str, resolution: float = None, buffer: float = 0.0, logger=None
):
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
        logger = get_logger(cls.__name__)
    logger.info("creating from raster: %s", fname)
    projection = None
    with rasterio.open(fname, "r") as ds:
        t = ds.transform
        shape = ds.shape
        if ds.crs:
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
        shape, top_left = get_shape_top_left(bounds, resolution, buffer)
    else:
        resolution = t.a
        top_left = t.c, t.f
    return cls(
        resolution=resolution,
        shape=shape,
        top_left=top_left,
        projection=projection,
        logger=logger,
    )


@classmethod
def from_vector(
    cls,
    fname: str,
    resolution: float,
    filter: dict = None,
    buffer: float = 0.0,
    layer=None,
    logger=None,
):
    """Create grid information from a vector source.

    Bounds are "snapped" to a multiple of the resolution.

    Parameters
    ----------
    fname : str
        Input file, such as a shapefile.
    resolution : float
        A grid resolution, e.g. 250.0 for 250m x 250m
    filter : dict, str, optional
        Property filter criteria. For example ``{"id": 4}`` to select one
        feature with attribute "id" value 4. Or ``{"id": [4, 7, 19]}`` to
        select features with several values. A SQL WHERE statement can also be
        used if Fiona 1.9 or later is installed.
    buffer : float, default 0.0
        Add buffer to extents of vector data.
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
        logger = get_logger(cls.__name__)
    logger.info("reading from a vector source: %s", fname)
    if layer is None:
        layers = fiona.listlayers(fname)
        if len(layers) > 1:
            logger.warning("choosing the first of %d layers: %s", len(layers), layers)
            layer = layers[0]
    with fiona.open(fname, "r", layer=layer) as ds:
        projection = ds.crs_wkt
        if filter:
            from gridit.file import fiona_filter_collection

            flt = fiona_filter_collection(ds, filter)
            if len(flt) == 0:
                logger.error("no features filtered with %s", filter)
            bounds = flt.bounds
            flt.close()
        else:  # full shapefile bounds
            bounds = ds.bounds
    shape, top_left = get_shape_top_left(bounds, resolution, buffer)
    return cls(
        resolution=resolution,
        shape=shape,
        top_left=top_left,
        projection=projection,
        logger=logger,
    )
