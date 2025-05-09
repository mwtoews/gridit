"""Grid from_* classmethods."""

from decimal import Decimal
from itertools import product
from math import ceil, floor

from gridit.logger import get_logger

_lr = ["left", "right"]
_tb = ["top", "bottom"]
snap_modes = ["full", "half"] + list("-".join(two) for two in product(_tb, _lr))


def get_shape_top_left(
    bounds: tuple,
    resolution: float | Decimal,
    buffer: float | Decimal | tuple = Decimal("0"),
    snap: str | tuple = "full",
):
    """Get shape and top-left coordinate to define a grid from bounds.

    Parameters
    ----------
    bounds : tuple of float or Decimal
        Bounding box, ordered (minx, miny, maxx, maxy).
    resolution : float or Decimal
        Grid resolution for x- and y-directions.
    buffer : float, Decimal, or tuple; default Decimal('0')
        Add buffer to extents of bounding box. Negative values contract bounds.
        A tuple of buffers can specify two directions (leftright, bottomtop),
        or four sides (left, bottom, right, top).
    snap : {full, half, top-left, top-right, bottom-left, bottom-right} or tuple
        Snap mode used to evaluate grid size and offset. Default 'full' will
        snap bounds to a multiple of resolution, and 'half' will snap to
        half-resolution. Corner specifications, e.g. 'bottom-left' will
        snap the grid to align with this coordinate. Alternatively,
        a coordinate tuple (snapx, snapy) can be provided to snap the grid,
        although the grid does not necessarily include the coordinate.

    Returns
    -------
    shape: tuple of int
        Dimensions of grid (ny, nx).
    top_left: tuple of float
        Snapped top-left corner of grid (minx, maxy).

    """
    minx, miny, maxx, maxy = (
        x if isinstance(x, Decimal) else Decimal(str(x)) for x in bounds
    )
    if not isinstance(resolution, Decimal):
        resolution = Decimal(str(resolution))
    if isinstance(buffer, tuple):
        if len(buffer) not in {2, 4}:
            raise ValueError("'buffer' tuple must have 2 or 4 items")
        buffer_items = [
            buf if isinstance(buf, Decimal) else Decimal(str(buf)) for buf in buffer
        ]
        buffer = any(buffer_items)
        if len(buffer_items) == 2:
            buffer_items *= 2
    else:
        if not isinstance(buffer, Decimal):
            buffer = Decimal(str(buffer))
        buffer_items = [buffer] * 4
    if not (minx <= maxx):
        raise ValueError("'minx' must be less than 'maxx'")
    if not (miny <= maxy):
        raise ValueError("'miny' must be less than 'maxy'")
    if resolution <= 0:
        raise ValueError("'resolution' must be greater than zero")
    if buffer:
        minx -= buffer_items[0]
        miny -= buffer_items[1]
        maxx += buffer_items[2]
        maxy += buffer_items[3]
        if min(buffer_items) < 0.0:
            # correct too much contraction with average of bounds
            if minx > maxx:
                midx = False
                if isinstance(snap, str):
                    if "left" in snap:
                        maxx = minx
                    elif "right" in snap:
                        minx = maxx
                    else:
                        midx = True
                else:
                    midx = True
                if midx:
                    minx = maxx = (minx + maxx) / 2
            if miny > maxy:
                midy = False
                if isinstance(snap, str):
                    if "top" in snap:
                        miny = maxy
                    elif "bottom" in snap:
                        maxy = miny
                    else:
                        midy = True
                else:
                    midy = True
                if midy:
                    miny = maxy = (miny + maxy) / 2
    dx = dy = resolution
    if snap == "full":
        snapx = snapy = Decimal("0")
    elif snap == "half":
        snapx = dx / Decimal("2")
        snapy = dy / Decimal("2")
    elif isinstance(snap, tuple):
        if len(snap) != 2:
            raise TypeError("'snap' tuple must have 2 items: (snapx, snapy)")
        if all(isinstance(x, Decimal) for x in snap):
            snapx, snapy = snap
        else:
            snapx, snapy = map(lambda x: Decimal(str(x)), snap)
    elif snap in snap_modes:
        if "top" in snap:
            snapy = maxy
        else:
            assert "bottom" in snap, snap
            leny = maxy - miny
            ny = ceil(leny / dy) or 1
            snapy = miny + ny * dy
            if leny == 0.0:
                miny += dy
                maxy += dy
        if "left" in snap:
            snapx = minx
        else:
            assert "right" in snap, snap
            lenx = maxx - minx
            nx = ceil(lenx / dx) or 1
            snapx = maxx - nx * dx
            if lenx == 0.0:
                minx -= dx
                maxx -= dx
    else:
        raise ValueError(f"'snap' must be one of {snap_modes} or tuple (snapx, snapy)")
    snapx %= dx
    snapy %= dy
    minx = dx * floor((minx - snapx) / dx) + snapx
    maxx = dx * ceil((maxx - snapx) / dx) + snapx
    miny = dy * floor((miny - snapy) / dy) + snapy
    maxy = dy * ceil((maxy - snapy) / dy) + snapy
    nx = int((maxx - minx) / dx) or 1
    ny = int((maxy - miny) / dy) or 1
    shape = ny, nx
    top_left = (float(minx), float(maxy))
    # Uncomment to see WKT for bounds
    # maxx = minx + dx * nx
    # miny = maxy - dy * ny
    # print(
    #     f"POLYGON (({minx} {maxy}, {minx} {miny}, "
    #     f"{maxx} {miny}, {maxx} {maxy}, {minx} {maxy}))"
    # )
    return shape, top_left


@classmethod
def from_bbox(
    cls,
    minx: float | Decimal,
    miny: float | Decimal,
    maxx: float | Decimal,
    maxy: float | Decimal,
    resolution: float | Decimal,
    *,
    buffer: float | Decimal = Decimal("0"),
    snap: str | tuple = "full",
    projection: str | None = None,
    logger=None,
):
    """Create grid information from a bounding box and resolution.

    Bounds are "snapped" to a multiple of the resolution.

    Parameters
    ----------
    fname : str
        Input file, such as a shapefile.
    minx, miny, maxx, maxy : float or Decimal
        Extents of a bounding box.
    resolution : float or Decimal
        A grid resolution, e.g. 250.0 for 250m x 250m
    buffer : float, Decimal, or tuple; default Decimal('0')
        Add buffer to extents of bounding box. Negative values contract bounds.
        A tuple of buffers can specify two directions (leftright, bottomtop),
        or four sides (left, bottom, right, top).
    snap : {full, half, top-left, top-right, bottom-left, bottom-right} or tuple
        Snap mode used to evaluate grid size and offset. Default 'full' will
        snap bounds to a multiple of resolution, and 'half' will snap to
        half-resolution. Corner specifications, e.g. 'bottom-left' will
        snap the grid to align with this coordinate. Alternatively,
        a coordinate tuple (snapx, snapy) can be provided to snap the grid,
        although the grid does not necessarily include the coordinate.
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
    shape, top_left = get_shape_top_left(bounds, resolution, buffer, snap)
    return cls(
        resolution=resolution,
        shape=shape,
        top_left=top_left,
        projection=projection,
        logger=logger,
    )


@classmethod
def from_raster(
    cls,
    fname: str,
    resolution: float | Decimal | None = None,
    *,
    buffer: float | Decimal | tuple = Decimal("0"),
    snap: str | tuple = "full",
    logger=None,
):
    """Fetch grid information from a raster.

    Parameters
    ----------
    fname : str
        Input file, such as a shapefile.
    resolution : float or Decimal, optional
        An optional grid resolution. If not specified, the grid will have the
        same resolution, bounds and shape as the raster. If specified, the
        bounds may change relative to 'snap' option, which has default 'full'.
    buffer : float, Decimal, or tuple; default Decimal('0')
        Add buffer to extents of raster. Negative values contract bounds.
        A tuple of buffers can specify two directions (leftright, bottomtop),
        or four sides (left, bottom, right, top).
    snap : {full, half, top-left, top-right, bottom-left, bottom-right} or tuple
        Snap mode used to evaluate grid size and offset. Default 'full' will
        snap bounds to a multiple of resolution, and 'half' will snap to
        half-resolution. Corner specifications, e.g. 'bottom-left' will
        snap the grid to align with this coordinate. Alternatively,
        a coordinate tuple (snapx, snapy) can be provided to snap the grid,
        although the grid does not necessarily include the coordinate.

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
    if resolution is not None or buffer:
        ny, nx = shape
        a, _, c, _, e, f = map(lambda x: Decimal(str(x)), list(t)[:6])
        bounds = c, f + ny * e, c + nx * a, f
        if resolution is None:
            resolution = a
        # Shape can change here
        shape, top_left = get_shape_top_left(bounds, resolution, buffer, snap)
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
    resolution: float | Decimal,
    *,
    filter: dict | str | None = None,
    buffer: float | Decimal | tuple = Decimal("0"),
    snap: str | tuple = "full",
    layer=None,
    logger=None,
):
    """Create grid information from a vector source.

    Bounds are "snapped" to a multiple of the resolution.

    Parameters
    ----------
    fname : str
        Input file, such as a shapefile.
    resolution : float or Decimal
        A grid resolution, e.g. 250.0 for 250m x 250m
    filter : dict, str, optional
        Property filter criteria. For example ``{"id": 4}`` to select one
        feature with attribute "id" value 4. Or ``{"id": [4, 7, 19]}`` to
        select features with several values. A SQL WHERE statement can also be
        used if Fiona 1.9 or later is installed.
    buffer : float, Decimal, or tuple; default Decimal('0')
        Add buffer to extents of vector data. Negative values contract bounds.
        A tuple of buffers can specify two directions (leftright, bottomtop),
        or four sides (left, bottom, right, top).
    snap : {full, half, top-left, top-right, bottom-left, bottom-right} or tuple
        Snap mode used to evaluate grid size and offset. Default 'full' will
        snap bounds to a multiple of resolution, and 'half' will snap to
        half-resolution. Corner specifications, e.g. 'bottom-left' will
        snap the grid to align with this coordinate. Alternatively,
        a coordinate tuple (snapx, snapy) can be provided to snap the grid,
        although the grid does not necessarily include the coordinate.
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
    layers = fiona.listlayers(fname)
    if layer is None:
        if len(layers) > 1:
            logger.warning("choosing the first of %d layers: %s", len(layers), layers)
            layer = layers[0]
    elif layer not in layers:
        # show error message before fiona/GDAL raise error with "Null layer"
        if len(layers) == 1:
            logger.error(
                "layer %r does not match vector source layer %r", layer, layers[0]
            )
        else:
            logger.error("layer %r nout found in source layers: %r", layer, layers)
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
    shape, top_left = get_shape_top_left(bounds, resolution, buffer, snap)
    return cls(
        resolution=resolution,
        shape=shape,
        top_left=top_left,
        projection=projection,
        logger=logger,
    )
