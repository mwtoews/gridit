"""Command-line interface utilities for the grid module."""

__all__ = []

import re
from decimal import Decimal, InvalidOperation
from importlib.util import find_spec
from textwrap import dedent

from gridit.grid import Grid

has_flopy = find_spec("flopy") is not None


def decimal_splitter(inp: str):
    """Returns Decimal or tuple of Decimal from input."""
    try:  # return scalar Decimal
        return Decimal(inp)
    except InvalidOperation:
        pass
    # return tuple of Decimal
    items = inp
    if items.startswith("(") and items.endswith(")"):
        items = items[1:-1]
    if "," in items:
        items = items.replace(",", " ")
    try:
        return tuple(Decimal(item) for item in items.split())
    except InvalidOperation:
        raise ValueError(f"cannot convert {inp!r} into Decimal or tuple of Decimal")


def add_grid_parser_arguments(parser):
    """Add parser arguments.

    Returns
    -------
    None

    """
    grid_group = parser.add_argument_group(
        "Grid definition", "Use one of the --grid-from-* methods"
    )
    grid_group.add_argument(
        "--grid-from-bbox",
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        type=Decimal,
        nargs=4,
        help="Define grid from bounding box values, requires --resolution",
    )
    grid_group.add_argument(
        "--grid-from-vector",
        metavar="FILE",
        help=dedent(
            """\
        Define grid from a vector file, e.g. catchment polygons,
        and may require ':layer' to be specified for multi-layer sources;
        requires --resolution.
        """
        ),
    )
    grid_group.add_argument(
        "--grid-from-raster",
        metavar="FILE",
        help="Use grid definition from a raster file, e.g. DTM",
    )
    if has_flopy:
        grid_group.add_argument(
            "--grid-from-modflow",
            metavar="PATH[:MODEL]",
            help=dedent(
                """\
            Use a MODFLOW grid, which must have constant row and column
            spacing. For 'classic' MODFLOW, this is a path to a NAM file.
            For MODFLOW 6, this is a path to a mfsim.nam file or directory,
            and model name specified after ':',
            e.g. 'mfsim.nam:name_of_model'. If a model name is not
            specified, the first will be selected with a warning.
            Alternatively, a binary grid (with .dis.grb) can be used.
            """
            ),
        )
    grid_group.add_argument(
        "--resolution",
        metavar="RES",
        type=Decimal,
        help="Grid resolution along X and Y directions, e.g. 100 m",
    )
    grid_group.add_argument(
        "--buffer",
        metavar="BUF",
        default=Decimal("0"),
        help=dedent(
            """\
        Add buffer to extents of data. Negative values contract bounds.
        A tuple of buffers can specify two directions (leftright, bottomtop),
        or four sides (left, bottom, right, top).
        Default 0 does not modify bounds.
        """
        ),
    )
    grid_group.add_argument(
        "--snap",
        metavar="SNAP",
        help=dedent(
            """\
        Snap mode used to evaluate grid size and offset. Default 'full' will
        snap bounds to a multiple of resolution, and 'half' will snap to
        half-resolution. Corner specifications, e.g. 'bottom-left' will
        snap the grid to align with this coordinate. Alternatively,
        a coordinate tuple (snapx, snapy) can be provided to snap the grid,
        although the grid does not necessarily include the coordinate.
        """
        ),
    )
    grid_group.add_argument(
        "--projection",
        metavar="STR",
        help="Projection or coordinate reference system for --grid-from-bbox. "
        "Use (e.g.) EPSG:2193 for New Zealand Transverse Mercator 2000.",
    )


def process_grid_options(args, logger):
    """Process grid options, returning a tuple of (grid, mask).

    Returns
    -------
    tuple
        grid, mask

    Raises
    ------
    ValueError
        If there is an issue with the argument(s); should show usage.
    OSError
        If the grid cannot be read.

    """

    def raise_error(msg: str, name: str = ""):
        msg = str(msg)
        if name:
            msg = "--" + name.replace("_", "-") + ": " + msg
        raise ValueError(msg)

    mask = None
    grid_args = {"logger": logger}
    if args.resolution is not None:
        grid_args["resolution"] = args.resolution
    if args.buffer:
        try:
            buffer = decimal_splitter(args.buffer)
        except ValueError as err:
            raise_error(err, "buffer")
        if isinstance(buffer, tuple) and len(buffer) not in {2, 4}:
            raise_error("must have two or four values", "buffer")
        grid_args["buffer"] = buffer
    if args.snap:
        if re.fullmatch(r"[a-zA-Z\-]+", args.snap):
            from gridit.classmethods import snap_modes

            snap = args.snap
            if snap not in snap_modes:
                raise_error(f"must be one of: {snap_modes}", "snap")
        else:
            try:
                snap = decimal_splitter(args.snap)
            except ValueError as err:
                raise_error(err, "snap")
            if isinstance(snap, tuple) and len(snap) != 2:
                raise_error("must have two values", "snap")
        grid_args["snap"] = snap
    if args.projection:
        grid_args["projection"] = args.projection
    from_grid_methods = ["bbox", "raster", "vector"]
    if has_flopy:
        from_grid_methods.append("modflow")
    from_grid_args = ["--grid-from-" + x for x in from_grid_methods]
    from_grid_count = sum(
        getattr(args, x.lstrip("-").replace("-", "_")) is not None
        for x in from_grid_args
    )
    if from_grid_count != 1:
        raise_error(
            "one of {} options must be specified; found {}".format(
                ", ".join(repr(x) for x in from_grid_args), from_grid_count
            )
        )
    elif args.grid_from_bbox is not None:
        if args.resolution is None:
            raise_error("requires --resolution", "grid_from_bbox")
        grid = Grid.from_bbox(*args.grid_from_bbox, **grid_args)
    elif args.grid_from_raster is not None:
        try:
            import rasterio
        except ModuleNotFoundError as err:
            raise_error(f"cannot read from raster: {err}", "grid_from_raster")
        try:
            grid = Grid.from_raster(args.grid_from_raster, **grid_args)
        except rasterio.errors.RasterioIOError as err:
            raise_error(f"cannot read from raster: {err}", "grid_from_raster")
    elif args.grid_from_vector is not None:
        try:
            import fiona
        except ModuleNotFoundError as err:
            raise_error(f"cannot read from vector: {err}", "grid_from_vector")
        if args.resolution is None:
            raise_error("requires --resolution", "grid_from_vector")
        fname = args.grid_from_vector
        layer = None
        if ":" in fname and (split := fname.rindex(":")) > 1:
            fname = args.grid_from_vector[:split]
            layer = args.grid_from_vector[(1 + split) :]
        try:
            grid = Grid.from_vector(fname, layer=layer, **grid_args)
        except fiona.errors.DriverError as err:
            raise_error(f"cannot read from vector: {err}", "grid_from_vector")
        mask = grid.mask_from_vector(fname, layer=layer)
    elif has_flopy and args.grid_from_modflow is not None:
        model = args.grid_from_modflow
        model_name = None
        if ":" in model and (split := model.rindex(":")) > 1:
            model = args.grid_from_modflow[:split]
            model_name = args.grid_from_modflow[(1 + split) :]
        projection = args.projection if args.projection else None
        grid = Grid.from_modflow(model, model_name, projection=projection)
        mask = grid.mask_from_modflow(model, model_name=model_name)
    else:
        raise NotImplementedError("whoops")
    return grid, mask


def process_nc_arg(arg):
    """Process netCDF argument with format 'file.nc:idx_name:var_name[:xidx]'.

    Returns
    -------
    tuple
        (fname, idx_name, var_name, xidx)

    Raises
    ------
    ValueError

    """
    col_count = arg.count(":")
    if col_count == 2:
        fname, idx_name, var_name = arg.split(":", 2)
        xidx = None
    elif col_count == 3:
        fname, idx_name, var_name, xidx = arg.split(":", 3)
        try:
            xidx = int(xidx)
        except ValueError:
            raise ValueError(f"xidx must be provided as an integer; found {xidx!r}")
    else:
        raise ValueError("expected format 'file.nc:idx_name:var_name[:xidx]'")
    return fname, idx_name, var_name, xidx
