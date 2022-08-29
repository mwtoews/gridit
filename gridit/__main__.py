"""Command-line interface for grid package."""
import argparse
import sys
from textwrap import dedent

try:
    import rasterio
except ModuleNotFoundError:
    rasterio = None

try:
    import netCDF4
except ModuleNotFoundError:
    netCDF4 = None

try:
    import fiona
except ModuleNotFoundError:
    fiona = None


from . import cli, GridPolyConv
from .display import print_array
from .logger import get_logger


def cli_main():
    """Command-line interface for the gridit package.

    To use:
    $ gridit -h
    """
    parser = argparse.ArgumentParser(
        prog=__package__, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
        Examples:

          Grid from vector:
          $ gridit --grid-from-vector tests/data/Mana_polygons.shp --resolution 10

          Array from vector:
          $ gridit --grid-from-vector tests/data/Mana_polygons.shp --resolution 10 --array-from-vector tests/data:Mana_polygons --array-from-vector-attribute=K_m_d

          Array from raster:
          $ gridit --grid-from-vector tests/data/Mana_polygons.shp --resolution 10 --array-from-raster tests/data/Mana.tif

          Array from netCDF:
          $ gridit --grid-from-vector tests/data/waitaku2.shp --resolution 250 --array-from-vector tests/data/waitaku2.shp --array-from-vector-attribute rid --array-from-netcdf tests/data/waitaku2.nc:rid:__xarray_dataarray_variable__ --time-stats "quantile(0.75),max"
        """))  # noqa

    parser.add_argument(
        "--logger", metavar="LEVEL", default="INFO",
        help="Logger level, default INFO")

    cli.add_grid_parser_arguments(parser)

    if rasterio:
        array_from_raster_group = parser.add_argument_group(
            "Array from raster")
        array_from_raster_group.add_argument(
            "--array-from-raster", metavar="FILE",
            help="Source raster file"
        )
        array_from_raster_group.add_argument(
            "--array-from-raster-bidx", metavar="BIDX", type=int, default=1,
            help="Source raster band index, default 1 (first)"
        )
        array_from_raster_group.add_argument(
            "--array-from-raster-resampling", metavar="SMP",
            help="Raster resampling method, default None to "
                 "automatically choose. Use one of: " +
                 ", ".join(rasterio.enums.Resampling.__members__.keys())
        )
    else:
        parser.add_argument_group(
            "Array from raster", "rasterio not installed")

    if netCDF4 and fiona:
        array_from_netcdf_group = parser.add_argument_group(
            "Array from catchment netCDF")
        array_from_netcdf_group.add_argument(
            "--array-from-netcdf", metavar="F:I:V",
            help=dedent("""\
                Source netCDF of catchment values supplied in the format:
                'file.nc:idx_name:var_name' where 'file.nc' is a path to a
                netCDF file, 'idx_name' is the variable name with the polygon
                index, and 'var_name' is the name of the variable for values.
                If the variable has a time dimension, it is reduced by
                evaluating time statistics, with default 'mean'."""))
        array_from_netcdf_group.add_argument(
            "--time-stats", metavar="TYPE", default="mean",
            help=dedent("""\
                Compute time-statistics along time dimension.
                Default "mean" evaluates the mean values. Other types may
                include "min", "median", "max", "quantile(N)" where N is a
                real value between 0.0 and 1.0. An optional time-window can
                specify a range of months or specify hydrlogic years, which
                modifies "min", "median" and "max" calculations to find
                years with "lowest", "middle" or "highest" total values.
                E.g. "Jul-Jun:min" will find the NZ water year with lowest
                values."""))
    else:
        parser.add_argument_group(
            "Array from catchment netCDF",
            "netCDF4 and/or fiona not installed")

    if fiona:
        array_from_vector_group = parser.add_argument_group(
            "Array from vector")
        array_from_vector_group.add_argument(
            "--array-from-vector", metavar="FILE",
            help="Source vector file. For multilayer datasources, use the "
            "format 'datasource:layer'"
        )
        array_from_vector_group.add_argument(
            "--array-from-vector-attribute", metavar="NAME",
            help="Name of attribute to rasterize. If None, a boolean result "
                 "where polygon features are located is returned."
        )
        array_from_vector_group.add_argument(
            "--array-from-vector-fill", metavar="FILL", default=0,
            help="Fill value, only used where polygon does not cover unmasked "
                 "grid. Default fill value is 0."
        )
        array_from_vector_group.add_argument(
            "--array-from-vector-refine",
            metavar="INT", type=int, default=5,
            help="If greater than 1, refine each dimension by a factor as a "
                 "pre-processing step to approximate more details from the "
                 "vector file to the gridded result. Default 5."
        )
        array_from_vector_group.add_argument(
            "--array-from-vector-max-levels",
            metavar="INT", type=int, default=5,
            help="If refine is greater than 1, set a maximum number "
                 "of refine levels. Default 5."
        )
    else:
        parser.add_argument_group(
            "Array from vector", "fiona not installed")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    logger = get_logger(__package__, args.logger)

    def error(msg, name="", show_usage=False, exit=1):
        if show_usage:
            parser.print_usage(sys.stderr)
        m = __package__ + ": error: "
        if name:
            m += "--" + name.replace("_", "-") + ": "
        print(m + msg, file=sys.stderr)
        if exit:
            sys.exit(exit)

    # Process grid options

    try:
        grid, mask = cli.process_grid_options(args, logger)
    except ValueError as err:
        error(str(err), show_usage=True)
    except (ModuleNotFoundError, OSError) as err:
        error(str(err), show_usage=False)

    logger.info("%s", grid)
    logger.info("has mask: %s", mask is not None)

    # Process array from * options

    array = None
    if getattr(args, "array_from_raster", None):
        try:
            array = grid.array_from_raster(
                fname=args.array_from_raster,
                bidx=args.array_from_raster_bidx,
                resampling=args.array_from_raster_resampling,
            )
        except fiona.errors.DriverError as err:
            error(str(err), "array_from_raster", exit=False)
        print_array(array, logger=logger)

    if getattr(args, "array_from_netcdf", None):
        name_nc = "array_from_netcdf"
        nc_arg = getattr(args, name_nc)
        try:
            nc_fname, idx_name, var_name = cli.process_nc_arg(nc_arg)
        except ValueError as err:
            error(str(err), name_nc, show_usage=True)

        vector_fname = getattr(args, "array_from_vector", None)
        attr_name = "array_from_vector_attribute"
        attr = getattr(args, attr_name, None)
        if vector_fname is None or attr is None:
            error("missing --array-from-vector and/or "
                  "--array-from-vector-attribute, which are required "
                  "to provide spatial distributions of catchment polygons, "
                  "and the common index attribute name", name_nc)
        if ":" in vector_fname:
            split = vector_fname.index(":")
            layer = vector_fname[(1 + split):]
            vector_fname = vector_fname[:split]
        else:
            layer = None

        gpc = GridPolyConv.from_grid_vector(
            grid, vector_fname, attr, layer=layer,
            refine=args.array_from_vector_refine,
            max_levels=args.array_from_vector_max_levels)

        fill = args.array_from_vector_fill
        ar_d = gpc.array_from_netcdf(
            nc_fname, idx_name, var_name,
            time_stats=args.time_stats, fill=fill, enforce1d=True)
        for key, array in ar_d.items():
            logger.info("time stats: %s", key)
            if array.ndim == 3:
                idxs = [0]
                if array.shape[0] > 1:
                    idxs.append(array.shape[0] - 1)
                for idx in idxs:
                    logger.info("array index: %s", idx)
                    print_array(array[idx], logger=logger)
            else:
                print_array(array, logger=logger)
        logger.info("done")
        return

    if getattr(args, "array_from_vector", None):
        if ":" in args.array_from_vector:
            split = args.array_from_vector.index(":")
            fname = args.array_from_vector[:split]
            layer = args.array_from_vector[(1 + split):]
        else:
            fname = args.array_from_vector
            layer = None
        try:
            array = grid.array_from_vector(
                fname=fname, layer=layer,
                attribute=args.array_from_vector_attribute,
                fill=args.array_from_vector_fill,
                refine=args.array_from_vector_refine,
            )
        except rasterio.errors.RasterioIOError as err:
            error(str(err), "array_from_vector", exit=False)
        print_array(array, logger=logger)

    logger.info("done")


if __name__ == "__main__":
    cli_main()
