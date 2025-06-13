"""Spatial utilities module."""

import re
from typing import Any

import numpy as np

from gridit.logger import get_logger

__all__ = [
    "flat_grid_intersect",
    "is_same_crs",
]


def is_same_crs(crs1: Any, crs2: Any) -> bool:
    """Determine if two CRS as pyproj objects or WKT strings are nearly the same.

    First try to compare simple EPSG codes. Otherwise, use
    SequenceMatcher from difflib to determine if ratio is greater than 2/3.
    """
    # use pyproj's methods with CRS objects and/or WKT strings
    if hasattr(crs1, "equals"):
        return crs1.equals(crs2)
    if hasattr(crs2, "equals"):
        return crs2.equals(crs1)

    if crs1.upper() == crs2.upper():
        return True

    def epsg_code(wkt):
        if match := re.fullmatch(r"EPSG:(\d+)", wkt, re.IGNORECASE):
            return match.groups()[0]
        return None

    code1 = epsg_code(crs1)
    code2 = epsg_code(crs2)
    if code1 is not None and code2 is not None:
        return code1 == code2

    from difflib import SequenceMatcher

    ratio = SequenceMatcher(None, crs2, crs1).ratio()
    return ratio > 2 / 3.0


def flat_grid_intersect(this, other, method="vector"):
    """Returns an intersection generator.

    Parameters
    ----------
    this, other : grid-like
        Input grids to intersect.
    method : {"vector", "raster"}
        Default "vector" is more accurate, but may take longer to process.
        A "raster" method is less precise, but faster.

    Yields
    ------
    (this_idx_fraction, other_idx_fraction)

    """
    logger = this.logger or other.logger
    if not logger:
        logger = get_logger("flat_grid_intersect")
    logger.warning("ignoring different CRS from grids")
    this_crs = this.projection
    other_crs = other.projection
    if not this_crs and not other_crs:
        # TODO: is there a better catch-all projection?
        this_crs = other_crs = "EPSG:3857"
    elif not this_crs:
        this_crs = other_crs
    elif not other_crs:
        other_crs = this_crs

    if method == "vector":
        import shapely
        from shapely.strtree import STRtree

        logger.debug(
            "using shapely-%s to perform vector intersect", shapely.__version__
        )
        if this_crs != other_crs:
            logger.warning("ignoring different CRS from grids")
        this_geoms = this.cell_geoms()
        other_geoms = other.cell_geoms()
        classic_shapely = shapely.__version__.startswith("1.")
        same_resolution = this.resolution == other.resolution
        if classic_shapely:
            logger.warning("recommend upgrading shapely>=2 for better performance")
        other_tree = STRtree(other_geoms)
        this_area = this.resolution**2
        other_area = other.resolution**2

        for this_idx, this_geom in enumerate(this_geoms):
            if classic_shapely:
                other_bbox = other_tree.query_items(this_geom)
            else:
                other_bbox = other_tree.query(this_geom)
            if len(other_bbox) == 0:
                continue  # outside both grid bounds
            if same_resolution:
                # find exact geometry matches
                if classic_shapely:
                    other_pbox = other_tree.query_items(this_geom.centroid)
                else:
                    other_pbox = other_tree.query(this_geom.centroid)
                if len(other_pbox) == 1:
                    other_idx = other_pbox[0]
                    if this_geom.equals(other_geoms[other_idx]):
                        yield (this_idx, 1.0), (other_idx, 1.0)
                        continue
            # evaluate intersection areas
            for other_idx in other_bbox:
                other_geom = other_geoms[other_idx]
                ina = other_geom.intersection(this_geom).area
                if ina > 0.0:
                    yield (this_idx, ina / this_area), (other_idx, ina / other_area)

    elif method == "raster":
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject

        logger.debug(
            "using rasterio-%s to perform raster intersect", rasterio.__version__
        )

        if this.resolution <= other.resolution:
            logger.info("resampling other grid to this")
            src = other
            dst = this
            swap = False
        else:
            assert this.resolution > other.resolution
            logger.info("resampling this grid to other")
            src = this
            dst = other
            swap = True

        dst_frac = (dst.resolution / src.resolution) ** 2
        src_idx_size = src.shape[0] * src.shape[1]
        idx_dtype = rasterio.dtypes.get_minimum_dtype(src_idx_size + 1)
        src_idx1_ar = np.arange(src_idx_size, dtype=idx_dtype).reshape(src.shape) + 1
        dst_idx1_ar = np.empty(dst.shape, dtype=idx_dtype)
        src_crs = this_crs
        dst_crs = other_crs
        nodata = 0
        if swap:
            src_crs, dst_crs = dst_crs, src_crs

        _ = reproject(
            src_idx1_ar,
            dst_idx1_ar,
            src_transform=src.transform,
            dst_transform=dst.transform,
            src_crs=src_crs,
            dst_crs=dst_crs,
            src_nodata=nodata,
            dst_nodata=nodata,
            resampling=Resampling.nearest,
        )

        for src_idx, dst_idx1 in enumerate(dst_idx1_ar.flatten()):
            if dst_idx1 == nodata:
                continue  # outside grid bounds
            src_item = src_idx, 1.0
            dst_item = dst_idx1 - 1, dst_frac
            if swap:
                yield dst_item, src_item
            else:
                yield src_item, dst_item
    else:
        raise NotImplementedError(f"{method=} not implemented")
