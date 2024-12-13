"""Grid class and spatial tools to read array datasets."""

import math
from decimal import Decimal
from typing import Optional

__all__ = ["Grid"]
mask_cache = {}


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
    rotation : float, default 0.0
        Rotation angle around top-left corner,
        positive degrees rotate the grid anti-clockwise.
    projection : optional str, default None
        WKT coordinate reference system string.
    logger : logging.Logger, optional
        Logger to show messages.

    Attributes
    ----------
    transform : Affine
        Affine transformation object; requires affine.

    """

    from gridit.array_from import (
        array_from_array,
        array_from_raster,
        array_from_vector,
        mask_from_raster,
        mask_from_vector,
    )
    from gridit.cell import cell_geodataframe, cell_geoms, cell_geoseries
    from gridit.classmethods import from_bbox, from_raster, from_vector
    from gridit.file import write_raster, write_vector
    from gridit.modflow import from_modflow, mask_from_modflow

    def __init__(
        self,
        resolution: float,
        shape: tuple,
        top_left: tuple = (0.0, 0.0),
        rotation: float = 0.0,
        projection: Optional[str] = None,
        logger=None,
    ):
        if logger is None:
            from gridit.logger import get_logger

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
        # projection used to be the 4th positional value (now rotation), so support this
        if projection is None and (
            isinstance(rotation, str) or not isinstance(rotation, (float, int))
        ):
            self.logger.warning(
                "4th positional value is rotation, but projection was provided. "
                "A suggested change is to specify `projection=%r`",
                rotation,
            )
            projection = rotation
            rotation = 0.0
        self.rotation = float(rotation)
        self.projection = str(projection) if projection else None

    def __iter__(self):
        """Return object datasets with an iterator."""
        yield "resolution", self.resolution
        yield "shape", self.shape
        yield "top_left", self.top_left
        yield "rotation", self.rotation
        yield "projection", self.projection

    def __hash__(self):
        """Return unique hash based on content."""
        return hash(tuple(self))

    def __getstate__(self):
        """Serialize object attributes for pickle dumps."""
        state = dict(self)
        # Remove unused items
        if not state.get("rotation"):
            del state["rotation"]
        if not state.get("projection"):
            del state["projection"]
        return state

    def __setstate__(self, state):
        """Set object attributes from pickle loads."""
        self.__init__(**state)

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
        items = dict(self)
        del items["projection"]
        if items["rotation"] == 0.0:
            del items["rotation"]
        content = ", ".join(f"{k}={v}" for k, v in items.items())
        return f"<{self.__class__.__name__}: {content} />"

    @property
    def bounds(self):
        """Return bounds tuple of (xmin, ymin, xmax, ymax).

        If grid has non-zero rotation, the bounds are expanded to fit grid extents.
        """
        nrow, ncol = self.shape
        if self.rotation != 0.0:
            raise NotImplementedError
            c0x, c0y = self.top_left
            rad = math.degrees(self.rotation)
            sinrot = math.sin(rad)
            cosrot = math.cos(rad)
            lenx = ncol * self.resolution
            leny = nrow * self.resolution
            c1x = c0x + sinrot
        else:
            xmin, ymax = map(lambda x: Decimal(str(x)), self.top_left)
            res = Decimal(str(self.resolution))
            xmax = xmin + ncol * res
            ymin = ymax - nrow * res
            return tuple(map(float, (xmin, ymin, xmax, ymax)))

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
