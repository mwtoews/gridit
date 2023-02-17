"""Grid class and spatial tools to read array datasets."""

__all__ = ["Grid"]

import numpy as np
from pathlib import Path

from gridit.display import shorten
from gridit.logger import get_logger
from gridit.spatial import is_same_crs

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
    projection : str, default ""
        WKT coordinate reference system string.
    logger : logging.Logger, optional
        Logger to show messages.

    Attributes
    ----------
    transform : Affine
        Affine transformation object; requires affine.

    """
    from gridit.classmethods import from_bbox, from_raster, from_vector
    from gridit.array_from import (
        array_from_array, array_from_raster, array_from_vector,
        mask_from_raster, mask_from_vector
    )
    from gridit.cell import cell_geoms, cell_geoseries, cell_geodataframe
    from gridit.modflow import from_modflow, mask_from_modflow

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
        from gridit.file import write_raster

        write_raster(self, array=array, fname=fname, driver=driver)

    def write_vector(
            self, array, fname, attribute, layer=None, driver=None, **kwargs):
        """Write array to a vector file format.

        Parameters
        ----------
        array : array_like
            Array to write; must have 2-dimensions that match shape.
        fname : path-like or str
            Output file to write.
        attribute : str or list
            Attribute name. If array is 2D, this is either a str or list with
            length 1. If array is 3D, this is a list with the same lenght as
            the first dimension.
        layer : str or None (default)
            Vector layer, if implemented by vector driver.
        driver : str or None (default)
            Vector driver. Default None will try to determine driver from
            fname.
        kwargs : dict, optional
            Other driver-specific parameters that will be interpreted by the
            OGR library as layer creation or opening options.

        Raises
        ------
        ModuleNotFoundError
            If fiona and/or shapely is not installed.
        """
        from gridit.file import write_vector

        write_vector(
            self, array=array, fname=fname, attribute=attribute, layer=layer,
            driver=driver, **kwargs)
