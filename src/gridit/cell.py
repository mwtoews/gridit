"""Cell geometry methods."""

from warnings import warn

import numpy as np


def cell_geoms(self, *, mask=None, order="C", style="square", **kwargs):
    """Returns array of shapely geometry objects for grid cells.

    The flat array is indexed in C-order, such that rows and columns can
    be evaluated using (e.g.) :func:`numpy.unravel_index`.

    Parameters
    ----------
    mask : array_like, optional
        Optional 2D bool array with same shape as grid, used to limit the
        size of the array.
    order : {"C", "F"}, optional
        C-style indexes are row-major, and Fortran-style are column-major.
        See :meth:`numpy.ravel` for more information.
    style : {"square", "mid", "row", "col"}, optional
        Geometry style. Default "square" will return square polygons,
        "mid" will return centre points, "row" will return grid lines along
        rows, and "col" will return grid lines along columns.
    **kwargs
        Keyword `point` is deprecated, use (e.g.) `style="mid"` instead.

    Returns
    -------
    array_like
        Shapely geometry objects.

    Raises
    ------
    ModuleNotFoundError
        If shapely is not installed.

    Examples
    --------
    >>> from gridit import Grid
    >>> import numpy as np
    >>> g = Grid(10, (2, 3))
    >>> rows, cols = np.unravel_index(np.arange(2 * 3), g.shape)
    >>> for row, col, geom in zip(rows, cols, g.cell_geoms()):
    ...     print(f"({row}, {col}): {geom}")
    (0, 0): POLYGON ((0 0, 10 0, 10 -10, 0 -10, 0 0))
    (0, 1): POLYGON ((10 0, 20 0, 20 -10, 10 -10, 10 0))
    (0, 2): POLYGON ((20 0, 30 0, 30 -10, 20 -10, 20 0))
    (1, 0): POLYGON ((0 -10, 10 -10, 10 -20, 0 -20, 0 -10))
    (1, 1): POLYGON ((10 -10, 20 -10, 20 -20, 10 -20, 10 -10))
    (1, 2): POLYGON ((20 -10, 30 -10, 30 -20, 20 -20, 20 -10))
    >>> for row, col, geom in zip(rows, cols, g.cell_geoms(style="mid")):
    ...     print(f"({row}, {col}): {geom}")
    (0, 0): POINT (5 -5)
    (0, 1): POINT (15 -5)
    (0, 2): POINT (25 -5)
    (1, 0): POINT (5 -15)
    (1, 1): POINT (15 -15)
    (1, 2): POINT (25 -15)
    """
    try:
        import shapely
    except ModuleNotFoundError:
        raise ModuleNotFoundError("cell_geoms() needs shapely to be installed")

    if mask is not None:
        if getattr(mask, "shape", None) != self.shape:
            raise ValueError("mask must be an array the same shape as the grid")
        if not np.issubdtype(mask.dtype, np.bool_):
            mask = mask.astype(bool)
        if mask.all():
            return np.empty((0,), dtype=object)
        sel = ~mask.ravel(order=order)
        if sel.all():
            mask = None

    if "point" in kwargs:
        point = kwargs.pop("point")
        msg = "keyword `point` is deprecated"
        if point is True:
            msg += '; use `style="mid"` instead'
            style = "mid"
        elif point is False:
            msg += f'; using `style="{style}"` instead'
        warn(msg, DeprecationWarning, stacklevel=2)
    if kwargs:
        raise TypeError(f"unhandled keyword arguments: {kwargs}")

    valid_style = {"square", "mid", "row", "col"}
    if style not in valid_style:
        raise ValueError(f"style must be one of {valid_style}; found {style=}")

    if style in {"mid", "row", "col"}:
        xc, yc = self.meshgrid(corner=False)

    if style == "mid":
        # grid cells as grid-centre Point geometries
        x = xc.ravel(order=order)
        y = yc.ravel(order=order)
        if mask is not None:
            x = x[sel]
            y = y[sel]
        try:  # shapely 2+: use vectorized version
            geoms = shapely.points(x, y)
        except AttributeError:  # shapely 1.x
            from shapely.geometry import Point

            geoms_list = []
            for xc, yc in zip(x, y):
                geoms_list.append(Point(xc, yc))
            try:
                geoms = np.array(geoms_list)
            except NotImplementedError:
                # shapely<1.8
                geoms = np.empty(len(geoms_list), dtype=object)
                for idx, geom in enumerate(geoms_list):
                    geoms[idx] = geom
        return geoms

    # Other geometry styles
    nrow, ncol = self.shape
    Ia, Ja = np.ogrid[0:nrow, 0:ncol]
    xv, yv = self.meshgrid(corner=True)

    # Create arrays of coordinates for cells
    if style == "square":
        # grid cells as rectangular Polygon geometries, 4 coords
        ncoords = 4
        xcoords = [
            xv[Ia, Ja],
            xv[Ia, Ja + 1],
            xv[Ia + 1, Ja + 1],
            xv[Ia + 1, Ja],
        ]
        ycoords = [
            yv[Ia, Ja],
            yv[Ia, Ja + 1],
            yv[Ia + 1, Ja + 1],
            yv[Ia + 1, Ja],
        ]
    elif style == "row":
        # grid cells as LineString geometries along rows, 2 coords
        ncoords = 2
        xcoords = [xv[Ia, Ja], xv[Ia, Ja + 1]]
        ycoords = [yc[Ia, Ja], yc[Ia, Ja]]
    elif style == "col":
        # grid cells as LineString geometries along columns, 2 coords
        ncoords = 2
        xcoords = [xc[Ia, Ja], xc[Ia, Ja]]
        ycoords = [yv[Ia, Ja], yv[Ia + 1, Ja]]

    if order == "C":
        xcoords = np.stack(xcoords).transpose((1, 2, 0))
        ycoords = np.stack(ycoords).transpose((1, 2, 0))
    elif order == "F":
        xcoords = np.stack(xcoords)
        ycoords = np.stack(ycoords)
    else:
        raise ValueError('order must be "C" or "F"')

    try:  # shapely 2+: use vectorized version
        x = xcoords.ravel(order=order)
        y = ycoords.ravel(order=order)
        if mask is None:
            indices = np.repeat(np.arange(nrow * ncol), ncoords)
        else:
            selrep = np.repeat(sel, ncoords)
            x = x[selrep]
            y = y[selrep]
            indices = np.repeat(np.arange(sel.sum()), ncoords)
        if style == "square":
            rings = shapely.linearrings(x, y=y, indices=indices)
            geoms = shapely.polygons(rings)
        else:  # linestring geometries for row or column
            geoms = shapely.linestrings(x, y=y, indices=indices)
    except AttributeError:  # shapely 1.x
        from itertools import product

        if style == "square":
            from shapely.geometry import Polygon as GeomType
        else:
            from shapely.geometry import LineString as GeomType

        geoms_list = []
        if order == "C":
            swap = False
            prod_iter = product(range(nrow), range(ncol))
        elif order == "F":
            swap = True
            prod_iter = product(range(ncol), range(nrow))
            xcoords = xcoords.transpose((1, 2, 0))
            ycoords = ycoords.transpose((1, 2, 0))
        for idx, ij in enumerate(prod_iter):
            if mask is not None and not sel[idx]:
                continue
            if swap:
                j, i = ij
            else:
                i, j = ij
            geoms_list.append(GeomType(zip(xcoords[i, j], ycoords[i, j])))
        try:
            geoms = np.array(geoms_list)
        except NotImplementedError:
            # shapely<1.8
            geoms = np.empty(len(geoms_list), dtype=object)
            for idx, geom in enumerate(geoms_list):
                geoms[idx] = geom
    return geoms


def cell_geoseries(self, *, mask=None, order="C", style="square", **kwargs):
    """Return GeoSeries from cell geometries.

    Parameters
    ----------
    mask : array_like, optional
        Optional 2D bool array with same shape as grid, used to limit the
        size of the series.
    order : {"C", "F"}, optional
        C-style indexes are row-major, and Fortran-style are column-major.
        See :meth:`numpy.ravel` for more information.
    style : {"square", "mid", "row", "col"}, optional
        Geometry style. Default "square" will return rectangular polygons,
        "mid" will return centre points, "row" will return grid lines along
        rows, and "col" will return grid lines along columns.
    **kwargs
        Keyword `point` is deprecated, use (e.g.) `style="mid"` instead.

    Returns
    -------
    pandas.GeoSeries

    Raises
    ------
    ModuleNotFoundError
        If geopandas is not installed.

    See Also
    --------
    Grid.cell_geoms : Return array of shapely geometries for grid cells.
    Grid.cell_geodataframe : Return GeoDataFrame from cell geometries.

    Examples
    --------
    >>> from gridit import Grid
    >>> grid = Grid(10, (2, 3), projection="EPSG:3857")
    >>> gs = grid.cell_geoseries()

    """
    try:
        import geopandas
    except ModuleNotFoundError:
        raise ModuleNotFoundError("cell_geoseries() needs geopandas to be installed")

    if "point" in kwargs:
        point = kwargs.pop("point")
        msg = "keyword `point` is deprecated"
        if point is True:
            msg += '; use `style="mid"` instead'
            style = "mid"
        elif point is False:
            msg += f'; using `style="{style}"` instead'
        warn(msg, DeprecationWarning, stacklevel=2)
    if kwargs:
        raise TypeError(f"unhandled keyword arguments: {kwargs}")

    geoms = self.cell_geoms(mask=mask, order=order, style=style)
    gs = geopandas.GeoSeries(geoms, crs=self.projection)
    if mask is not None:
        if not np.issubdtype(mask.dtype, np.bool_):
            mask = mask.astype(bool)
        if mask.all():
            return gs[0:0]
        sel = ~mask.ravel(order=order)
        if not sel.all():
            gs.index = np.where(sel)[0]
    return gs


def cell_geodataframe(
    self, *, values=None, mask=None, order="C", style="square", **kwargs
):
    """Return GeoDataFrame from cell geometries, rows and columns.

    Parameters
    ----------
    values : dict, optional
        Add values to DataFrame from dict, where keys are names and values
        are arrays the same shape as grid.
    mask : array_like, optional
        Optional 2D bool array with same shape as grid, used to limit the
        size of the series.
    order : {"C", "F"}, optional
        C-style indexes are row-major, and Fortran-style are column-major.
        See :meth:`numpy.ravel` for more information.
    style : {"square", "mid", "row", "col"}, optional
        Geometry style. Default "square" will return rectangular polygons,
        "mid" will return centre points, "row" will return grid lines along
        rows, and "col" will return grid lines along columns.
    **kwargs
        Keyword `point` is deprecated, use (e.g.) `style="mid"` instead.

    Returns
    -------
    pandas.GeoDataFrame

    Raises
    ------
    ModuleNotFoundError
        If geopandas is not installed.

    See Also
    --------
    Grid.cell_geoms : Return array of shapely geometries for grid cells.
    Grid.cell_geoseries : Return pandas.GeoSeries from cell geometries.

    Examples
    --------
    >>> from gridit import Grid
    >>> grid = Grid(10, (2, 3), projection="EPSG:3857")
    >>> gdf = grid.cell_geodataframe(values={"a": np.ones(grid.shape)})
    >>> gdf[["row", "col", "a"]]
       row  col    a
    0    0    0  1.0
    1    0    1  1.0
    2    0    2  1.0
    3    1    0  1.0
    4    1    1  1.0
    5    1    2  1.0
    >>> rows = grid.cell_geodataframe(style="row").dissolve("row").line_merge()
    >>> rows.simplify(0)
    row
    0      LINESTRING (0 -5, 30 -5)
    1    LINESTRING (0 -15, 30 -15)
    dtype: geometry
    >>> cols = grid.cell_geodataframe(style="col").dissolve("col").line_merge()
    >>> cols.simplify(0)
    col
    0      LINESTRING (5 0, 5 -20)
    1    LINESTRING (15 0, 15 -20)
    2    LINESTRING (25 0, 25 -20)
    dtype: geometry

    """
    try:
        import geopandas
    except ModuleNotFoundError:
        raise ModuleNotFoundError("cell_geodataframe() needs geopandas to be installed")
    if "point" in kwargs:
        point = kwargs.pop("point")
        msg = "keyword `point` is deprecated"
        if point is True:
            msg += '; use `style="mid"` instead'
            style = "mid"
        elif point is False:
            msg += f'; using `style="{style}"` instead'
        warn(msg, DeprecationWarning, stacklevel=2)
    if kwargs:
        raise TypeError(f"unhandled keyword arguments: {kwargs}")

    gs = self.cell_geoseries(mask=mask, order=order, style=style)
    gdf = geopandas.GeoDataFrame(geometry=gs, crs=self.projection)
    if mask is not None:
        if not np.issubdtype(mask.dtype, np.bool_):
            mask = mask.astype(bool)
        sel = ~mask.ravel(order=order)
        if not sel.all():
            gdf.index = np.where(sel)[0]
    else:
        sel = np.ones(len(gdf), bool)
    rows, cols = np.unravel_index(gs.index, self.shape, order=order)
    gdf["row"] = rows
    gdf["col"] = cols
    if values is not None:
        if not isinstance(values, dict):
            raise ValueError("values must be dict")
        for name, array in values.items():
            if not isinstance(name, str):
                raise ValueError("key for values must be str")
            if getattr(array, "shape", None) != self.shape:
                raise ValueError(
                    f"array {name!r} in values must have the same shape as the grid"
                )
            gdf[name] = array.ravel(order=order)[sel]
    return gdf
