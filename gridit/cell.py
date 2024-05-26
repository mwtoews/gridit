"""Cell geometry methods."""

import numpy as np


def cell_geoms(self, *, mask=None, order="C"):
    """Returns array of shapely Polygon objects for grid cells.

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

    Returns
    -------
    array_like
        Shapely Polygon geometry objects.

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

    nrow, ncol = self.shape
    xmin, ymax = self.top_left
    xedge = np.arange(ncol + 1) * self.resolution + xmin
    yedge = np.arange(nrow + 1) * -self.resolution + ymax
    xvertices, yvertices = np.meshgrid(xedge, yedge)

    # arrays of coordinates for rectangle cells
    Ia, Ja = np.ogrid[0:nrow, 0:ncol]
    xverts = np.stack(
        [
            xvertices[Ia, Ja],
            xvertices[Ia, Ja + 1],
            xvertices[Ia + 1, Ja + 1],
            xvertices[Ia + 1, Ja],
        ]
    )
    yverts = np.stack(
        [
            yvertices[Ia, Ja],
            yvertices[Ia, Ja + 1],
            yvertices[Ia + 1, Ja + 1],
            yvertices[Ia + 1, Ja],
        ]
    )
    if order == "C":
        xverts = xverts.transpose((1, 2, 0))
        yverts = yverts.transpose((1, 2, 0))
    elif order == "F":
        pass
    else:
        raise ValueError('order must be "C" or "F"')

    try:  # shapely 2+: use vectorized version
        if mask is None:
            rings = shapely.linearrings(
                xverts.ravel(order=order),
                y=yverts.ravel(order=order),
                indices=np.repeat(np.arange(nrow * ncol), 4),
            )
        else:
            selrep = np.repeat(sel, 4)
            rings = shapely.linearrings(
                xverts.ravel(order=order)[selrep],
                y=yverts.ravel(order=order)[selrep],
                indices=np.repeat(np.arange(sel.sum()), 4),
            )
        geoms = shapely.polygons(rings)
    except AttributeError:  # shapely 1.x
        from itertools import product

        from shapely.geometry import Polygon

        geoms_list = []
        if order == "C":
            swap = False
            prod_iter = product(range(nrow), range(ncol))
        elif order == "F":
            swap = True
            prod_iter = product(range(ncol), range(nrow))
            xverts = xverts.transpose((1, 2, 0))
            yverts = yverts.transpose((1, 2, 0))
        for idx, ij in enumerate(prod_iter):
            if mask is not None and not sel[idx]:
                continue
            if swap:
                j, i = ij
            else:
                i, j = ij
            geoms_list.append(Polygon(zip(xverts[i, j], yverts[i, j])))
        try:
            geoms = np.array(geoms_list)
        except NotImplementedError:
            # shapely<1.8
            geoms = np.empty(len(geoms_list), dtype=object)
            for idx, geom in enumerate(geoms_list):
                geoms[idx] = geom
    return geoms


def cell_geoseries(self, *, mask=None, order="C"):
    """Return GeoSeries from cell geometries.

    Parameters
    ----------
    mask : array_like, optional
        Optional 2D bool array with same shape as grid, used to limit the
        size of the series.
    order : {"C", "F"}, optional
        C-style indexes are row-major, and Fortran-style are column-major.
        See :meth:`numpy.ravel` for more information.

    Returns
    -------
    pandas.GeoSeries

    Raises
    ------
    ModuleNotFoundError
        If geopandas is not installed.

    See Also
    --------
    Grid.cell_geoms : Return array of shapely Polygons for grid cells.
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
    geoms = self.cell_geoms(mask=mask, order=order)
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


def cell_geodataframe(self, *, values=None, mask=None, order="C"):
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

    Returns
    -------
    pandas.GeoDataFrame

    Raises
    ------
    ModuleNotFoundError
        If geopandas is not installed.

    See Also
    --------
    Grid.cell_geoms : Return array of shapely Polygons for grid cells.
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

    """
    try:
        import geopandas
    except ModuleNotFoundError:
        raise ModuleNotFoundError("cell_geodataframe() needs geopandas to be installed")
    gs = self.cell_geoseries(mask=mask, order=order)
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
            elif getattr(array, "shape", None) != self.shape:
                raise ValueError(
                    f"array {name!r} in values must have the same shape " "as the grid"
                )
            gdf[name] = array.ravel(order=order)[sel]
    return gdf
