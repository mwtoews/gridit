"""Support for PEST's Grid Specification File."""

from os import PathLike
from typing import TextIO

import numpy as np

from gridit.logger import get_logger


@classmethod
def from_specfile(
    cls,
    fname: str | PathLike | TextIO,
    *,
    projection: str | None = None,
    logger=None,
):
    """Fetch grid information from PEST's Grid Specification file.

    Parameters
    ----------
    fname : str, PathLike or TextIO
        Input PEST Grid Specification file.
    projection : optional str, default None
        WKT coordinate reference system string.
    logger : logging.Logger, optional
        Logger to show messages.
    """
    if logger is None:
        logger = get_logger(cls.__name__)

    def read_items(fp: TextIO, num_items: int) -> list[str]:
        """Read list items individually or with multipliers N*value"""
        items: list[str] = []
        while (len(items) < num_items) and (line := fp.readline()):
            for item in line.replace(",", " ").split():
                if "*" in item:
                    n, b = item.split("*")
                    items += int(n) * [b]
                else:
                    items.append(item)
        if len(items) < num_items:
            raise ValueError(f"expected {num_items} item(s) but found {len(items)}")
        if len(items) > num_items:
            logger.warning("too many items gathered; trimming")
            items = items[:num_items]
        return items

    def read_file_object(fp: TextIO) -> tuple:
        # Line 1: nrow, ncol
        line = fp.readline().replace(",", " ")
        nrow, ncol = map(int, line.split(maxsplit=2)[:2])
        # Line 2: ulx, uly, rotation
        line = fp.readline().replace(",", " ")
        ulx, uly, rotation = map(float, line.split(maxsplit=3)[:3])
        # Line 3+: list of grid row widths in order of increasing column index
        delr = np.array(read_items(fp, ncol), float)
        # Line 4+: list of grid cell widths in the column direction
        delc = np.array(read_items(fp, nrow), float)
        return nrow, ncol, ulx, uly, rotation, delr, delc

    if isinstance(fname, str | PathLike):
        with open(fname) as fp:
            nrow, ncol, ulx, uly, rotation, delr, delc = read_file_object(fp)
    elif hasattr(fname, "readline"):
        nrow, ncol, ulx, uly, rotation, delr, delc = read_file_object(fname)

    if rotation != 0.0:
        logger.error("rotated specfile grids are not supported")
    if not (delr[0] == delr).all():
        logger.error("specfile delr is not constant %s", delr[0])
    if not (delc[0] == delc).all():
        logger.error("specfile delc is not constant %s", delc[0])
    if delr[0] != delc[0]:
        logger.error(
            "specfile delr and delc are different (%s vs. %s)", delr[0], delc[0]
        )

    return cls(
        resolution=delr[0],
        shape=(nrow, ncol),
        top_left=(ulx, uly),
        projection=projection,
        logger=logger,
    )


def write_specfile(grid, fname: str | PathLike | TextIO | None) -> None | str:
    """Write PEST's Grid Specification file

    Parameters
    ----------
    fname : str, PathLike, TextIO or None
        Output file to write. Use None to return a str of file contents.

    Returns
    -------
    None or str
        Returns file contents as str if `fname` is None.
    """
    nrow, ncol = grid.shape
    ulx, uly = grid.top_left
    resolution = grid.resolution
    rotation = 0.0
    lines = [
        f"{nrow} {ncol}\n",
        f"{ulx} {uly} {rotation}\n",
    ]
    lines.append((f"{ncol}*" if ncol > 1 else "") + f"{resolution}\n")
    lines.append((f"{nrow}*" if nrow > 1 else "") + f"{resolution}\n")

    if fname is None:
        return "".join(lines)
    if isinstance(fname, str | PathLike):
        with open(fname, "w") as fp:
            fp.writelines(lines)
    elif hasattr(fname, "writelines"):
        fname.writelines(lines)
    else:
        raise TypeError(fname)
    return None
