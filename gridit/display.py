"""Display utilities module."""

__all__ = ["shorten", "print_array"]

import numpy as np

from .logger import get_logger


def shorten(text, width):
    """Simlar to textwrap.shorten, but works with WKT."""
    text = text.strip()
    if len(text) < width:
        return text
    else:
        return text[:(width - 5)] + "[...]"


def print_array(ar, logger=None):
    """Print 2D array to ASCII."""
    if logger is None:
        logger = get_logger("print_array")
    uvals = np.unique(ar)
    if (~ar.mask).all() and len(uvals) == 1:
        logger.info("all raster values are %s", uvals[0])
    try:
        import shutil
        from scipy import ndimage
        cols, rows = shutil.get_terminal_size((50, 20))
        # fit either width or height
        r2c = 2.0
        zf1 = float(cols - 1) / float(ar.shape[1])
        zf0 = zf1 / r2c
        zf_cols = (zf0, zf1)
        zf0 = float(rows - 3) / float(ar.shape[0])
        zf1 = zf0 * r2c
        zf_rows = (zf0, zf1)
        if zf_rows[0] < zf_cols[0]:
            zf = zf_rows
        else:
            zf = zf_cols
        im = ndimage.zoom(ar.filled(ar.min()).astype(float), zf, order=0)
        if im.min() == im.max():
            im.fill(0.5)
        else:
            im -= im.min()
            im /= im.max()
        msk = ndimage.zoom(ar.mask, zf, order=0, cval=True)
        col = ".;-:!>7?8CO$QHNM"
        string = ""
        height, width = im.shape
        for h in range(height):
            for w in range(width):
                if msk[h, w]:
                    string += ' '
                else:
                    string += col[int(im[h, w] * 15)]
            string += "\n"
        print(string[:-1])
    except ModuleNotFoundError:
        print(ar)
    info = f"min: {ar.min()!s}, max: {ar.max()!s}, "
    if len(uvals) < 8:
        info += f"unique values: {uvals}"
    else:
        info += f"number of unique values: {len(uvals)}"
    logger.info(info)
