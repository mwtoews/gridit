"""Spatial utilities module."""

__all__ = ["is_same_crs"]


def is_same_crs(wkt1, wkt2):
    """Determine if two CRS strings (as WKT) are nearly the same.

    Uses SequenceMatcher from difflib to determine if ratio is greater
    than 2/3.
    """
    if wkt1.upper() == wkt2.upper():
        return True
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, wkt2, wkt1).ratio()
    return ratio > 2/3.
