"""Gridit package."""
__license__ = "BSD"
__author__ = "Mike Taves"
__email__ = "mwtoews@gmail.com"

try:
    from ._version import __version__
except ImportError:
    __version__ = None

__all__ = ["Grid", "GridPolyConv"]

from .grid import Grid
from .gridpolyconv import GridPolyConv
