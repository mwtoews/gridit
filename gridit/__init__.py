"""Gridit package."""

__license__ = "BSD"
__author__ = "Mike Taves"
__email__ = "mwtoews@gmail.com"

try:
    from gridit._version import __version__
except ImportError:
    __version__ = None

__all__ = ["Grid", "GridPolyConv", "logger"]

from gridit import logger
from gridit.grid import Grid
from gridit.gridpolyconv import GridPolyConv
