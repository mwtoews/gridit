"""Module logger."""

__all__ = ["disable_logger", "get_logger", "module_logger"]

import logging
from contextlib import contextmanager

_module_logger_name = __name__
module_logger = logging.getLogger(_module_logger_name)
if _module_logger_name not in [_.name for _ in module_logger.handlers]:
    if logging.root.handlers:
        module_logger.addHandler(logging.root.handlers[0])
    else:
        import sys

        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s:%(message)s", "%H:%M:%S"
        )
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.name = _module_logger_name
        handler.setFormatter(formatter)
        module_logger.addHandler(handler)
        del sys, formatter, handler


@contextmanager
def disable_logger(logger):
    """Context manager that will disable logging messages."""
    toggle = not logger.disabled
    if toggle:
        logger.disabled = True
    try:
        yield
    finally:
        if toggle:
            logger.disabled = False


def get_logger(name, level=None):
    """Return a named module logger."""
    logger = logging.getLogger(_module_logger_name)
    logger.name = name
    if level is None:
        level = module_logger.level
    logger.setLevel(level)
    return logger
