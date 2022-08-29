"""Common code for testing."""
import contextlib
import importlib
import os
import pkg_resources
import pytest
import sys
from pathlib import Path
from subprocess import Popen, PIPE

try:
    # avoid fiona._env:transform.py:94 Unable to open EPSG support file gcs.csv
    import fiona
    del fiona
except ModuleNotFoundError:
    pass

_has_pkg_cache = {}


def has_pkg(pkg):
    """Return True if Python package is available."""
    if pkg not in _has_pkg_cache:
        try:
            _has_pkg_cache[pkg] = bool(importlib.import_module(pkg))
        except ModuleNotFoundError:
            try:
                _has_pkg_cache[pkg] = bool(pkg_resources.get_distribution(pkg))
            except pkg_resources.DistributionNotFound:
                _has_pkg_cache[pkg] = False

    return _has_pkg_cache[pkg]


def requires_pkg(*pkgs):
    """Use to skip tests that don't have required packages."""
    missing = {pkg for pkg in pkgs if not has_pkg(pkg)}
    return pytest.mark.skipif(
        missing,
        reason=f"missing package{'s' if len(missing) != 1 else ''}: " +
               ", ".join(missing),
    )


datadir = Path("tests") / "data"


@contextlib.contextmanager
def set_env(**environ):
    """Temporarily set the process environment variables."""
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


def run_cli(args):
    """Run a Python command, return tuple (stdout, stderr, returncode)."""
    args = [sys.executable, "-m"] + [str(g) for g in args]
    print("running: " + " ".join(args))
    p = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode()
    stderr = stderr.decode()
    returncode = p.returncode
    return stdout, stderr, returncode
