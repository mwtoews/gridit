"""Common code for testing."""

import contextlib
import importlib
import os
import sys
from importlib import metadata
from pathlib import Path
from subprocess import PIPE, Popen

import pytest

_has_pkg_cache = {}


def has_pkg(pkg):
    """Return True if Python package is available."""
    if pkg not in _has_pkg_cache:
        try:
            metadata.distribution(pkg)
            _has_pkg_cache[pkg] = True
        except metadata.PackageNotFoundError:
            try:
                importlib.import_module(pkg)
                _has_pkg_cache[pkg] = True
            except ModuleNotFoundError:
                _has_pkg_cache[pkg] = False

    return _has_pkg_cache[pkg]


def requires_pkg(*pkgs):
    """Use to skip tests that don't have required packages."""
    missing = {pkg for pkg in pkgs if not has_pkg(pkg)}
    return pytest.mark.skipif(
        missing,
        reason=f"missing package{'s' if len(missing) != 1 else ''}: "
        + ", ".join(missing),
    )


datadir = Path("tests") / "data"
outdir = datadir / "out"


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
    args = [sys.executable, "-W", "ignore", "-m"] + [str(g) for g in args]
    print("running: " + " ".join(args))
    p = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode()
    stderr = stderr.decode()
    returncode = p.returncode
    return stdout, stderr, returncode


def get_str_list_count(find: str, msgs: list[str]) -> int:
    """Count 'find' matches in 'msgs'"""
    return sum(find in msg for msg in msgs)
