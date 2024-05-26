"""Common code for testing."""

import contextlib
import importlib
import os
import re
import sys
from importlib import metadata
from pathlib import Path
from subprocess import PIPE, Popen

import pytest

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
    args = [sys.executable, "-m"] + [str(g) for g in args]
    print("running: " + " ".join(args))
    p = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode()
    stderr = stderr.decode()
    returncode = p.returncode
    return stdout, stderr, returncode


def pytest_report_header(config):
    """Header for pytest to show versions of packages."""
    required = []
    extra = {}
    for item in metadata.requires("gridit"):
        pkg_name = re.findall(r"[a-z0-9_\-]+", item, re.IGNORECASE)[0]
        if res := re.findall("extra == ['\"](.+)['\"]", item):
            assert len(res) == 1, item
            pkg_extra = res[0]
            if pkg_extra not in extra:
                extra[pkg_extra] = []
            extra[pkg_extra].append(pkg_name)
        else:
            required.append(pkg_name)

    processed = set()
    lines = []
    items = []
    for name in required:
        processed.add(name)
        try:
            version = metadata.version(name)
            items.append(f"{name}-{version}")
        except metadata.PackageNotFoundError:
            items.append(f"{name} (not found)")
    lines.append("required packages: " + ", ".join(items))
    installed = []
    not_found = []
    for name in extra["optional"]:
        if name in processed:
            continue
        processed.add(name)
        try:
            version = metadata.version(name)
            installed.append(f"{name}-{version}")
        except metadata.PackageNotFoundError:
            not_found.append(name)
    if installed:
        lines.append("optional packages: " + ", ".join(installed))
    if not_found:
        lines.append("optional packages not found: " + ", ".join(not_found))
    return "\n".join(lines)


def pytest_addoption(parser):
    parser.addoption(
        "--write-files", action="store_true", help="Write files to tests/data/out"
    )


@pytest.fixture
def write_files(request):
    if not outdir.exists():
        outdir.mkdir()
    return request.config.getoption("--write-files")
