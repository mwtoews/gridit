"""Pytest fixtures and features."""

import re
from importlib import metadata

import pytest

from .common import outdir

try:
    # avoid fiona._env:transform.py:94 Unable to open EPSG support file gcs.csv
    import fiona

    del fiona
except ModuleNotFoundError:
    pass


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
