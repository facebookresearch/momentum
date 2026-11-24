#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Finish PyPI pymomentum wheels: add the install check.

Up to 0.1.114, pymomentum-cpu and pymomentum-gpu were full packages that
installed every module under pymomentum/. Two install states then break:

- such an add-on stays installed next to a newer pymomentum-core;
- a plain `pip install -U` of the add-on installs the new core first and then
  uninstalls the old add-on, which deletes every path in the old add-on's
  RECORD, including core files at the same paths.

The check raises ImportError with the repair commands in either state. It
runs from the first of these that exists:

- pymomentum/__init__.py, from pymomentum-core;
- pymomentum/__init__.pyw, from pymomentum-core on Windows, where Python also
  imports .pyw sources; pip removes __init__.py, .pyc and .pyo files listed in
  an old RECORD, but not .pyw;
- pymomentum/__init__.pyc, a sourceless copy from pymomentum-cpu or
  pymomentum-gpu, which pip installs after removing the old add-on.

Usage: finalize_pypi_wheel.py WHEEL...
"""

import ast
import base64
import hashlib
import importlib.util
import marshal
import re
import subprocess
import sys
import zipfile
from pathlib import Path

INIT = "pymomentum/__init__.py"
INIT_PYC = "pymomentum/__init__.pyc"
INIT_PYW = "pymomentum/__init__.pyw"
GUARD_MARKER = "# pymomentum install check"
GUARD = (
    GUARD_MARKER
    + """
def _check_pymomentum_install():
    import os

    def _read(path):
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""

    package_dir = os.path.dirname(os.path.abspath(__file__))
    site_dir = os.path.dirname(package_dir)
    try:
        package_names = os.listdir(package_dir)
        site_names = os.listdir(site_dir)
    except OSError:
        return
    problems = []
    if not any(
        name.startswith("geometry.") and name.endswith((".so", ".pyd"))
        for name in package_names
    ):
        problems.append("pymomentum-core files are missing")
    pip_distributions = ["pymomentum-core"]
    conda_distributions = []
    for name in site_names:
        if not (
            name.startswith(("pymomentum_cpu-", "pymomentum_gpu-"))
            and name.endswith(".dist-info")
        ):
            continue
        distribution, version = name[: -len(".dist-info")].split("-")[:2]
        distribution = distribution.replace("_", "-")
        metadata = _read(os.path.join(site_dir, name, "METADATA"))
        installer = _read(os.path.join(site_dir, name, "INSTALLER")).strip()
        if installer == "conda":
            conda_distributions.append(distribution)
        else:
            pip_distributions.append(distribution)
        if metadata and "Requires-Dist: pymomentum-core" not in metadata:
            problems.append(
                f"{distribution} {version} is a full package that overlaps "
                "pymomentum-core"
            )
    if problems:
        commands = []
        if conda_distributions:
            commands.append("conda remove -y " + " ".join(conda_distributions))
        commands.append("pip uninstall -y " + " ".join(pip_distributions))
        commands.append(
            "pip install " + (" ".join(pip_distributions[1:]) or "pymomentum-core")
        )
        raise ImportError(
            "Broken pymomentum installation: "
            + "; ".join(problems)
            + ". Repair with:\\n    "
            + "\\n    ".join(commands)
        )


_check_pymomentum_install()
del _check_pymomentum_install
"""
)


def record_line(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{name},sha256={digest.decode()},{len(data)}"


def wheel_python(wheel: Path) -> tuple[int, int]:
    match = re.search(r"-cp(\d)(\d+)-", wheel.name)
    if not match:
        raise RuntimeError(f"{wheel.name} has no CPython tag")
    return int(match[1]), int(match[2])


def compile_init(source: str) -> bytes:
    data = source.encode("utf-8")
    code = compile(data, INIT, "exec", dont_inherit=True)
    # Same layout py_compile writes for an unchecked hash-based .pyc.
    pyc = bytearray(importlib.util.MAGIC_NUMBER)
    pyc += (0b01).to_bytes(4, "little")
    pyc += importlib.util.source_hash(data)
    pyc += marshal.dumps(code)
    return bytes(pyc)


def add_guard(init: str) -> str:
    """Insert the check after any docstring and `from __future__` imports."""
    if GUARD_MARKER in init:
        return init
    lines = init.splitlines(keepends=True)
    insert_at = 0
    for node in ast.parse(init).body:
        is_docstring = (
            insert_at == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if not (is_docstring or is_future):
            break
        insert_at = node.end_lineno
    result = "".join(lines[:insert_at]) + GUARD + "".join(lines[insert_at:])
    compile(result, INIT, "exec", dont_inherit=True)
    return result


def finalize_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as source:
        members = {
            info.filename: (info, source.read(info))
            for info in source.infolist()
            if not info.filename.endswith("/")
        }
    record_name = next(name for name in members if name.endswith(".dist-info/RECORD"))
    del members[record_name]

    if wheel.name.startswith("pymomentum_core-"):
        init_info, init_data = members.get(INIT, (zipfile.ZipInfo(INIT), b""))
        init = add_guard(init_data.decode("utf-8"))
        members[INIT] = (init_info, init.encode("utf-8"))
        if "-win" in wheel.name:
            members[INIT_PYW] = (zipfile.ZipInfo(INIT_PYW), init.encode("utf-8"))
    else:
        members[INIT_PYC] = (zipfile.ZipInfo(INIT_PYC), compile_init(GUARD))

    records = [record_line(name, data) for name, (_, data) in members.items()]
    records.append(f"{record_name},,")
    temporary = wheel.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as target:
        for name, (info, data) in members.items():
            new_info = zipfile.ZipInfo(name, date_time=info.date_time)
            new_info.external_attr = info.external_attr or (0o644 << 16)
            new_info.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(new_info, data)
        target.writestr(record_name, "\n".join(records) + "\n")
    temporary.replace(wheel)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    for arg in argv:
        major, minor = wheel_python(Path(arg))
        if (major, minor) != sys.version_info[:2]:
            # The sourceless .pyc only loads on the Python version that wrote it.
            python = Path(f"/opt/python/cp{major}{minor}-cp{major}{minor}/bin/python")
            if not python.exists():
                raise RuntimeError(f"{arg} needs Python {major}.{minor}")
            subprocess.check_call([str(python), __file__, arg])
            continue
        finalize_wheel(Path(arg))
        print(f"{arg}: added the install check")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
