# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

project = "PyMomentum"

extensions = [
    "sphinx.ext.autodoc",
]

# Configure autodoc to include constructors and other special methods
autodoc_default_options = {
    "special-members": "__init__",
    "show-inheritance": True,
    "members": True,
    "undoc-members": True,
}

exclude_patterns = [
    ".pixi",
    "build",
]

html_theme = "sphinx_rtd_theme"


def skip_pycapsule(app, what, name, obj, skip, options):
    """Skip PyCapsule members to avoid RST warnings from their docstrings.

    PyCapsule is an internal CPython/pybind11 type used for cross-module type
    sharing. Its __init__ docstring contains '*' characters that trigger RST
    parsing warnings. exclude-members only hides it from output but doesn't
    prevent docstring parsing; this event fires earlier and skips it entirely.
    """
    if name == "PyCapsule":
        return True
    return skip


_BARE_GEOMETRY_TYPE = re.compile(r"(?<![\w.])(Character|Mesh)\b")


def qualify_geometry_types(app, what, name, obj, options, signature, return_annotation):
    """Qualify bare ``Character``/``Mesh`` annotations as ``pymomentum.geometry`` types.

    Modules such as :mod:`pymomentum.rerun_vis` import these names only under
    ``TYPE_CHECKING``, so autodoc renders them unqualified, and Sphinx then finds
    two targets (:mod:`pymomentum.geometry` and :mod:`pymomentum.torch.character`).
    """
    if name.startswith("pymomentum.torch."):
        return None

    def qualify(text):
        if text is None:
            return None
        return _BARE_GEOMETRY_TYPE.sub(r"pymomentum.geometry.\1", text)

    return qualify(signature), qualify(return_annotation)


def setup(app):
    app.connect("autodoc-skip-member", skip_pycapsule)
    app.connect("autodoc-process-signature", qualify_geometry_types)
