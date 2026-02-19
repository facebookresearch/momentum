# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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


def setup(app):
    app.connect("autodoc-skip-member", skip_pycapsule)
