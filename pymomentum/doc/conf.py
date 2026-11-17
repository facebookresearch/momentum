# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import re

from sphinx.ext.autodoc import FunctionDocumenter
from sphinx.ext.napoleon import Config as NapoleonConfig
from sphinx.ext.napoleon.docstring import GoogleDocstring

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


class TorchScriptFunctionDocumenter(FunctionDocumenter):
    """Document ``torch.jit.script`` functions through the Python function they wrap.

    ``torch.jit.ScriptFunction`` is not a Python function, so autodoc would otherwise
    skip these members entirely.
    """

    objtype = "torchscriptfunction"
    directivetype = "function"
    priority = FunctionDocumenter.priority + 1

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return type(member).__name__ == "ScriptFunction" and hasattr(
            member, "_torchdynamo_inline"
        )

    def import_object(self, raiseerror=False):
        ret = super().import_object(raiseerror)
        if ret and type(self.object).__name__ == "ScriptFunction":
            self.object = self.object._torchdynamo_inline
        return ret


def _defining_class_docstring_owner(name):
    """Return the class whose docstring autodoc inherits for method ``name``, if any."""
    module_name, _, rest = name.rpartition(".")
    parts = []
    while module_name:
        try:
            obj = importlib.import_module(module_name)
            break
        except ImportError:
            module_name, _, part = module_name.rpartition(".")
            parts.insert(0, part)
    else:
        return None
    for part in parts:
        obj = getattr(obj, part, None)
    if not isinstance(obj, type):
        return None
    for base in obj.__mro__:
        attr = vars(base).get(rest)
        if getattr(getattr(attr, "__func__", attr), "__doc__", None):
            return base
    return None


def drop_foreign_inherited_docstrings(app, what, name, obj, options, lines):
    """Drop docstrings that pymomentum classes inherit from PyTorch or ``object``.

    Text such as ``torch.nn.Module.forward``'s "Should be overridden by all
    subclasses" describes the base class, not the pymomentum override.
    """
    if what != "method" or getattr(obj, "__doc__", None):
        return
    owner = _defining_class_docstring_owner(name)
    if owner is not None and not owner.__module__.startswith("pymomentum"):
        lines.clear()


_GOOGLE_SECTION = re.compile(r"^(Args|Returns|Raises|Yields):\s*$")
_NAPOLEON_CONFIG = NapoleonConfig(napoleon_use_rtype=False)


def convert_google_docstrings(app, what, name, obj, options, lines):
    """Render Google-style (``Args:``/``Returns:``) docstrings as Sphinx fields.

    Only functions, methods and classes whose docstrings use Google section headers
    are converted, so the reST-style docstrings used elsewhere are left untouched.
    """
    if what not in ("function", "method", "class", "torchscriptfunction"):
        return
    if not any(_GOOGLE_SECTION.match(line) for line in lines):
        return
    lines[:] = GoogleDocstring(
        lines, _NAPOLEON_CONFIG, app, what, name, obj, options
    ).lines()


def setup(app):
    app.add_autodocumenter(TorchScriptFunctionDocumenter)
    app.connect("autodoc-skip-member", skip_pycapsule)
    app.connect("autodoc-process-signature", qualify_geometry_types)
    app.connect("autodoc-process-docstring", drop_foreign_inherited_docstrings)
    app.connect("autodoc-process-docstring", convert_google_docstrings)
