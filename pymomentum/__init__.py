# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import warnings


def _is_submodule_import():
    """Check if we're being imported as 'import pymomentum.submodule' vs 'import pymomentum'."""
    # Walk up the call stack to find import machinery frames
    frame = sys._getframe(1)
    while frame:
        # Look for the import machinery in importlib
        code_file = frame.f_code.co_filename
        if "importlib" in code_file and (
            "_bootstrap" in code_file or "_load" in code_file
        ):
            # Check the 'name' variable which contains the module being imported
            if "name" in frame.f_locals:
                name = frame.f_locals["name"]
                # If importing 'pymomentum.something', it's a submodule import
                if name.startswith("pymomentum."):
                    return True
        frame = frame.f_back
    return False


# Only warn for bare 'import pymomentum', not 'import pymomentum.geometry'
if not _is_submodule_import():
    warnings.warn(
        "\n"
        "=" * 70 + "\n"
        "WARNING: Direct import of 'pymomentum' is not supported.\n\n"
        "PyMomentum no longer provides a populated __init__.py to avoid\n"
        "heavy loading times. Please import individual modules instead:\n\n"
        "  import pymomentum.geometry\n"
        "  import pymomentum.solver\n"
        "  import pymomentum.marker_tracking\n"
        "=" * 70,
        UserWarning,
        stacklevel=2,
    )
