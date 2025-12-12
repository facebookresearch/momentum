# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

try:
    import pymomentum.axel as axel
    import pymomentum.geometry as geom
    import pymomentum.marker_tracking as marker_tracking
    import pymomentum.solver as solver
    import pymomentum.solver2 as solver2

    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}", file=sys.stderr)
    sys.exit(1)
