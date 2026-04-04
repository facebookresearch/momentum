# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Pytest configuration for pymomentum tests.

This file configures pytest to skip tests when their dependencies are unavailable.
For Meta internal builds (Buck/TestX), the load_tests() protocol in individual
test files handles conditional test loading. This conftest.py handles the same
logic for pytest (used in GitHub CI).
"""

# pyre-strict

import os

# Exclude test files whose dependencies are not available.
# collect_ignore runs before pytest imports the test modules, preventing
# ImportError during collection (e.g. when pymomentum.io_usd is not built).
collect_ignore: list[str] = []

try:
    import pymomentum.io_usd as _io_usd  # noqa: F401

    usd_available = True
except ImportError:
    usd_available = False

if not usd_available:
    collect_ignore.append(os.path.join(os.path.dirname(__file__), "test_usd.py"))
