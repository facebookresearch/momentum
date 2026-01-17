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

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001 - required by pytest hook signature
    items: list[pytest.Item],
) -> None:
    """
    Skip tests at collection time when their dependencies are unavailable.

    This is the pytest equivalent of load_tests() returning an empty TestSuite.
    It prevents tests from running (and failing) when required dependencies
    are not available in the build environment.
    """
    # Check USD availability once at collection time
    try:
        import pymomentum.geometry as pym_geometry

        usd_available = pym_geometry.is_usd_available()
    except (ImportError, AttributeError):
        usd_available = False

    skip_usd = pytest.mark.skip(reason="USD support not available in this build")

    for item in items:
        # Skip USD tests when USD is not available
        if "test_usd" in item.nodeid and not usd_available:
            item.add_marker(skip_usd)
