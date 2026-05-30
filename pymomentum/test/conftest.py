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
    import torch

    cuda_available = torch.cuda.is_available()
except ImportError:
    cuda_available = False

try:
    import pymomentum.io_usd as _io_usd  # noqa: F401  # @manual=:io_usd

    usd_available = True
except ImportError:
    usd_available = False

if not usd_available:
    collect_ignore.append(os.path.join(os.path.dirname(__file__), "test_usd.py"))

if not cuda_available:
    for test_file in (
        "test_triton_fk_backend.py",
        "test_triton_quaternion_backend.py",
        "test_triton_skel_state_backend.py",
        "test_triton_skinning_backend.py",
    ):
        collect_ignore.append(os.path.join(os.path.dirname(__file__), test_file))
