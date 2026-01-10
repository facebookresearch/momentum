# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Consistency tests between pymomentum.geometry and pymomentum.diff_geometry.

This file contains tests that verify that the numpy-based implementations in
pymomentum.geometry produce the same results as the torch-based implementations
in pymomentum.diff_geometry.
"""

import unittest

import numpy as np
import pymomentum.diff_geometry as pym_diff_geometry
import pymomentum.geometry as pym_geometry
import torch


class TestGeometryDiffGeometryConsistency(unittest.TestCase):
    """Tests verifying consistency between geometry and diff_geometry modules."""
