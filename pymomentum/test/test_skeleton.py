# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from pymomentum.geometry import Character, create_test_character


class TestSkeleton(unittest.TestCase):
    def test_get_skeleton_offsets_and_prerotations(self) -> None:
        char: Character = create_test_character()
        offsets = char.skeleton.offsets
        pre_rotations = char.skeleton.pre_rotations
        num_joints = len(char.skeleton.joint_names)
        self.assertEqual(tuple(offsets.shape), (num_joints, 3))
        self.assertEqual(tuple(pre_rotations.shape), (num_joints, 4))


if __name__ == "__main__":
    unittest.main()
