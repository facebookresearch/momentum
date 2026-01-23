# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import torch
from pymomentum.geometry import (
    Character,
    create_test_character,
    ParameterTransform,
    uniform_random_to_model_parameters,
)


class TestParameterTransform(unittest.TestCase):
    def test_get_transform(self) -> None:
        character: Character = create_test_character()
        transform = character.parameter_transform.transform
        self.assertEqual(tuple(transform.shape), (3 * 7, 10))

        np.random.seed(42)
        model_params = uniform_random_to_model_parameters(
            character, np.random.rand(1, 10).astype(np.float32)
        ).squeeze()
        joint_params_1 = torch.from_numpy(
            character.parameter_transform.apply(model_params[None, :])
        ).flatten()
        joint_params_2 = torch.matmul(transform, torch.from_numpy(model_params))
        self.assertTrue(torch.allclose(joint_params_1, joint_params_2))

    def test_parameter_transform_round_trip(self) -> None:
        """Test that converting parameter transform to dense matrix and back is lossless."""
        # Create a test character with parameter transform
        character: Character = create_test_character()
        original_pt = character.parameter_transform

        # Get the transform as a dense matrix
        transform_matrix = original_pt.transform
        transform_numpy = transform_matrix.numpy()

        # Create a new parameter transform from the dense matrix
        new_pt = ParameterTransform(
            names=original_pt.names,
            skeleton=character.skeleton,
            transform=transform_numpy,
        )

        # Verify the transform produces identical results
        np.random.seed(42)
        model_params = uniform_random_to_model_parameters(
            character, np.random.rand(5, 10).astype(np.float32)
        )

        joint_params_original = torch.from_numpy(original_pt.apply(model_params))
        joint_params_new = torch.from_numpy(new_pt.apply(model_params))

        self.assertTrue(
            torch.allclose(joint_params_original, joint_params_new, atol=1e-6)
        )

        # Verify parameter names are preserved
        self.assertEqual(original_pt.names, new_pt.names)

        # Verify size is preserved
        self.assertEqual(original_pt.size, new_pt.size)


if __name__ == "__main__":
    unittest.main()
