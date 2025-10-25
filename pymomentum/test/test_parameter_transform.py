# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

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

        torch.manual_seed(42)
        model_params = uniform_random_to_model_parameters(
            character, torch.rand(1, 10)
        ).squeeze()
        joint_params_1 = character.parameter_transform.apply(model_params)
        joint_params_2 = torch.matmul(transform, model_params)
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
        torch.manual_seed(42)
        model_params = uniform_random_to_model_parameters(character, torch.rand(5, 10))

        joint_params_original = original_pt.apply(model_params)
        joint_params_new = new_pt.apply(model_params)

        self.assertTrue(
            torch.allclose(joint_params_original, joint_params_new, atol=1e-6)
        )

        # Verify parameter names are preserved
        self.assertEqual(original_pt.names, new_pt.names)

        # Verify size is preserved
        self.assertEqual(original_pt.size, new_pt.size)


if __name__ == "__main__":
    unittest.main()
