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


if __name__ == "__main__":
    unittest.main()
