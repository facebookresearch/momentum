# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest

import pymomentum.quaternion as quaternion
import torch


def generateRandomQuats(sz: int) -> torch.Tensor:
    return quaternion.normalize(
        torch.normal(
            mean=0,
            std=4,
            size=(sz, 4),
            dtype=torch.float64,
            requires_grad=True,
        )
    )


class TestQuaternion(unittest.TestCase):
    def test_euler_conversion(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nBatch = 6
        nMat = 5
        euler1 = torch.normal(
            mean=0,
            std=4,
            size=(nBatch, nMat, 3),
            dtype=torch.float64,
            requires_grad=True,
        )
        quats1 = quaternion.euler_xyz_to_quaternion(euler1)
        euler2 = quaternion.quaternion_to_xyz_euler(quats1)
        quats2 = quaternion.euler_xyz_to_quaternion(euler2)

        quaternion.inverse(quats1)

        self.assertLess(
            torch.norm(
                quaternion.to_rotation_matrix(quats1)
                - quaternion.to_rotation_matrix(quats2)
            ),
            1e-4,
            "Expected rotation difference to be small.",
        )

        # check corner case conversion
        q_cornercase = torch.FloatTensor([0.4832, -0.5161, 0.4835, 0.5162])
        euler_cornercase = quaternion.quaternion_to_xyz_euler(q_cornercase)
        self.assertFalse(torch.isnan(euler_cornercase).any())

        # check the gradients:
        torch.autograd.gradcheck(
            quaternion.euler_xyz_to_quaternion,
            [euler1],
            raise_exception=True,
        )

        torch.autograd.gradcheck(
            quaternion.quaternion_to_xyz_euler,
            [quats1],
            raise_exception=True,
        )

    def test_matrix_to_quaternion(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nBatch = 6
        quats = generateRandomQuats(nBatch)
        mats = quaternion.to_rotation_matrix(quats)
        quats2 = quaternion.from_rotation_matrix(mats)
        mats2 = quaternion.to_rotation_matrix(quats2)
        diff = torch.minimum(
            (quats - quats).norm(dim=-1), (quats + quats2).norm(dim=-1)
        )
        self.assertLess(
            torch.norm(diff), 1e-4, "Expected quaternions to match (up to sign)"
        )
        self.assertLess(
            torch.norm(mats2 - mats), 1e-4, "Expected quaternions to match (up to sign)"
        )

    def test_multiply(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nMat = 5
        q1 = generateRandomQuats(nMat)
        q2 = generateRandomQuats(nMat)
        q12 = quaternion.multiply(q1, q2)

        m1 = quaternion.to_rotation_matrix(q1)
        m2 = quaternion.to_rotation_matrix(q2)
        m12 = torch.bmm(m1, m2)

        self.assertLess(
            torch.norm(quaternion.to_rotation_matrix(q12) - m12),
            1e-4,
            "Expected rotation difference to be small.",
        )

    def test_identity(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nMat = 5
        q1 = generateRandomQuats(nMat)
        ident = quaternion.identity().unsqueeze(0).expand_as(q1).type_as(q1)

        self.assertLess(
            torch.norm(quaternion.multiply(q1, ident) - q1),
            1e-4,
            "Identity on the right",
        )

        self.assertLess(
            torch.norm(quaternion.multiply(ident, q1) - q1),
            1e-4,
            "Identity on the right",
        )

    def test_rotate_vector(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nMat = 5
        q = generateRandomQuats(nMat)

        vec = torch.normal(
            mean=0,
            std=4,
            size=(nMat, 3),
            dtype=torch.float64,
            requires_grad=True,
        )

        rotated1 = quaternion.rotate_vector(q, vec)
        rotated2 = torch.bmm(
            quaternion.to_rotation_matrix(q), vec.unsqueeze(-1)
        ).squeeze(-1)

        self.assertLess(
            torch.norm(rotated1 - rotated2),
            1e-4,
            "Matrix rotation should match quaternion rotation",
        )

    def test_blend_same(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nMat = 5
        q = generateRandomQuats(nMat)
        q_dup = q.unsqueeze(1).expand(-1, 5, -1)
        q_blend = quaternion.blend(q_dup)

        self.assertLess(
            torch.norm(
                quaternion.to_rotation_matrix(q_blend)
                - quaternion.to_rotation_matrix(q)
            ),
            1e-4,
            "quaternion blending of the same quaternion should be identity.",
        )

    def test_blend_euler(self) -> None:
        unit_x = torch.tensor([0.25 * math.pi, 0, 0]).unsqueeze(0)

        # Blend of two single-axis rotations should be the midpoint:
        quats1 = quaternion.euler_xyz_to_quaternion(0.3 * math.pi * unit_x)
        quats2 = quaternion.euler_xyz_to_quaternion(0.5 * math.pi * unit_x)
        q_blend = quaternion.blend(torch.cat([quats1, quats2], 0))
        euler_blend = quaternion.euler_xyz_to_quaternion(0.4 * math.pi * unit_x)

        self.assertLess(
            torch.norm(
                quaternion.to_rotation_matrix(q_blend)
                - quaternion.to_rotation_matrix(euler_blend)
            ),
            1e-4,
            "quaternion blending of single-axis Euler rotation should be the midway rotation.",
        )

        weights = torch.tensor([0.75, 0.25])
        q_blend_weighted = quaternion.blend(torch.cat([quats1, quats2], 0), weights)
        euler_blend_weighted = quaternion.euler_xyz_to_quaternion(
            (0.3 * weights[0] + 0.5 * weights[1]) * math.pi * unit_x
        )
        self.assertLess(
            torch.norm(
                quaternion.to_rotation_matrix(q_blend_weighted)
                - quaternion.to_rotation_matrix(euler_blend_weighted)
            ),
            1e-2,
            "quaternion blending of single-axis Euler rotation should be the midway rotation.",
        )

    def test_from_two_vectors(self) -> None:
        # Test with two random vectors
        n_batch = 5
        v1 = torch.nn.functional.normalize(torch.randn(n_batch, 3), dim=-1)
        v2 = torch.nn.functional.normalize(torch.randn(n_batch, 3), dim=-1)

        q = quaternion.from_two_vectors(v1, v2)
        rotated_v1 = quaternion.rotate_vector(q, v1)

        self.assertTrue(torch.allclose(rotated_v1, v2, rtol=1e-4, atol=1e-6))

    def test_from_two_vectors_opposite(self) -> None:
        # Test with two opposite vectors
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([-1.0, 0.0, 0.0])

        q = quaternion.from_two_vectors(v1, v2)
        rotated_v1 = quaternion.rotate_vector(q, v1)

        self.assertTrue(torch.allclose(rotated_v1, v2))

    def test_from_two_vectors_parallel(self) -> None:
        # Test with two parallel vectors
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([2.0, 0.0, 0.0])

        q = quaternion.from_two_vectors(v1, v2)
        rotated_v1 = quaternion.rotate_vector(q, v1)

        self.assertTrue(torch.allclose(rotated_v1, v1))

    def test_check_and_normalize_weights_squeeze(self) -> None:
        # Test case where weights.dim() == quaternions.dim() (triggers squeeze)
        quaternions = torch.randn(2, 3, 4)  # Shape: (2, 3, 4)
        weights = torch.ones(2, 3, 1)  # Shape: (2, 3, 1) - same dim as quaternions

        # This should trigger the squeeze operation
        normalized_weights = quaternion.check_and_normalize_weights(
            quaternions, weights
        )

        # Verify the weights were properly normalized
        self.assertEqual(normalized_weights.shape, (2, 3))
        self.assertTrue(torch.allclose(normalized_weights.sum(dim=-1), torch.ones(2)))

    def test_check_and_normalize_weights_dim_mismatch(self) -> None:
        # Test case where weights.dim() + 1 != quaternions.dim()
        quaternions = torch.randn(2, 3, 4)  # Shape: (2, 3, 4) - 3 dimensions
        weights = torch.ones(2)  # Shape: (2,) - 1 dimension, but 1 + 1 != 3

        with self.assertRaises(ValueError) as context:
            quaternion.check_and_normalize_weights(quaternions, weights)

        self.assertIn(
            "Expected weights vector to match quaternion vector", str(context.exception)
        )
        self.assertIn("weights=torch.Size([2])", str(context.exception))
        self.assertIn("quaternions=torch.Size([2, 3, 4])", str(context.exception))

    def test_check_and_normalize_weights_size_mismatch(self) -> None:
        # Test case where weights.size(i) != quaternions.size(i) for some dimension i
        quaternions = torch.randn(2, 3, 4)  # Shape: (2, 3, 4)
        weights = torch.ones(2, 5)  # Shape: (2, 5) - first dim matches, second doesn't

        with self.assertRaises(ValueError) as context:
            quaternion.check_and_normalize_weights(quaternions, weights)

        self.assertIn(
            "Expected weights vector to match quaternion vector", str(context.exception)
        )
        self.assertIn("weights=torch.Size([2, 5])", str(context.exception))
        self.assertIn("quaternions=torch.Size([2, 3, 4])", str(context.exception))


if __name__ == "__main__":
    unittest.main()
