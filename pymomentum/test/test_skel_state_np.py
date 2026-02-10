# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import pymomentum.geometry as pym_geometry
import pymomentum.quaternion_np as pym_quaternion
import pymomentum.skel_state_np as pym_skel_state


def generate_skel_state_components(
    sz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    trans = rng.normal(loc=0, scale=4, size=(sz, 3))
    rot = pym_quaternion.normalize(rng.normal(loc=0, scale=4, size=(sz, 4)))
    scale = rng.random(size=(sz, 1))
    return (trans, rot, scale)


def generate_random_skel_state(sz: int) -> np.ndarray:
    (trans, rot, scale) = generate_skel_state_components(sz)
    return np.concatenate([trans, rot, scale], axis=-1)


def init_sim3_like_skel_state(
    batch_size: int, nr_joints: int, seed: int = 0
) -> np.ndarray:
    """Initialize skel_state similar to how sim3 tests initialize their state"""
    rng = np.random.default_rng(seed)
    t = rng.standard_normal((batch_size, nr_joints, 3)).astype(np.float32)
    q = rng.standard_normal((batch_size, nr_joints, 4)).astype(np.float32)
    s = rng.standard_normal((batch_size, nr_joints, 1)).astype(np.float32)

    q = pym_quaternion.normalize(q)
    s = np.minimum(np.exp2(s), 2.0)
    return np.concatenate([t, q, s], axis=-1)


class TestSkelStateNumpy(unittest.TestCase):
    def test_skel_state_to_transforms(self) -> None:
        character = pym_geometry.create_test_character()
        nBatch = 2
        modelParams = 0.2 * np.ones(
            (nBatch, character.parameter_transform.size),
            dtype=np.float64,
        )
        joint_params = character.parameter_transform.apply(modelParams)
        skel_state = pym_geometry.joint_parameters_to_skeleton_state(
            character, joint_params
        )
        skel_state_d = pym_geometry.joint_parameters_to_skeleton_state(
            character, joint_params.astype(np.float64)
        )

        # Verify to_matrix produces valid results
        matrix = pym_skel_state.to_matrix(skel_state_d)
        self.assertEqual(matrix.shape[-2:], (4, 4))

        self.assertTrue(np.allclose(skel_state, skel_state_d, atol=1e-3))

    def test_multiply(self) -> None:
        nMats = 6
        s1 = generate_random_skel_state(nMats)
        s2 = generate_random_skel_state(nMats)

        s12 = pym_skel_state.multiply(s1, s2)
        m12 = pym_skel_state.to_matrix(s12)

        m1 = pym_skel_state.to_matrix(s1)
        m2 = pym_skel_state.to_matrix(s2)

        self.assertLess(
            np.linalg.norm(np.einsum("bij,bjk->bik", m1, m2) - m12),
            1e-4,
            "Multiplication is correct",
        )

    def test_inverse(self) -> None:
        nMats = 6
        s = generate_random_skel_state(nMats)
        s_inv = pym_skel_state.inverse(s)

        m = pym_skel_state.to_matrix(s)
        m_inv = pym_skel_state.to_matrix(s_inv)
        m_inv2 = np.linalg.inv(m)

        self.assertLess(
            np.linalg.norm(m_inv - m_inv2),
            1e-4,
            "Inverse is correct",
        )

    def test_transform_points(self) -> None:
        rng = np.random.default_rng(0)
        nMats = 6
        s = generate_random_skel_state(nMats)
        m = pym_skel_state.to_matrix(s)
        pts = rng.random(size=(nMats, 3))
        hpts = np.concatenate([pts, np.ones((nMats, 1))], axis=-1)
        transformed1 = np.einsum("bij,bj->bi", m, hpts)[:, 0:3]
        transformed2 = pym_skel_state.transform_points(s, pts)

        self.assertLess(
            np.linalg.norm(transformed1 - transformed2),
            1e-4,
            "Transform points is correct",
        )

    def test_construct(self) -> None:
        nMats = 4
        (trans, rot, scale) = generate_skel_state_components(nMats)

        s1 = np.concatenate([trans, rot, scale], axis=-1)
        s2 = pym_skel_state.multiply(
            pym_skel_state.from_translation(trans),
            pym_skel_state.multiply(
                pym_skel_state.from_quaternion(rot), pym_skel_state.from_scale(scale)
            ),
        )

        self.assertLess(
            np.linalg.norm(s1 - s2),
            1e-4,
            "constructing from components",
        )

    def test_matrix_to_skel_state(self) -> None:
        nMats = 6
        s1 = generate_random_skel_state(nMats)
        m1 = pym_skel_state.to_matrix(s1)
        s2 = pym_skel_state.from_matrix(m1)
        m2 = pym_skel_state.to_matrix(s2)
        self.assertLess(
            np.linalg.norm(m1 - m2),
            1e-4,
            "matrix-to-skel-state conversion: matrices should always match",
        )

        # Translation and scale should match.  Rotation might have its sign flipped:
        self.assertLess(
            np.linalg.norm(s1[..., 0:3] - s2[..., 0:3]),
            1e-4,
            "matrix-to-skel-state conversion: translations should match",
        )
        self.assertLess(
            np.linalg.norm(s1[..., 7] - s2[..., 7]),
            1e-4,
            "matrix-to-skel-state conversion: scales should match",
        )

    def test_skel_state_blending(self) -> None:
        t1 = np.array([1.0, 0.0, 0.0])
        t2 = np.array([0.0, 1.0, 0.0])

        s1 = pym_skel_state.from_translation(t1)
        s2 = pym_skel_state.from_translation(t2)
        s_test = pym_skel_state.from_translation(np.array([0.25, 0.75, 0.0]))

        weights = np.array([0.25, 0.75])
        s_blended = pym_skel_state.blend(np.stack([s1, s2], axis=0), weights)

        self.assertLess(
            np.linalg.norm(s_blended - s_test),
            1e-4,
            "skel-state blending",
        )

    def test_skel_state_splitting(self) -> None:
        t1 = np.array([1.0, 0.0, 0.0])
        skel_state1 = pym_skel_state.from_translation(t1)
        t2, _, _ = pym_skel_state.split(skel_state1)
        self.assertTrue(np.allclose(t1, t2))

        r3 = np.array([0.0, 1.0, 0.0, 0.0])
        skel_state3 = pym_skel_state.from_quaternion(r3)
        _, r4, _ = pym_skel_state.split(skel_state3)
        self.assertTrue(np.allclose(r3, r4))

        s5 = np.array([2.0])
        skel_state5 = pym_skel_state.from_scale(s5)
        _, _, s6 = pym_skel_state.split(skel_state5)
        self.assertTrue(np.allclose(s5, s6))

    def test_check_invalid_shape(self) -> None:
        rng = np.random.default_rng(0)
        invalid_skel_state = rng.standard_normal((2, 7))

        with self.assertRaises(ValueError) as context:
            pym_skel_state.check(invalid_skel_state)

        self.assertIn(
            "Expected skeleton state to have last dimension 8", str(context.exception)
        )

    def test_identity_no_size(self) -> None:
        identity_state = pym_skel_state.identity()

        self.assertEqual(identity_state.shape, (8,))
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        self.assertTrue(np.allclose(identity_state, expected))

    def test_from_matrix_invalid_dimensions(self) -> None:
        rng = np.random.default_rng(0)
        invalid_matrix = rng.standard_normal((3, 3))

        with self.assertRaises(ValueError) as context:
            pym_skel_state.from_matrix(invalid_matrix)

        self.assertIn("Expected an array of 4x4 matrices", str(context.exception))

    def test_from_matrix_2d_case(self) -> None:
        matrix_2d = np.eye(4)
        matrix_2d[:3, 3] = np.array([1.0, 2.0, 3.0])

        skel_state = pym_skel_state.from_matrix(matrix_2d)

        self.assertEqual(skel_state.shape, (8,))
        self.assertTrue(np.allclose(skel_state[:3], np.array([1.0, 2.0, 3.0])))

    def test_transform_points_invalid_dimensions(self) -> None:
        skel_state = pym_skel_state.identity()

        invalid_points = np.random.default_rng(0).standard_normal((5, 2))

        with self.assertRaises(ValueError) as context:
            pym_skel_state.transform_points(skel_state, invalid_points)

        self.assertIn(
            "Points tensor should have last dimension 3", str(context.exception)
        )

        scalar_points = np.array(1.0)

        with self.assertRaises(ValueError) as context:
            pym_skel_state.transform_points(skel_state, scalar_points)

        self.assertIn(
            "Points tensor should have last dimension 3", str(context.exception)
        )

    def test_slerp_extremes(self) -> None:
        nMats = 6
        s0 = generate_random_skel_state(nMats)
        s1 = generate_random_skel_state(nMats)

        # Test slerp at t=0 should return s0
        s_slerp_0 = pym_skel_state.slerp(s0, s1, np.array([0.0]))
        self.assertTrue(
            np.allclose(
                pym_skel_state.to_matrix(s_slerp_0),
                pym_skel_state.to_matrix(s0),
                atol=1e-4,
            ),
            "Expected slerp result to be s0 when t=0.",
        )

        # Test slerp at t=1 should return s1
        s_slerp_1 = pym_skel_state.slerp(s0, s1, np.array([1.0]))
        self.assertTrue(
            np.allclose(
                pym_skel_state.to_matrix(s_slerp_1),
                pym_skel_state.to_matrix(s1),
                atol=1e-4,
            ),
            "Expected slerp result to be s1 when t=1.",
        )

    def test_slerp_midpoint(self) -> None:
        nMats = 6
        s0 = generate_random_skel_state(nMats)
        s1 = generate_random_skel_state(nMats)

        # Perform slerp at t=0.5
        s_slerp_mid = pym_skel_state.slerp(s0, s1, np.array([0.5]))

        # Extract components
        t0, _, s0_scale = pym_skel_state.split(s0)
        t1, _, s1_scale = pym_skel_state.split(s1)
        t_mid, _, s_mid_scale = pym_skel_state.split(s_slerp_mid)

        # Check if translation and scale are in the middle
        self.assertTrue(
            np.allclose(t_mid, (t0 + t1) / 2, atol=1e-4),
            "Expected translation to be the midpoint when t=0.5.",
        )
        self.assertTrue(
            np.allclose(s_mid_scale, (s0_scale + s1_scale) / 2, atol=1e-4),
            "Expected scale to be the midpoint when t=0.5.",
        )

    def test_bulk_multiplication(self) -> None:
        """Test multiplication using the same approach as sim3 tests"""
        state1 = init_sim3_like_skel_state(1024, 159, seed=1002389)
        state2 = init_sim3_like_skel_state(1024, 159, seed=1002390)
        state = pym_skel_state.multiply(state1, state2)

        t1, q1, s1 = pym_skel_state.split(state1)
        t2, q2, s2 = pym_skel_state.split(state2)
        r1 = pym_quaternion.to_rotation_matrix(q1)
        r2 = pym_quaternion.to_rotation_matrix(q2)

        t_expected = np.einsum("bjxy,bjy->bjx", r1, t2) * s1 + t1
        r_expected = np.einsum("bjxy,bjyz->bjxz", r1, r2)
        s_expected = s1 * s2

        tt, qq, ss = pym_skel_state.split(state)
        rr = pym_quaternion.to_rotation_matrix(qq)

        self.assertTrue(np.allclose(t_expected, tt, atol=1e-5, rtol=1e-5))
        self.assertTrue(np.allclose(r_expected, rr, atol=1e-5, rtol=1e-5))
        self.assertTrue(np.allclose(s_expected, ss, atol=1e-5, rtol=1e-5))

    def test_bulk_points_transformation(self) -> None:
        """Test point transformation using the same approach as sim3 tests"""
        rng = np.random.default_rng(1002389)
        state = init_sim3_like_skel_state(1024, 159, seed=1002389)
        points = rng.standard_normal((1024, 159, 3)).astype(np.float32)

        p1 = pym_skel_state.transform_points(state, points)

        tt, qq, ss = pym_skel_state.split(state)
        rr = pym_quaternion.to_rotation_matrix(qq)
        p2 = ss * np.einsum("bjxy,bjy->bjx", rr, points) + tt

        self.assertTrue(np.allclose(p1, p2, atol=1e-5, rtol=1e-5))

    def test_bulk_inverse(self) -> None:
        """Test inverse using the same approach as sim3 tests"""
        state = init_sim3_like_skel_state(1024, 159, seed=1002389)
        state_inv = pym_skel_state.inverse(state)

        state_expected = np.zeros_like(state)
        state_expected[..., 6] = 1  # quaternion w component = 1
        state_expected[..., 7] = 1  # scale component = 1

        state2 = pym_skel_state.multiply(state, state_inv)
        self.assertTrue(np.allclose(state2, state_expected, atol=1e-5, rtol=1e-5))

        state3 = pym_skel_state.multiply(state_inv, state)
        self.assertTrue(np.allclose(state3, state_expected, atol=1e-5, rtol=1e-5))

    def test_multiply_broadcasting(self) -> None:
        """Test that multiplication works correctly with broadcasting for different array dimensions"""
        state1 = generate_random_skel_state(6).reshape(3, 2, 8)
        state2 = generate_random_skel_state(2).reshape(1, 2, 8)

        result_broadcast = pym_skel_state.multiply(state1, state2)

        state2_expanded = np.broadcast_to(state2, (3, 2, 8)).copy()
        result_manual = pym_skel_state.multiply(state1, state2_expanded)

        self.assertTrue(
            np.allclose(result_broadcast, result_manual, atol=1e-6),
            "Broadcasting multiplication should match manually expanded multiplication",
        )

        # Test with different broadcasting patterns
        state_narrow = generate_random_skel_state(2).reshape(2, 1, 8)
        state_wide = generate_random_skel_state(6).reshape(2, 3, 8)

        result_narrow_broadcast = pym_skel_state.multiply(state_narrow, state_wide)
        narrow_expanded = np.broadcast_to(state_narrow, (2, 3, 8)).copy()
        result_narrow_manual = pym_skel_state.multiply(narrow_expanded, state_wide)

        self.assertTrue(
            np.allclose(result_narrow_broadcast, result_narrow_manual, atol=1e-6),
            "Narrow state broadcasting should work correctly",
        )

        # Test reverse broadcasting
        result_reverse_broadcast = pym_skel_state.multiply(state_wide, state_narrow)
        result_reverse_manual = pym_skel_state.multiply(state_wide, narrow_expanded)

        self.assertTrue(
            np.allclose(result_reverse_broadcast, result_reverse_manual, atol=1e-6),
            "Reverse broadcasting should work correctly",
        )

        # Verify shapes
        self.assertEqual(result_broadcast.shape, (3, 2, 8))
        self.assertEqual(result_narrow_broadcast.shape, (2, 3, 8))
        self.assertEqual(result_reverse_broadcast.shape, (2, 3, 8))

        # Verify transformation properties via matrices
        for i in range(3):
            for j in range(2):
                m1 = pym_skel_state.to_matrix(state1[i, j])
                m2 = pym_skel_state.to_matrix(state2[0, j])
                expected_matrix = m1 @ m2
                actual_matrix = pym_skel_state.to_matrix(result_broadcast[i, j])

                self.assertTrue(
                    np.allclose(expected_matrix, actual_matrix, atol=1e-5),
                    f"Matrix multiplication should match for element [{i}, {j}]",
                )


if __name__ == "__main__":
    unittest.main()
