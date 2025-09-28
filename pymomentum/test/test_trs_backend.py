# pyre-strict
import unittest

import torch as th

th.autograd.set_detect_anomaly(True)

import pymomentum.geometry as pym_geometry
import pymomentum.quaternion as pym_quaternion
import pymomentum.trs as pym_trs
from pymomentum.backend.trs_backend import (
    fk_from_local_state,
    get_local_state_from_joint_params,
    skinning,
)
from pymomentum.backend.utils import (
    calc_fk_prefix_multiplication_indices,
    create_lbs_from_trinity,
    flatten_skinning_weights_and_indices,
)
from torch.nn import Parameter as P


class TestTRSBackend(unittest.TestCase):
    """
    Test TRS backend functions against the ground truth pymomentum.geometry functions.

    This class compares the TRS backend implementations with the pymomentum.geometry
    equivalents by converting skeleton states to TRS format using trs.from_skeleton_state().
    """

    def assert_all_close(self, a: th.Tensor, b: th.Tensor, tol: float = 1e-5) -> None:
        mask = th.isclose(a, b, atol=tol, rtol=tol)
        if not th.all(mask):
            print(
                f"assert_all_close failed with {tol=} on joint",
                th.unique(th.where(~mask)[1].cpu()),
            )
        self.assertTrue(th.allclose(a, b, atol=tol, rtol=tol))

    def init_joint_params(self, batch_size: int, nr_joints: int) -> th.Tensor:
        """Initialize random joint parameters for testing."""
        joint_params = th.randn((batch_size, nr_joints, 7), dtype=th.float32)
        # Keep reasonable scale values (prevent extreme scales)
        joint_params[:, :, 6] = joint_params[:, :, 6].clamp(-1, 1)
        return joint_params

    def test_get_local_state_from_joint_params(self) -> None:
        """Test get_local_state_from_joint_params against pymomentum.geometry.joint_parameters_to_local_skeleton_state."""
        th.manual_seed(42)
        lbs = create_lbs_from_trinity(device="cpu")
        batch_size = 8

        # Generate random joint parameters
        joint_params = self.init_joint_params(batch_size, lbs.nr_joints)

        # Convert joint_rotation from quaternions to rotation matrices if needed
        if lbs.joint_rotation.shape[-1] == 4:
            # joint_rotation is quaternions, convert to rotation matrices
            joint_rotation_matrices = pym_quaternion.to_rotation_matrix(
                lbs.joint_rotation
            )
        else:
            # joint_rotation is already rotation matrices
            joint_rotation_matrices = lbs.joint_rotation

        # TRS backend version
        local_state_t, local_state_r, local_state_s = get_local_state_from_joint_params(
            joint_params, lbs.joint_offset, joint_rotation_matrices
        )

        # PyMomentum geometry version (ground truth)
        joint_params_flat = joint_params.view(batch_size, -1)
        local_skel_state_gt = pym_geometry.joint_parameters_to_local_skeleton_state(
            lbs.character, joint_params_flat
        )
        local_skel_state_gt = local_skel_state_gt.view(batch_size, lbs.nr_joints, 8)

        # Convert skeleton state to TRS
        local_state_t_gt, local_state_r_gt, local_state_s_gt = (
            pym_trs.from_skeleton_state(local_skel_state_gt)
        )

        # Compare results
        self.assert_all_close(local_state_t, local_state_t_gt, 1e-5)
        self.assert_all_close(local_state_r, local_state_r_gt, 1e-5)
        self.assert_all_close(local_state_s, local_state_s_gt, 1e-5)

    def test_fk_from_local_state(self) -> None:
        """Test fk_from_local_state against pymomentum.geometry.joint_parameters_to_skeleton_state."""
        th.manual_seed(42)
        lbs = create_lbs_from_trinity(device="cpu")
        prefix_mul_indices = calc_fk_prefix_multiplication_indices(lbs.joint_parents)
        batch_size = 8

        # Generate random joint parameters
        joint_params = self.init_joint_params(batch_size, lbs.nr_joints)

        # Convert joint_rotation from quaternions to rotation matrices if needed
        if lbs.joint_rotation.shape[-1] == 4:
            joint_rotation_matrices = pym_quaternion.to_rotation_matrix(
                lbs.joint_rotation
            )
        else:
            joint_rotation_matrices = lbs.joint_rotation

        # Get local state from joint params (using TRS backend)
        local_state_t, local_state_r, local_state_s = get_local_state_from_joint_params(
            joint_params, lbs.joint_offset, joint_rotation_matrices
        )

        # TRS backend FK
        global_state_t, global_state_r, global_state_s = fk_from_local_state(
            local_state_t, local_state_r, local_state_s, prefix_mul_indices
        )

        # PyMomentum geometry version (ground truth)
        joint_params_flat = joint_params.view(batch_size, -1)
        global_skel_state_gt = pym_geometry.joint_parameters_to_skeleton_state(
            lbs.character, joint_params_flat
        )
        global_skel_state_gt = global_skel_state_gt.view(batch_size, lbs.nr_joints, 8)

        # Convert skeleton state to TRS
        global_state_t_gt, global_state_r_gt, global_state_s_gt = (
            pym_trs.from_skeleton_state(global_skel_state_gt)
        )

        # Compare results
        self.assert_all_close(global_state_t, global_state_t_gt, 1e-4)
        self.assert_all_close(global_state_r, global_state_r_gt, 1e-4)
        self.assert_all_close(global_state_s, global_state_s_gt, 1e-4)

    def test_fk_from_local_state_gradients(self) -> None:
        """Test gradients of fk_from_local_state."""
        th.manual_seed(42)
        lbs = create_lbs_from_trinity(device="cpu")
        prefix_mul_indices = calc_fk_prefix_multiplication_indices(lbs.joint_parents)
        batch_size = 4

        # Generate random joint parameters with gradients
        joint_params = P(self.init_joint_params(batch_size, lbs.nr_joints))

        # Convert joint_rotation from quaternions to rotation matrices if needed
        if lbs.joint_rotation.shape[-1] == 4:
            joint_rotation_matrices = pym_quaternion.to_rotation_matrix(
                lbs.joint_rotation
            )
        else:
            joint_rotation_matrices = lbs.joint_rotation

        # Get local state from joint params (using TRS backend)
        local_state_t, local_state_r, local_state_s = get_local_state_from_joint_params(
            joint_params, lbs.joint_offset, joint_rotation_matrices
        )

        # TRS backend FK
        global_state_t, global_state_r, global_state_s = fk_from_local_state(
            local_state_t, local_state_r, local_state_s, prefix_mul_indices
        )

        # Generate random gradients
        grad_t = th.randn_like(global_state_t)
        grad_r = th.randn_like(global_state_r)
        grad_s = th.randn_like(global_state_s)

        # Compute gradients
        (grad_joint_params,) = th.autograd.grad(
            outputs=[global_state_t, global_state_r, global_state_s],
            inputs=[joint_params],
            grad_outputs=[grad_t, grad_r, grad_s],
        )

        # Verify gradients are computed and have correct shape
        self.assertEqual(grad_joint_params.shape, joint_params.shape)
        self.assertFalse(th.isnan(grad_joint_params).any())

    def test_skinning(self) -> None:
        """Test skinning function against pymomentum.geometry.Character.skin_points."""
        th.manual_seed(42)
        lbs = create_lbs_from_trinity(device="cpu")
        prefix_mul_indices = calc_fk_prefix_multiplication_indices(lbs.joint_parents)
        batch_size = 4

        # Generate random joint parameters
        joint_params = self.init_joint_params(batch_size, lbs.nr_joints)

        # Convert joint_rotation from quaternions to rotation matrices if needed
        if lbs.joint_rotation.shape[-1] == 4:
            joint_rotation_matrices = pym_quaternion.to_rotation_matrix(
                lbs.joint_rotation
            )
        else:
            joint_rotation_matrices = lbs.joint_rotation

        # Get global state using TRS backend
        local_state_t, local_state_r, local_state_s = get_local_state_from_joint_params(
            joint_params, lbs.joint_offset, joint_rotation_matrices
        )
        global_state_t, global_state_r, global_state_s = fk_from_local_state(
            local_state_t, local_state_r, local_state_s, prefix_mul_indices
        )

        # Prepare skinning data
        (
            skin_indices_flattened,
            skin_weights_flattened,
            vert_indices_flattened,
        ) = flatten_skinning_weights_and_indices(lbs.skin_weights, lbs.skin_indices)

        # TRS backend skinning
        template = lbs.mesh_vertices.unsqueeze(0).expand(batch_size, -1, -1)
        print(lbs.mesh_vertices[0])
        print(lbs.character.mesh.vertices[0])
        skinned_trs = skinning(
            template=template,
            t=global_state_t,
            r=global_state_r,
            s=global_state_s,
            t0=lbs.joint_state_t_zero,
            r0=lbs.joint_state_r_zero,
            skin_indices_flattened=skin_indices_flattened,
            skin_weights_flattened=skin_weights_flattened,
            vert_indices_flattened=vert_indices_flattened,
        )

        # PyMomentum geometry version (ground truth)
        joint_params_flat = joint_params.view(batch_size, -1)
        global_skel_state_gt = pym_geometry.joint_parameters_to_skeleton_state(
            lbs.character, joint_params_flat
        )

        skinned_gt = lbs.character.skin_points(global_skel_state_gt)

        # Compare results
        self.assert_all_close(skinned_trs, skinned_gt, 1e-3)

    def test_skinning_gradients(self) -> None:
        """Test gradients of skinning function."""
        th.manual_seed(42)
        lbs = create_lbs_from_trinity(device="cpu")
        prefix_mul_indices = calc_fk_prefix_multiplication_indices(lbs.joint_parents)
        batch_size = 2  # Smaller batch for gradient test

        # Generate random joint parameters with gradients
        joint_params = P(self.init_joint_params(batch_size, lbs.nr_joints))

        # Convert joint_rotation from quaternions to rotation matrices if needed
        if lbs.joint_rotation.shape[-1] == 4:
            joint_rotation_matrices = pym_quaternion.to_rotation_matrix(
                lbs.joint_rotation
            )
        else:
            joint_rotation_matrices = lbs.joint_rotation

        # Get global state using TRS backend
        local_state_t, local_state_r, local_state_s = get_local_state_from_joint_params(
            joint_params, lbs.joint_offset, joint_rotation_matrices
        )
        global_state_t, global_state_r, global_state_s = fk_from_local_state(
            local_state_t, local_state_r, local_state_s, prefix_mul_indices
        )

        # Prepare skinning data
        (
            skin_indices_flattened,
            skin_weights_flattened,
            vert_indices_flattened,
        ) = flatten_skinning_weights_and_indices(lbs.skin_weights, lbs.skin_indices)

        # TRS backend skinning
        template = P(lbs.mesh_vertices.unsqueeze(0).expand(batch_size, -1, -1))
        skinned_trs = skinning(
            template=template,
            t=global_state_t,
            r=global_state_r,
            s=global_state_s,
            t0=lbs.joint_state_t_zero,
            r0=lbs.joint_state_r_zero,
            skin_indices_flattened=skin_indices_flattened,
            skin_weights_flattened=skin_weights_flattened,
            vert_indices_flattened=vert_indices_flattened,
        )

        # Generate random gradient and compute gradients
        grad_skinned = th.randn_like(skinned_trs)

        (grad_joint_params, grad_template) = th.autograd.grad(
            outputs=[skinned_trs],
            inputs=[joint_params, template],
            grad_outputs=[grad_skinned],
        )

        # Verify gradients are computed and have correct shapes
        self.assertEqual(grad_joint_params.shape, joint_params.shape)
        self.assertEqual(grad_template.shape, template.shape)
        self.assertFalse(th.isnan(grad_joint_params).any())
        self.assertFalse(th.isnan(grad_template).any())

    def test_end_to_end_consistency(self) -> None:
        """Test end-to-end consistency: joint_params -> local_state -> global_state -> skinning."""
        th.manual_seed(123)
        lbs = create_lbs_from_trinity(device="cpu")
        prefix_mul_indices = calc_fk_prefix_multiplication_indices(lbs.joint_parents)
        batch_size = 6

        # Generate random joint parameters
        joint_params = self.init_joint_params(batch_size, lbs.nr_joints)

        # Convert joint_rotation from quaternions to rotation matrices if needed
        if lbs.joint_rotation.shape[-1] == 4:
            joint_rotation_matrices = pym_quaternion.to_rotation_matrix(
                lbs.joint_rotation
            )
        else:
            joint_rotation_matrices = lbs.joint_rotation

        # Full TRS backend pipeline
        local_state_t, local_state_r, local_state_s = get_local_state_from_joint_params(
            joint_params, lbs.joint_offset, joint_rotation_matrices
        )
        global_state_t, global_state_r, global_state_s = fk_from_local_state(
            local_state_t, local_state_r, local_state_s, prefix_mul_indices
        )

        (
            skin_indices_flattened,
            skin_weights_flattened,
            vert_indices_flattened,
        ) = flatten_skinning_weights_and_indices(lbs.skin_weights, lbs.skin_indices)

        template = lbs.mesh_vertices.unsqueeze(0).expand(batch_size, -1, -1)
        skinned_trs = skinning(
            template=template,
            t=global_state_t,
            r=global_state_r,
            s=global_state_s,
            t0=lbs.joint_state_t_zero,
            r0=lbs.joint_state_r_zero,
            skin_indices_flattened=skin_indices_flattened,
            skin_weights_flattened=skin_weights_flattened,
            vert_indices_flattened=vert_indices_flattened,
        )

        # PyMomentum geometry pipeline (ground truth)
        joint_params_flat = joint_params.view(batch_size, -1)
        global_skel_state_gt = pym_geometry.joint_parameters_to_skeleton_state(
            lbs.character, joint_params_flat
        )
        global_skel_state_gt = global_skel_state_gt.view(batch_size, lbs.nr_joints, 8)
        skinned_gt = lbs.character.skin_points(global_skel_state_gt)

        # Compare final results
        self.assert_all_close(skinned_trs, skinned_gt, 1e-3)

    def test_single_joint_case(self) -> None:
        """Test with a simple single joint case for easier debugging."""
        # Create a simple skeleton with just a root joint
        joint_parents = th.LongTensor([-1])
        prefix_mul_indices = calc_fk_prefix_multiplication_indices(joint_parents)

        batch_size = 2
        joint_params = th.zeros((batch_size, 1, 7))
        joint_params[:, 0, :3] = th.tensor([1.0, 2.0, 3.0])  # Translation
        joint_params[:, 0, 3:6] = th.tensor([0.1, 0.2, 0.3])  # Rotation (small angles)
        joint_params[:, 0, 6] = 0.0  # Scale (exp2(0) = 1)

        joint_offset = th.zeros((1, 3))
        joint_rotation = th.eye(3).unsqueeze(0)

        # Test get_local_state_from_joint_params
        local_state_t, local_state_r, local_state_s = get_local_state_from_joint_params(
            joint_params, joint_offset, joint_rotation
        )

        # For a root joint, local state should equal global state
        global_state_t, global_state_r, global_state_s = fk_from_local_state(
            local_state_t, local_state_r, local_state_s, prefix_mul_indices
        )

        # Local and global should be identical for root joint
        self.assert_all_close(local_state_t, global_state_t, 1e-6)
        self.assert_all_close(local_state_r, global_state_r, 1e-6)
        self.assert_all_close(local_state_s, global_state_s, 1e-6)

        # Verify expected values
        expected_t = joint_params[:, 0, :3]  # Translation should match
        expected_s = th.ones((batch_size, 1, 1))  # Scale should be 1 (exp2(0))

        self.assert_all_close(global_state_t, expected_t, 1e-6)
        self.assert_all_close(global_state_s, expected_s, 1e-6)


if __name__ == "__main__":
    unittest.main()
