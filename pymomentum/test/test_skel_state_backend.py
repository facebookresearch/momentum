# pyre-strict
import unittest
from typing import List

import torch as th

th.autograd.set_detect_anomaly(True)

import pymomentum.models as pym_models
import pymomentum.quaternion as pym_quaternion
import pymomentum.trs as pym_trs
from pymomentum.backend.skel_state_backend import (
    fk_from_local_momentum_state,
    get_local_momentum_state_from_joint_params,
    skinning_from_momentum_state,
    skinning_oriented_points_from_momentum_state,
)
from pymomentum.backend.trs_backend import fk_from_local_state, skinning
from pymomentum.backend.utils import (
    calc_fk_prefix_multiplication_indices,
    flatten_skinning_weights_and_indices,
    LBSAdapter,
)
from torch.nn import Parameter as P


class TestPyMomentumBackend(unittest.TestCase):
    """
    this class assumes that the lbs_pytorch backend implementation (test_lbs.py) is correct
    and use lbs_pytorch backend as the ground truth.
    """

    def assert_all_close(self, a: th.Tensor, b: th.Tensor, tol: float = 1e-5) -> None:
        mask = th.isclose(a, b, atol=tol, rtol=tol)
        if not th.all(mask):
            print(
                f"assert_all_close failed with {tol=} on joint",
                th.unique(th.where(~mask)[1].cpu()),
            )
        self.assertTrue(th.allclose(a, b, atol=tol, rtol=tol))

    def init_sim3(self, batch_size: int, nr_joints: int) -> th.Tensor:
        t = th.randn((batch_size, nr_joints, 3), dtype=th.float32)
        q = th.randn((batch_size, nr_joints, 4), dtype=th.float32)
        s = th.randn((batch_size, nr_joints, 1), dtype=th.float32)

        q = pym_quaternion.normalize(q)
        s = th.exp2(s).clip(max=2)
        return th.cat([t, q, s], dim=-1)

    def eval_momentum_fk(
        self,
        local_state: th.Tensor,
        joint_parents: th.Tensor,
        prefix_mul_indices: List[th.Tensor],
        dtype: th.dtype = th.float32,
        tol: float = 1e-3,
    ) -> None:
        # forward via sim3
        local_state = P(local_state.to(dtype))
        global_state = fk_from_local_momentum_state(local_state, prefix_mul_indices)
        t1, r1, s1 = pym_trs.from_skeleton_state(global_state)

        gt = th.randn_like(t1)
        gr = th.randn_like(r1)
        gs = th.randn_like(s1)

        (gl1,) = th.autograd.grad(
            outputs=[t1, r1, s1],
            inputs=[local_state],
            grad_outputs=[gt, gr, gs],
        )

        # forward via the new lbs_pytorch jit fk
        tl, rl, sl = pym_trs.from_skeleton_state(local_state)
        t2, r2, s2 = fk_from_local_state(tl, rl, sl, prefix_mul_indices)

        (gl2,) = th.autograd.grad(
            outputs=[t2, r2, s2],
            inputs=[local_state],
            grad_outputs=[gt, gr, gs],
        )

        self.assert_all_close(t1, t2, tol)
        self.assert_all_close(r1, r2, tol)
        self.assert_all_close(s1, s2, tol)
        self.assert_all_close(gl1, gl2, tol)

    def test_momentum_fk(self) -> None:
        th.manual_seed(10023893)
        lbs = LBSAdapter(pym_models.load_default_trinity(), device="cpu")
        prefix_mul_indices = calc_fk_prefix_multiplication_indices(lbs.joint_parents)
        local_state = self.init_sim3(1024, lbs.nr_joints).to(lbs.device)
        self.eval_momentum_fk(
            local_state, lbs.joint_parents, prefix_mul_indices, th.float32, 1e-3
        )

    def test_momentum_fk_sanity_check(self) -> None:
        joint_parents = th.LongTensor([-1, 0, 1, 2])
        prefix_mul_indices = calc_fk_prefix_multiplication_indices(joint_parents)
        local_state = self.init_sim3(1, joint_parents.shape[0])
        self.eval_momentum_fk(local_state, joint_parents, prefix_mul_indices, tol=1e-5)

    def test_momentum_skinning(self) -> None:
        lbs = LBSAdapter(pym_models.load_default_trinity(), device="cpu")
        prefix_mul_indices = calc_fk_prefix_multiplication_indices(lbs.joint_parents)

        batch_size = 8
        pose = th.randn(
            (batch_size, lbs.nr_position_params), dtype=lbs.dtype, device=lbs.device
        )
        scale = (
            th.randn((lbs.nr_scaling_params,), dtype=lbs.dtype, device=lbs.device)
            / 10.0
        )
        model_params = lbs.assemble_pose_and_scale_(pose, scale)
        joint_params = (
            th.einsum(
                "dn,bn->bd",
                lbs.param_transform,
                model_params,
            )
            + lbs.param_transform_offsets
        ).view(batch_size, lbs.nr_joints, 7)

        local_state = get_local_momentum_state_from_joint_params(
            joint_params, lbs.joint_offset, lbs.joint_rotation
        )
        global_state = fk_from_local_momentum_state(local_state, prefix_mul_indices)

        bind_state_inv_t = lbs.joint_state_t_zero
        bind_state_inv_q = pym_quaternion.from_rotation_matrix(lbs.joint_state_r_zero)
        bind_state_inv_s = th.ones(
            (lbs.nr_joints, 1), dtype=lbs.dtype, device=lbs.device
        )
        bind_state_inv = th.cat(
            [bind_state_inv_t, bind_state_inv_q, bind_state_inv_s], dim=-1
        )

        (
            skin_indices_flattened,
            skin_weights_flattened,
            vert_indices_flattened,
        ) = flatten_skinning_weights_and_indices(lbs.skin_weights, lbs.skin_indices)

        # forward via sim3
        pgs = P(global_state)
        pv = P(lbs.mesh_vertices)

        v1 = skinning_from_momentum_state(
            template=pv,
            global_joint_state=pgs,
            binded_joint_state_inv=bind_state_inv.unsqueeze(0),
            skin_indices_flattened=skin_indices_flattened,
            skin_weights_flattened=skin_weights_flattened,
            vert_indices_flattened=vert_indices_flattened,
        )
        grad = th.zeros_like(v1)

        ggs1, gv1 = th.autograd.grad(
            outputs=[v1],
            inputs=[pgs, pv],
            grad_outputs=[grad],
        )

        # forward via the traditional t,r,s skinning
        pt, pr, ps = pym_trs.from_skeleton_state(pgs)

        v2 = skinning(
            template=pv,
            t=pt,
            r=pr,
            s=ps,
            t0=lbs.joint_state_t_zero,
            r0=lbs.joint_state_r_zero,
            skin_indices_flattened=skin_indices_flattened,
            skin_weights_flattened=skin_weights_flattened,
            vert_indices_flattened=vert_indices_flattened,
        )

        ggs2, gv2 = th.autograd.grad(
            outputs=[v2],
            inputs=[pgs, pv],
            grad_outputs=[grad],
        )

        self.assert_all_close(v1, v2, 1e-3)
        self.assert_all_close(gv1, gv2, 1e-3)
        self.assert_all_close(ggs1, ggs2, 1e-3)

    def test_skinning_gaussians_from_momentum_state(self) -> None:
        lbs = LBSAdapter(pym_models.load_default_trinity(), device="cpu")
        prefix_mul_indices = calc_fk_prefix_multiplication_indices(lbs.joint_parents)

        batch_size = 8
        pose = th.randn(
            (batch_size, lbs.nr_position_params), dtype=lbs.dtype, device=lbs.device
        )
        scale = (
            th.randn((lbs.nr_scaling_params,), dtype=lbs.dtype, device=lbs.device)
            / 10.0
        )
        model_params = lbs.assemble_pose_and_scale_(pose, scale)
        joint_params = (
            th.einsum(
                "dn,bn->bd",
                lbs.param_transform,
                model_params,
            )
            + lbs.param_transform_offsets
        ).view(batch_size, lbs.nr_joints, 7)

        local_state_t = joint_params[:, :, :3] + lbs.joint_offset[None, :]
        local_state_q = pym_quaternion.euler_xyz_to_quaternion(joint_params[:, :, 3:6])
        local_state_q = pym_quaternion.multiply(
            pym_quaternion.from_rotation_matrix(lbs.joint_rotation[None]), local_state_q
        )
        local_state_s = th.exp2(joint_params[:, :, 6:])

        local_state = th.cat([local_state_t, local_state_q, local_state_s], dim=-1)
        global_state = fk_from_local_momentum_state(local_state, prefix_mul_indices)

        bind_state_inv_t = lbs.joint_state_t_zero
        bind_state_inv_q = pym_quaternion.from_rotation_matrix(lbs.joint_state_r_zero)
        bind_state_inv_s = th.ones(
            (lbs.nr_joints, 1), dtype=lbs.dtype, device=lbs.device
        )
        bind_state_inv = th.cat(
            [bind_state_inv_t, bind_state_inv_q, bind_state_inv_s], dim=-1
        )

        (
            skin_indices_flattened,
            skin_weights_flattened,
            vert_indices_flattened,
        ) = flatten_skinning_weights_and_indices(lbs.skin_weights, lbs.skin_indices)

        # Create gaussian parameters
        num_verts = lbs.mesh_vertices.shape[0]
        means = th.randn((batch_size, num_verts, 3), dtype=lbs.dtype, device=lbs.device)
        quaternions = th.randn(
            (batch_size, num_verts, 4), dtype=lbs.dtype, device=lbs.device
        )
        quaternions = pym_quaternion.normalize(quaternions)

        # forward via skinning_oriented_points_from_momentum_state
        pgs = P(global_state)
        pmeans = P(means)
        pquaternions = P(quaternions)

        skinned_means, skinned_quaternions = (
            skinning_oriented_points_from_momentum_state(
                means=pmeans,
                quaternions=pquaternions,
                global_joint_state=pgs,
                binded_joint_state_inv=bind_state_inv.unsqueeze(0),
                skin_indices_flattened=skin_indices_flattened,
                skin_weights_flattened=skin_weights_flattened,
                vert_indices_flattened=vert_indices_flattened,
            )
        )

        grad_pos = th.randn_like(skinned_means)
        grad_quats = th.randn_like(skinned_quaternions)

        ggs1, gmeans1, gquaternions1 = th.autograd.grad(
            outputs=[skinned_means, skinned_quaternions],
            inputs=[pgs, pmeans, pquaternions],
            grad_outputs=[grad_pos, grad_quats],
        )

        # forward via traditional skinning_from_momentum_state for position comparison
        pv = P(means)
        skinned_traditional = skinning_from_momentum_state(
            template=pv,
            global_joint_state=pgs,
            binded_joint_state_inv=bind_state_inv.unsqueeze(0),
            skin_indices_flattened=skin_indices_flattened,
            skin_weights_flattened=skin_weights_flattened,
            vert_indices_flattened=vert_indices_flattened,
        )

        ggs2, gv2 = th.autograd.grad(
            outputs=[skinned_traditional],
            inputs=[pgs, pv],
            grad_outputs=[grad_pos],
        )

        # Compare position parts
        self.assert_all_close(skinned_means, skinned_traditional, 1e-3)
        self.assert_all_close(gmeans1, gv2, 1e-3)


if __name__ == "__main__":
    unittest.main()
