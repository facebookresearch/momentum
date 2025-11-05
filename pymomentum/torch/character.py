# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Tuple

import pymomentum.geometry as pym_geometry
import pymomentum.quaternion as pym_quaternion
import pymomentum.skel_state as pym_skel_state
import pymomentum.trs as pym_trs
import torch

from pymomentum.backend import skel_state_backend, trs_backend, utils as backend_utils

from pymomentum.torch.parameter_limits import ParameterLimits
from pymomentum.torch.utility import _unsqueeze_joint_params


class Skeleton(torch.nn.Module):
    def __init__(
        self,
        character: pym_geometry.Character,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.register_buffer(
            "joint_translation_offsets",
            torch.tensor(character.skeleton.offsets, dtype=dtype).clone().detach(),
        )
        self.register_buffer(
            "joint_prerotations",
            torch.tensor(character.skeleton.pre_rotations, dtype=dtype)
            .clone()
            .detach(),
        )

        prefix_multiplication_indices = (
            backend_utils.calc_fk_prefix_multiplication_indices(
                joint_parents=torch.tensor(
                    character.skeleton.joint_parents, dtype=torch.int32
                )
            )
        )

        self.register_buffer("pmi", torch.concat(prefix_multiplication_indices, dim=1))
        self._pmi_buffer_sizes: list[int] = [
            t.shape[1] for t in prefix_multiplication_indices
        ]

        self.joint_names: list[str] = character.skeleton.joint_names
        self.register_buffer(
            "joint_parents",
            torch.tensor(character.skeleton.joint_parents, dtype=torch.int32),
        )

    def joint_parameters_to_local_skeleton_state(
        self, joint_parameters: torch.Tensor
    ) -> torch.Tensor:
        if not hasattr(self, "joint_prerotations"):
            raise RuntimeError("Character has no skeleton")

        joint_parameters = _unsqueeze_joint_params(joint_parameters)

        joint_translation_offsets = self.joint_translation_offsets
        while joint_translation_offsets.ndim < joint_parameters.ndim:
            joint_translation_offsets = joint_translation_offsets.unsqueeze(0)

        joint_prerotations = self.joint_prerotations
        while joint_prerotations.ndim < joint_parameters.ndim:
            joint_prerotations = joint_prerotations.unsqueeze(0)

        local_state_t = (
            joint_parameters[..., :3] + self.joint_translation_offsets[None, :]
        )
        local_state_q = pym_quaternion.euler_xyz_to_quaternion(
            joint_parameters[..., 3:6]
        )
        local_state_q = pym_quaternion.multiply_assume_normalized(
            self.joint_prerotations[None],
            local_state_q,
        )
        # exp2 is not supported by all of our export formats, so we have to implement
        # it using exp and natural log instead. The constant here is ln(2.0)
        local_state_s = torch.exp(0.6931471824645996 * joint_parameters[..., 6:])
        return torch.cat([local_state_t, local_state_q, local_state_s], dim=-1)

    def joint_parameters_to_local_trs(
        self, joint_parameters: torch.Tensor
    ) -> pym_trs.TRSTransform:
        """
        Convert joint parameters to local TRS (Translation-Rotation-Scale) state.

        This function takes joint parameters and converts them to the TRS representation
        as a tuple, consistent with the pymomentum.trs module interface.
        The TRS format is useful for applications that need explicit access to individual
        transformation components and can be more efficient than the skeleton state format.

        Args:
            joint_parameters: Joint parameters tensor of shape (batch_size, num_joints * 7)
                or (batch_size, num_joints, 7). Each joint has 7 parameters:
                - [0:3] translation offset
                - [3:6] euler xyz rotation angles
                - [6] log2 scale factor

        Returns:
            TRS transform tuple (translation, rotation_matrix, scale) where:
                - translation: Local joint translations, shape (batch_size, num_joints, 3)
                - rotation_matrix: Local joint rotations as 3x3 matrices, shape (batch_size, num_joints, 3, 3)
                - scale: Local joint scales, shape (batch_size, num_joints, 1)

        Raises:
            RuntimeError: If the character has no skeleton

        Note:
            This function is equivalent to joint_parameters_to_local_skeleton_state() but
            returns the TRS format instead of the 8-parameter skeleton state format.
            It uses the TRS backend for efficient computation.
        """
        if not hasattr(self, "joint_prerotations"):
            raise RuntimeError("Character has no skeleton")

        joint_parameters = _unsqueeze_joint_params(joint_parameters)

        # Convert joint_prerotations from quaternions to rotation matrices
        joint_rotation_matrices = pym_quaternion.to_rotation_matrix(
            self.joint_prerotations
        )

        # Use the TRS backend for efficient conversion
        return trs_backend.get_local_state_from_joint_params(
            joint_params=joint_parameters,
            joint_offset=self.joint_translation_offsets,
            joint_rotation=joint_rotation_matrices,
        )

    def joint_parameters_to_trs(
        self, joint_parameters: torch.Tensor
    ) -> pym_trs.TRSTransform:
        """
        Convert joint parameters directly to global TRS (Translation-Rotation-Scale) state.

        This function performs the full forward kinematics pipeline using the TRS backend,
        converting joint parameters to local TRS state and then to global TRS state in
        a single call. This is more efficient than calling the two-step process separately.

        Args:
            joint_parameters: Joint parameters tensor of shape (batch_size, num_joints * 7)
                or (batch_size, num_joints, 7). Each joint has 7 parameters:
                - [0:3] translation offset
                - [3:6] euler xyz rotation angles
                - [6] log2 scale factor

        Returns:
            TRS transform tuple (translation, rotation_matrix, scale) where:
                - translation: Global joint translations, shape (batch_size, num_joints, 3)
                - rotation_matrix: Global joint rotations as 3x3 matrices, shape (batch_size, num_joints, 3, 3)
                - scale: Global joint scales, shape (batch_size, num_joints, 1)

        Raises:
            RuntimeError: If the character has no skeleton

        Note:
            This is equivalent to calling joint_parameters_to_local_trs() followed by
            local_trs_to_global_trs(), but more efficient as it avoids intermediate steps.
        """
        # Get local TRS state from joint parameters
        local_trs = self.joint_parameters_to_local_trs(joint_parameters)
        local_state_t, local_state_r, local_state_s = local_trs

        # Convert local TRS to global TRS using forward kinematics
        return trs_backend.global_trs_state_from_local_trs_state(
            local_state_t=local_state_t,
            local_state_r=local_state_r,
            local_state_s=local_state_s,
            prefix_mul_indices=list(
                self.pmi.split(
                    split_size=self._pmi_buffer_sizes,
                    dim=1,
                )
            ),
        )

    def local_trs_to_global_trs(
        self,
        local_trs: pym_trs.TRSTransform,
    ) -> pym_trs.TRSTransform:
        """
        Convert local TRS state to global TRS state using forward kinematics.

        This function performs forward kinematics on the local TRS transformations
        to compute the global joint transformations. This is useful when you have
        local TRS states from other sources and need to convert them to global states.

        Args:
            local_trs: Local TRS transform tuple (translation, rotation_matrix, scale) where:
                - translation: Local joint translations, shape (batch_size, num_joints, 3)
                - rotation_matrix: Local joint rotations as 3x3 matrices, shape (batch_size, num_joints, 3, 3)
                - scale: Local joint scales, shape (batch_size, num_joints, 1)

        Returns:
            TRS transform tuple (translation, rotation_matrix, scale) where:
                - translation: Global joint translations, shape (batch_size, num_joints, 3)
                - rotation_matrix: Global joint rotations as 3x3 matrices, shape (batch_size, num_joints, 3, 3)
                - scale: Global joint scales, shape (batch_size, num_joints, 1)

        Note:
            This function uses the TRS backend for efficient forward kinematics computation.
        """
        local_state_t, local_state_r, local_state_s = local_trs
        return trs_backend.global_trs_state_from_local_trs_state(
            local_state_t=local_state_t,
            local_state_r=local_state_r,
            local_state_s=local_state_s,
            prefix_mul_indices=list(
                self.pmi.split(
                    split_size=self._pmi_buffer_sizes,
                    dim=1,
                )
            ),
        )

    def local_skeleton_state_to_skeleton_state(
        self, local_skel_state: torch.Tensor
    ) -> torch.Tensor:
        return skel_state_backend.global_skel_state_from_local_skel_state(
            local_skel_state=local_skel_state,
            prefix_mul_indices=list(
                self.pmi.split(
                    split_size=self._pmi_buffer_sizes,
                    dim=1,
                )
            ),
        )

    def skeleton_state_to_local_skeleton_state(
        self, skel_state: torch.Tensor
    ) -> torch.Tensor:
        joint_parents = self.joint_parents
        parent_skel_states = skel_state.index_select(
            -2, torch.clamp(joint_parents, min=0)
        )
        # global = parent * local
        local_skel_states = pym_skel_state.multiply(
            pym_skel_state.inverse(parent_skel_states), skel_state
        )
        while joint_parents.ndim + 1 < local_skel_states.ndim:
            joint_parents = joint_parents[None, ...]

        local_skel_states = torch.where(
            (joint_parents >= 0)[..., None], local_skel_states, skel_state
        )

        return local_skel_states

    def local_skeleton_state_to_joint_parameters(
        self,
        local_skel_state: torch.Tensor,
    ) -> torch.Tensor:
        joint_translation_offsets = self.joint_translation_offsets
        joint_prerotations = self.joint_prerotations

        # Compute translation joint parameters
        translation_params = local_skel_state[..., :3] - joint_translation_offsets[None]

        # Invert out the pre-rotations:
        local_rotations = local_skel_state[..., 3:7]
        adjusted_rotations = pym_quaternion.multiply(
            pym_quaternion.inverse(joint_prerotations[None]),
            local_rotations,
        )

        # Use the pymomentum.quaternion implementation instead of real_lbs_quaternion's implementation
        # because it's more numerically stable. This is important for backpropagating through this
        # function.
        rotation_joint_params = pym_quaternion.quaternion_to_xyz_euler(
            adjusted_rotations, eps=1e-6
        )

        # Compute scale joint parameters
        scale_joint_params = torch.log2(local_skel_state[..., 7:8])

        return torch.cat(
            [translation_params, rotation_joint_params, scale_joint_params], dim=-1
        ).flatten(-2, -1)

    def skeleton_state_to_joint_parameters(
        self, skel_state: torch.Tensor
    ) -> torch.Tensor:
        return self.local_skeleton_state_to_joint_parameters(
            self.skeleton_state_to_local_skeleton_state(skel_state)
        )

    def forward(self, joint_parameters: torch.Tensor) -> torch.Tensor:
        if joint_parameters.ndim == 1:
            joint_parameters = joint_parameters[None, :]
        return self.local_skeleton_state_to_skeleton_state(
            self.joint_parameters_to_local_skeleton_state(joint_parameters)
        )


class LinearBlendSkinning(torch.nn.Module):
    def __init__(
        self,
        character: pym_geometry.Character,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.register_buffer(
            "inverse_bind_pose",
            pym_skel_state.from_matrix(
                torch.tensor(character.inverse_bind_pose, dtype=dtype)
            )
            .detach()
            .clone(),
        )

        self.num_vertices: int = character.skin_weights.index.shape[0]

        (
            skin_indices_flattened,
            skin_weights_flattened,
            vert_indices_flattened,
        ) = backend_utils.flatten_skinning_weights_and_indices(
            skin_weights=torch.tensor(character.skin_weights.weight, dtype=dtype),
            skin_indices=torch.tensor(character.skin_weights.index, dtype=torch.int32),
        )

        self.register_buffer(
            "skin_indices_flattened", skin_indices_flattened.detach().clone()
        )
        self.register_buffer(
            "skin_weights_flattened", skin_weights_flattened.detach().clone()
        )
        self.register_buffer(
            "vert_indices_flattened", vert_indices_flattened.detach().clone()
        )

    def forward(
        self,
        skel_state: torch.Tensor,
        rest_vertex_positions: torch.Tensor,
    ) -> torch.Tensor:
        assert rest_vertex_positions.shape[-1] == 3
        assert rest_vertex_positions.shape[-2] == self.num_vertices

        inverse_bind_pose = self.inverse_bind_pose
        while inverse_bind_pose.ndim < skel_state.ndim:
            inverse_bind_pose = inverse_bind_pose.unsqueeze(0)

        return skel_state_backend.skin_points_from_skel_state(
            template=rest_vertex_positions,
            global_skel_state=skel_state,
            binded_skel_state_inv=inverse_bind_pose,
            skin_indices_flattened=self.skin_indices_flattened,
            skin_weights_flattened=self.skin_weights_flattened,
            vert_indices_flattened=self.vert_indices_flattened,
        )

    def skin_with_trs(
        self,
        global_trs: pym_trs.TRSTransform,
        rest_vertex_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply skinning using global TRS state (Translation-Rotation-Scale).

        This method takes global TRS state as a tuple and applies linear blend skinning
        using the TRS backend. This is useful when working directly with TRS representations
        or when you want to avoid conversions to skeleton state format.

        Args:
            global_trs: Global TRS transform tuple (translation, rotation_matrix, scale) where:
                - translation: Global joint translations, shape (batch_size, num_joints, 3)
                - rotation_matrix: Global joint rotations as 3x3 matrices, shape (batch_size, num_joints, 3, 3)
                - scale: Global joint scales, shape (batch_size, num_joints, 1)
            rest_vertex_positions: Rest vertex positions, shape (batch_size, num_vertices, 3)
                or (num_vertices, 3) which will be expanded to batch size

        Returns:
            skinned_vertices: Skinned vertex positions, shape (batch_size, num_vertices, 3)

        Note:
            This method uses the TRS backend skinning function for efficient computation.
        """
        global_state_t, global_state_r, global_state_s = global_trs

        inv_bind_pose_rot: torch.Tensor = pym_quaternion.to_rotation_matrix(
            self.inverse_bind_pose[..., 3:7]
        )
        inv_bind_pose_trans: torch.Tensor = self.inverse_bind_pose[..., 0:3]

        return trs_backend.skinning(
            template=rest_vertex_positions,
            t=global_state_t,
            r=global_state_r,
            s=global_state_s,
            t0=inv_bind_pose_trans,
            r0=inv_bind_pose_rot,
            skin_indices_flattened=self.skin_indices_flattened,
            skin_weights_flattened=self.skin_weights_flattened,
            vert_indices_flattened=self.vert_indices_flattened,
        )

    def skin_with_transforms(
        self,
        global_state_t: torch.Tensor,
        global_state_r: torch.Tensor,
        global_state_s: torch.Tensor,
        rest_vertex_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply skinning using individual global TRS tensors (legacy interface).

        This method provides backward compatibility for code that passes individual
        tensors instead of the TRS tuple format. Internally, it creates a tuple
        and delegates to skin_with_trs().

        Args:
            global_state_t: Global joint translations, shape (batch_size, num_joints, 3)
            global_state_r: Global joint rotations as 3x3 matrices, shape (batch_size, num_joints, 3, 3)
            global_state_s: Global joint scales, shape (batch_size, num_joints, 1)
            rest_vertex_positions: Rest vertex positions, shape (batch_size, num_vertices, 3) or (num_vertices, 3) which will be expanded to batch size

        Returns:
            skinned_vertices: Skinned vertex positions, shape (batch_size, num_vertices, 3)

        Note:
            This method is provided for backward compatibility. New code should use
            skin_with_trs() with the TRS tuple interface.
        """
        global_trs = (global_state_t, global_state_r, global_state_s)
        return self.skin_with_trs(global_trs, rest_vertex_positions)

    def unpose(
        self, global_joint_state: torch.Tensor, verts: torch.Tensor
    ) -> torch.Tensor:
        return skel_state_backend.unpose_from_momentum_global_joint_state(
            verts,
            global_joint_state,
            self.inverse_bind_pose,
            self.skin_indices_flattened,
            self.skin_weights_flattened,
            self.vert_indices_flattened,
            with_high_precision=True,
        )


class Mesh(torch.nn.Module):
    def __init__(
        self,
        character: pym_geometry.Character,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.register_buffer(
            "rest_vertices",
            torch.tensor(character.mesh.vertices, dtype=dtype).clone().detach(),
        )

        self.register_buffer(
            "faces",
            torch.tensor(character.mesh.faces, dtype=torch.int32).clone().detach(),
        )

        self.register_buffer(
            "texcoords",
            torch.tensor(character.mesh.texcoords, dtype=dtype).clone().detach(),
        )

        self.register_buffer(
            "texcoord_faces",
            torch.tensor(character.mesh.texcoord_faces, dtype=torch.int32)
            .clone()
            .detach(),
        )


class BlendShapeBase(torch.nn.Module):
    def __init__(
        self,
        shape_vectors: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.register_buffer(
            "shape_vectors",
            shape_vectors.to(dtype=dtype).clone().detach(),
        )

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nvd, ...n -> ...vd", self.shape_vectors, coeffs)


class BlendShape(BlendShapeBase):
    def __init__(
        self,
        character: pym_geometry.Character,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        assert character.blend_shape is not None
        super().__init__(torch.tensor(character.blend_shape.shape_vectors), dtype=dtype)

        assert character.blend_shape is not None
        self.register_buffer(
            "base_shape",
            torch.tensor(character.blend_shape.base_shape, dtype=dtype)
            .clone()
            .detach(),
        )

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "base_shape"):
            raise RuntimeError("Character has no blendshapes")
        return super().forward(coeffs) + self.base_shape


class ParameterTransform(torch.nn.Module):
    def __init__(
        self,
        character: pym_geometry.Character,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.register_buffer(
            "parameter_transform",
            character.parameter_transform.transform.to(dtype=dtype)
            .clone()
            .detach()
            .requires_grad_(False),
        )

        self.register_buffer(
            "pose_parameters",
            character.parameter_transform.pose_parameters.clone()
            .detach()
            .requires_grad_(False),
        )

        self.register_buffer(
            "rigid_parameters",
            character.parameter_transform.rigid_parameters.clone()
            .detach()
            .requires_grad_(False),
        )

        self.register_buffer(
            "scaling_parameters",
            character.parameter_transform.scaling_parameters.clone()
            .detach()
            .requires_grad_(False),
        )

        self.parameter_names: list[str] = character.parameter_transform.names

    def forward(self, model_parameters: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "parameter_transform"):
            raise RuntimeError("Character has no parameter transform")
        return torch.einsum(
            "dn,...n->...d",
            self.parameter_transform,
            model_parameters,
        )


class InverseParameterTransform(torch.nn.Module):
    def __init__(
        self, parameter_transform: ParameterTransform, rcond: float = 1e-5
    ) -> None:
        super().__init__()
        self.rcond: float = rcond

        # We tried using torch.linalg.lstsq, but the fully-pivoting and SVD-based
        # versions don't run on the GPU while the non-pivoting version gives bad
        # results for non-unique parameter transforms.  Therefore let's precompute
        # the full SVD, this is maybe overkill but we only have to do it once.

        # Full SVD for general case
        U, S_inv, V = self._compute_svd(parameter_transform.parameter_transform)
        self.register_buffer("svd_U", U)
        self.register_buffer("svd_S_inv", S_inv)
        self.register_buffer("svd_V", V)

    def _compute_svd(
        self,
        matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the Singular Value Decomposition (SVD) of a matrix and return the components needed for pseudoinverse.

        This function performs SVD on the input matrix and returns the components U, S_inv, and V,
        where S_inv contains the reciprocals of singular values above the threshold defined by self.rcond,
        and zeros for singular values below the threshold to avoid numerical instability.

        Parameters
        ----------
        matrix : torch.Tensor
            The input matrix to decompose using SVD.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - U: Left singular vectors (detached from computation graph)
            - S_inv: Reciprocal of singular values with thresholding (detached from computation graph)
            - V: Right singular vectors (detached from computation graph)
        """
        U, S, V = torch.linalg.svd(matrix, full_matrices=False)
        # Avoid divide-by-zero
        S_inv = torch.where(S < self.rcond, 0.0, torch.reciprocal(S))
        return U.detach(), S_inv.detach(), V.detach()

    def _solve_via_svd(
        self, U: torch.Tensor, S_inv: torch.Tensor, V: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # U * S * V^T * model_params = y
        # model_params = V * S^-1 * U^T * y

        # First multiply by U^T:
        Ut_y = torch.einsum("jm,...j->...m", U, y)
        while S_inv.ndim < Ut_y.ndim:
            S_inv = S_inv.unsqueeze(0)
        # Next divide by S (a diagonal matrix):
        tmp = Ut_y * S_inv
        # Finally multiply by V^T:
        return torch.einsum("ij,...i->...j", V, tmp)

    def joint_parameters_to_model_parameters(
        self, joint_parameters: torch.Tensor
    ) -> torch.Tensor:
        return self._solve_via_svd(
            self.svd_U, self.svd_S_inv, self.svd_V, joint_parameters
        )


class AdvancedInverseParameterTransform(InverseParameterTransform):
    """
    This is a more advanced version of InverseParameterTransform that can handle
    the case where pose and scale are mutually exclusive (i.e. no joint parameter
    is affected by both pose and scale).  In this case, we can compute a pose-only
    SVD and use that to compute the inverse parameter transform.
    """

    def __init__(
        self, parameter_transform: ParameterTransform, rcond: float = 1e-5
    ) -> None:
        super().__init__(parameter_transform, rcond)
        self.parameter_transform: ParameterTransform = parameter_transform

        # Pose-only SVD (for partial inverse when scales are known)
        pose_scale_are_mutually_exclusive, affected_by_pose_mask = (
            self.check_if_pose_scale_are_mutually_exclusive()
        )
        self.pose_scale_are_mutually_exclusive: bool = pose_scale_are_mutually_exclusive
        self.register_buffer("affected_by_pose_mask", affected_by_pose_mask)
        if pose_scale_are_mutually_exclusive:
            # If pose and scale are mutually exclusive, we can compute a pose-only SVD
            pose_matrix = parameter_transform.parameter_transform[
                affected_by_pose_mask
            ][:, parameter_transform.pose_parameters]
            U_p, S_inv_p, V_p = self._compute_svd(pose_matrix)

            self.register_buffer("svd_U_poses", U_p)
            self.register_buffer("svd_S_poses_inv", S_inv_p)
            self.register_buffer("svd_V_poses", V_p)
            self.register_buffer("scale_only_parameter_transform", torch.tensor([]))

        else:
            pose_matrix = parameter_transform.parameter_transform[
                ..., parameter_transform.pose_parameters
            ]
            scale_matrix = parameter_transform.parameter_transform[
                ..., parameter_transform.scaling_parameters
            ]
            U_p, S_inv_p, V_p = self._compute_svd(pose_matrix)

            self.register_buffer(
                "scale_only_parameter_transform", scale_matrix.detach()
            )
            self.register_buffer("svd_U_poses", U_p)
            self.register_buffer("svd_S_poses_inv", S_inv_p)
            self.register_buffer("svd_V_poses", V_p)

    def check_if_pose_scale_are_mutually_exclusive(self) -> Tuple[bool, torch.Tensor]:
        # Check if pose and scale are mutually exclusive (i.e. no joint parameter is affected
        # by both pose and scale)
        pt = self.parameter_transform
        transform = pt.parameter_transform
        scale_mask = pt.scaling_parameters
        pose_mask = pt.pose_parameters

        # Compute which joint params are affected by scale vs pose
        affected_by_scale_mask = (transform[:, scale_mask] != 0).any(dim=1)
        affected_by_pose_mask = (transform[:, pose_mask] != 0).any(dim=1)

        # Mutually exclusive if no joint is affected by both
        return (
            not torch.any(affected_by_scale_mask & affected_by_pose_mask)
        ), affected_by_pose_mask

    def joint_parameters_to_model_parameters_with_known_scales(
        self, joint_parameters: torch.Tensor, scale_parameters: torch.Tensor
    ) -> torch.Tensor:
        if self.pose_scale_are_mutually_exclusive:
            adjusted_joint_parameters = joint_parameters[
                ..., self.affected_by_pose_mask
            ]
        else:
            # y_adjusted = y - M_fixed @ x_fixed
            adjusted_joint_parameters = joint_parameters - torch.einsum(
                "dn,...n->...d", self.scale_only_parameter_transform, scale_parameters
            )
        poses = self._solve_via_svd(
            self.svd_U_poses,
            self.svd_S_poses_inv,
            self.svd_V_poses,
            adjusted_joint_parameters,
        )

        result = torch.zeros(
            *poses.shape[:-1],
            self.parameter_transform.parameter_transform.shape[-1],
            device=poses.device,
            dtype=poses.dtype,
        )
        result[..., self.parameter_transform.pose_parameters] = poses
        result[..., self.parameter_transform.scaling_parameters] = scale_parameters
        return result


class Character(torch.nn.Module):
    def __init__(
        self,
        character: pym_geometry.Character,
        *,
        has_parameter_transform: bool = True,
        has_skeleton: bool = True,
        has_rest_mesh: bool = True,
        has_skinning: bool = True,
        has_limits: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if has_skeleton:
            self.skeleton: Skeleton = Skeleton(character, dtype=dtype)

        if has_rest_mesh:
            self.mesh: Mesh = Mesh(character, dtype=dtype)

        if has_parameter_transform:
            self.parameter_transform: ParameterTransform = ParameterTransform(
                character, dtype=dtype
            )

        if has_skinning:
            self.linear_blend_skinning: LinearBlendSkinning = LinearBlendSkinning(
                character, dtype=dtype
            )

        if has_limits:
            self.parameter_limits: ParameterLimits = ParameterLimits(
                character.parameter_limits, dtype=dtype
            )

        if has_rest_mesh and character.blend_shape is not None:
            self.blend_shape: BlendShape = BlendShape(character, dtype=dtype)

    def joint_parameters_to_local_skeleton_state(
        self, joint_parameters: torch.Tensor
    ) -> torch.Tensor:
        if not hasattr(self, "skeleton"):
            raise RuntimeError("Character has no skeleton, please provide one")
        return self.skeleton.joint_parameters_to_local_skeleton_state(joint_parameters)

    def model_parameters_to_local_skeleton_state(
        self, model_parameters: torch.Tensor
    ) -> torch.Tensor:
        if not hasattr(self, "skeleton"):
            raise RuntimeError("Character has no skeleton, please provide one")
        return self.joint_parameters_to_local_skeleton_state(
            self.model_parameters_to_joint_parameters(model_parameters)
        )

    def local_skeleton_state_to_skeleton_state(
        self, local_skel_state: torch.Tensor
    ) -> torch.Tensor:
        if not hasattr(self, "skeleton"):
            raise RuntimeError("Character has no skeleton, please provide one")
        return self.skeleton.local_skeleton_state_to_skeleton_state(local_skel_state)

    def model_parameters_to_joint_parameters(
        self, model_parameters: torch.Tensor
    ) -> torch.Tensor:
        if not hasattr(self, "parameter_transform"):
            raise RuntimeError(
                "Character has no parameter transform, please provide one"
            )
        return self.parameter_transform.forward(model_parameters)

    def joint_parameters_to_skeleton_state(
        self, joint_parameters: torch.Tensor
    ) -> torch.Tensor:
        local_skel_state = self.joint_parameters_to_local_skeleton_state(
            joint_parameters
        )
        return self.local_skeleton_state_to_skeleton_state(local_skel_state)

    def model_parameters_to_skeleton_state(
        self, model_parameters: torch.Tensor
    ) -> torch.Tensor:
        return self.joint_parameters_to_skeleton_state(
            self.model_parameters_to_joint_parameters(model_parameters)
        )

    def model_parameters_to_blendshape_coefficients(
        self, model_parameters: torch.Tensor
    ) -> torch.Tensor:
        return model_parameters[..., self.parameter_transform.blendshape_parameters]

    def bind_pose(self) -> torch.Tensor:
        return self.joint_parameters_to_skeleton_state(
            torch.zeros((1, len(self.skeleton.joint_names), 7))
        )

    def skin_points(
        self,
        skel_state: torch.Tensor,
        rest_vertex_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rest_vertex_positions is None:
            if not hasattr(self, "mesh"):
                raise RuntimeError("Character has no rest mesh, please provide one")
            rest_vertex_positions = self.mesh.rest_vertices

        if not hasattr(self, "linear_blend_skinning"):
            raise RuntimeError("Character has no skinning information")

        return self.linear_blend_skinning.forward(
            skel_state,
            rest_vertex_positions,
        )

    def unpose(
        self, skel_state: torch.Tensor, vertex_positions: torch.Tensor
    ) -> torch.Tensor:
        if not hasattr(self, "linear_blend_skinning"):
            raise RuntimeError("Character has no skinning information")

        return self.linear_blend_skinning.unpose(skel_state, vertex_positions)
