# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Backend Utility Functions for PyMomentum
=========================================

This module provides utility functions that are specific to backend operations
and were previously available in real_lbs_pytorch but are now implemented
within pymomentum for the backend porting effort.
"""

# pyre-strict

from typing import List, Tuple

import numpy as np

import pymomentum.geometry as pym_geometry

import torch


def calc_fk_prefix_multiplication_indices(
    joint_parents: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Calculate prefix multiplication indices for forward kinematics.

    This function computes the indices needed for efficient prefix multiplication
    during forward kinematics computation. The algorithm builds kinematic chains
    for each joint and determines the multiplication order for parallel processing.

    :parameter joint_parents: Parent joint index for each joint. For root joint, its parent is -1.
    :type joint_parents: torch.Tensor
    :return: List of prefix multiplication indices per level. For each level,
             index[0] is the source and index[1] is the target indices.
    :rtype: List[torch.Tensor]
    """
    device = joint_parents.device
    nr_joints = len(joint_parents)
    # get the kinematic chain per joint
    kc_joints = []
    for idx_joint in range(nr_joints):
        kc = [idx_joint]
        while joint_parents[idx_joint] >= 0:
            idx_joint = int(joint_parents[idx_joint])
            kc.append(idx_joint)
        kc_joints.append(kc[::-1])

    # get the multiplication target per joint per level
    prefix_mul_indices = []
    while True:
        level = len(prefix_mul_indices)
        source = []
        target = []
        for kc in kc_joints:
            idx = len(kc) - 1
            current_bit = (idx >> level) & 1
            if current_bit:
                source.append(kc[idx])
                target.append(kc[((idx >> level) << level) - 1])
        if source:
            prefix_mul_indices.append(
                torch.from_numpy(np.array([source, target])).long().to(device)
            )
        else:
            break

    return prefix_mul_indices


def flatten_skinning_weights_and_indices(
    skin_weights: torch.Tensor, skin_indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompress LBS skinning weights and indices into flattened arrays.

    This function takes the typical (V, 8) sparse representation of skinning weights
    and indices and converts them into flattened arrays by removing zero weights,
    making them suitable for efficient skinning computation.

    :parameter skin_weights: Skinning weights tensor of shape (V, 8).
    :type skin_weights: torch.Tensor
    :parameter skin_indices: Skinning joint indices tensor of shape (V, 8).
    :type skin_indices: torch.Tensor
    :return: Tuple of (skin_indices_flattened, skin_weights_flattened, vert_indices_flattened).
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    nr_vertices, nr_nbrs = skin_indices.shape
    device = skin_indices.device

    mask = skin_weights.flatten() > 1e-5

    skin_indices_flattened = skin_indices.clone().flatten()[mask]
    skin_weights_flattened = skin_weights.clone().flatten()[mask]
    vert_indices_flattened = (
        torch.arange(nr_vertices, dtype=torch.long, device=device)[:, None]
        .repeat(1, nr_nbrs)
        .clone()
        .flatten()[mask]
    )

    return skin_indices_flattened, skin_weights_flattened, vert_indices_flattened


class LBSAdapter:
    """Adapter class to make pymomentum Character compatible with LBS interface.

    This adapter provides a compatibility layer that allows pymomentum Character objects
    to be used in tests and code that originally expected the LBS interface from the
    real_lbs_pytorch library. It extracts and converts the necessary properties from
    a Character object into the expected tensor format and provides the methods that
    the original LBS interface had.
    """

    def __init__(self, character: pym_geometry.Character, device: str = "cpu") -> None:
        """Initialize the LBS adapter with a pymomentum Character.

        :param character: The pymomentum Character to adapt
        :param device: Device to place tensors on (default: "cpu")
        """
        self.character: pym_geometry.Character = character
        self._device: torch.device = torch.device(device)
        self.dtype: torch.dtype = torch.float32

        # Convert skeleton properties to tensors
        self.joint_parents: torch.Tensor = torch.tensor(
            character.skeleton.joint_parents, dtype=torch.long, device=self._device
        )
        self.nr_joints: int = character.skeleton.size

        # Convert joint offsets and rotations
        self.joint_offset: torch.Tensor = torch.tensor(
            character.skeleton.offsets, dtype=self.dtype, device=self._device
        )
        self.joint_rotation: torch.Tensor = torch.tensor(
            character.skeleton.pre_rotations, dtype=self.dtype, device=self._device
        )

        # Parameter transform properties
        self.param_transform: torch.Tensor = torch.tensor(
            character.parameter_transform.transform.numpy(),
            dtype=self.dtype,
            device=self._device,
        )
        # Assuming zero offsets for param transform if not available
        self.param_transform_offsets: torch.Tensor = torch.zeros(
            self.param_transform.shape[0], dtype=self.dtype, device=self._device
        )

        # Derived parameter counts from the boolean mask for backward compatibility
        self.nr_position_params: int = int(
            torch.logical_not(character.parameter_transform.scaling_parameters)
            .sum()
            .item()
        )
        self.nr_scaling_params: int = int(
            character.parameter_transform.scaling_parameters.sum().item()
        )

        # Mesh and skinning weights
        assert character.has_mesh
        self.mesh_vertices: torch.Tensor = torch.tensor(
            character.mesh.vertices, dtype=self.dtype, device=self._device
        )
        self.skin_weights: torch.Tensor = torch.tensor(
            character.skin_weights.weight, dtype=self.dtype, device=self._device
        )
        self.skin_indices: torch.Tensor = torch.tensor(
            character.skin_weights.index, dtype=torch.long, device=self._device
        )

        # Zero states for bind pose (for traditional skinning function)
        # t0 should be (J, 3) and r0 should be (J, 3, 3)
        self.joint_state_t_zero: torch.Tensor = torch.from_numpy(
            character.inverse_bind_pose[:, :3, 3]
        )
        self.joint_state_r_zero: torch.Tensor = torch.from_numpy(
            character.inverse_bind_pose[:, :3, :3]
        )

    @property
    def device(self) -> torch.device:
        """Get the device that tensors are stored on."""
        return self._device

    def to(self, device: torch.device) -> "LBSAdapter":
        """Move adapter to specified device.

        :param device: Target device to move tensors to
        :return: New LBSAdapter instance on the target device
        """
        if device == self._device:
            return self

        new_adapter = LBSAdapter(self.character, str(device))
        return new_adapter

    def assemble_pose_and_scale_(
        self, pose: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Assemble pose and scale parameters into model parameters.

        :param pose: Pose parameters tensor
        :param scale: Scale parameters tensor
        :return: Combined model parameters tensor
        """
        model_params = torch.zeros(
            pose.shape[0],
            self.param_transform.shape[1],
            dtype=pose.dtype,
            device=pose.device,
        )
        model_params[:, : self.nr_position_params] = pose
        if scale.numel() > 0:
            model_params[
                :,
                self.nr_position_params : self.nr_position_params
                + self.nr_scaling_params,
            ] = scale
        return model_params
