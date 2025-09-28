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
