# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

# pyre-strict


def _squeeze_joint_params(joint_params: torch.Tensor) -> torch.Tensor:
    if joint_params.shape[-1] == 7:
        joint_params = joint_params.flatten(start_dim=-2)
    return joint_params


def _unsqueeze_joint_params(joint_params: torch.Tensor) -> torch.Tensor:
    if joint_params.shape[-1] != 7:
        return joint_params.view(list(joint_params.shape[:-1]) + [-1, 7])
    return joint_params
