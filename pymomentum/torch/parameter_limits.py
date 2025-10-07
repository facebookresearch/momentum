# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import pymomentum.geometry as pym_geometry
import pymomentum.skel_state as pym_skel_state
import torch

from .utility import _squeeze_joint_params

# pyre-strict


class ParameterLimits(torch.nn.Module):
    """Native PyTorch implementation of momentum's LimitErrorFunction, used to
    enforce parameter limits using a soft constraint."""

    def __init__(
        self,
        parameter_limits: list[pym_geometry.ParameterLimit],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        kLimitWeight: float = 10.0  # https://www.internalfb.com/code/fbsource/arvr/libraries/momentum/character_solver/limit_error_function.h?lines=48
        kPositionWeight: float = 1e-4  # https://www.internalfb.com/code/fbsource/arvr/libraries/momentum/character_solver/limit_error_function.cpp?lines=21

        minmax_min: list[float] = []
        minmax_max: list[float] = []
        minmax_weight: list[float] = []
        minmax_parameter_index: list[int] = []

        minmaxjoint_index: list[int] = []
        minmaxjoint_min: list[float] = []
        minmaxjoint_max: list[float] = []
        minmaxjoint_weight: list[float] = []

        linear_refidx: list[int] = []
        linear_targetidx: list[int] = []
        linear_scale: list[float] = []
        linear_offset: list[float] = []
        linear_weight: list[float] = []
        linear_range_min: list[float | None] = []
        linear_range_max: list[float | None] = []

        linear_joint_refidx: List[int] = []
        linear_joint_targetidx: List[int] = []
        linear_joint_scale: List[float] = []
        linear_joint_offset: List[float] = []
        linear_joint_weight: List[float] = []
        linear_joint_range_min: List[Optional[float]] = []
        linear_joint_range_max: List[Optional[float]] = []

        halfplane_param1_idx: list[int] = []
        halfplane_param2_idx: list[int] = []
        halfplane_normal: list[list[float]] = []
        halfplane_offset: list[float] = []
        halfplane_weight: list[float] = []

        ellipsoid_parent: list[int] = []
        ellipsoid_ellipsoid_parent: list[int] = []
        ellipsoid_offset: list[torch.Tensor] = []
        ellipsoid_ellipsoid: list[torch.Tensor] = []
        ellipsoid_ellipsoid_inv: list[torch.Tensor] = []
        ellipsoid_weight: list[float] = []

        self.has_minmax: bool = False
        self.has_minmaxjoint: bool = False
        self.has_linear: bool = False
        self.has_linear_joint: bool = False
        self.has_halfplane: bool = False
        self.has_ellipsoid: bool = False

        for limit in parameter_limits:
            if limit.type == pym_geometry.LimitType.MinMax:
                self.has_minmax = True
                minmax_min.append(limit.data.minmax.min)
                minmax_max.append(limit.data.minmax.max)
                minmax_parameter_index.append(limit.data.minmax.model_parameter_index)
                minmax_weight.append(limit.weight)
            elif limit.type == pym_geometry.LimitType.MinMaxJoint:
                self.has_minmaxjoint = True
                minmaxjoint_index.append(
                    limit.data.minmax_joint.joint_index
                    * pym_geometry.PARAMETERS_PER_JOINT
                    + limit.data.minmax_joint.joint_parameter_index
                )
                minmaxjoint_min.append(limit.data.minmax_joint.min)
                minmaxjoint_max.append(limit.data.minmax_joint.max)
                minmaxjoint_weight.append(limit.weight)
            elif limit.type == pym_geometry.LimitType.MinMaxJointPassive:
                # It appears these limits are not used in error_function.cpp
                pass
            elif limit.type == pym_geometry.LimitType.Linear:
                self.has_linear = True
                linear_refidx.append(limit.data.linear.reference_model_parameter_index)
                linear_targetidx.append(limit.data.linear.target_model_parameter_index)
                linear_scale.append(limit.data.linear.scale)
                linear_offset.append(limit.data.linear.offset)
                linear_weight.append(limit.weight)
                linear_range_min.append(limit.data.linear.range_min)
                linear_range_max.append(limit.data.linear.range_max)
            elif limit.type == pym_geometry.LimitType.LinearJoint:
                self.has_linear_joint = True
                linear_joint_refidx.append(
                    limit.data.linear_joint.reference_joint_index
                    * pym_geometry.PARAMETERS_PER_JOINT
                    + limit.data.linear_joint.reference_joint_parameter
                )
                linear_joint_targetidx.append(
                    limit.data.linear_joint.target_joint_index
                    * pym_geometry.PARAMETERS_PER_JOINT
                    + limit.data.linear_joint.target_joint_parameter
                )
                linear_joint_scale.append(limit.data.linear_joint.scale)
                linear_joint_offset.append(limit.data.linear_joint.offset)
                linear_joint_weight.append(limit.weight)
                linear_joint_range_min.append(limit.data.linear_joint.range_min)
                linear_joint_range_max.append(limit.data.linear_joint.range_max)
            elif limit.type == pym_geometry.LimitType.HalfPlane:
                self.has_halfplane = True
                halfplane_param1_idx.append(limit.data.halfplane.param1_index)
                halfplane_param2_idx.append(limit.data.halfplane.param2_index)
                halfplane_normal.append(limit.data.halfplane.normal.tolist())
                halfplane_offset.append(limit.data.halfplane.offset)
                halfplane_weight.append(limit.weight)
            elif limit.type == pym_geometry.LimitType.Ellipsoid:
                self.has_ellipsoid = True
                ellipsoid_parent.append(limit.data.ellipsoid.parent)
                ellipsoid_ellipsoid_parent.append(limit.data.ellipsoid.ellipsoid_parent)
                ellipsoid_offset.append(
                    torch.tensor(limit.data.ellipsoid.offset, dtype=dtype)
                )
                ellipsoid_ellipsoid.append(
                    torch.tensor(limit.data.ellipsoid.ellipsoid, dtype=dtype)
                )
                ellipsoid_ellipsoid_inv.append(
                    torch.tensor(limit.data.ellipsoid.ellipsoid_inv, dtype=dtype)
                )
                ellipsoid_weight.append(limit.weight)
                pass

        if self.has_minmax:
            self.register_buffer("minmax_min", torch.tensor(minmax_min, dtype=dtype))
            self.register_buffer("minmax_max", torch.tensor(minmax_max, dtype=dtype))
            self.register_buffer(
                "minmax_weight",
                (kLimitWeight * torch.tensor(minmax_weight, dtype=dtype))
                .sqrt()
                .detach(),
            )
            self.register_buffer(
                "minmax_parameter_index",
                torch.tensor(minmax_parameter_index, dtype=torch.int32),
            )

        if self.has_minmaxjoint:
            self.register_buffer(
                "minmaxjoint_index",
                torch.tensor(minmaxjoint_index, dtype=torch.int32),
            )
            self.register_buffer(
                "minmaxjoint_min",
                torch.tensor(minmaxjoint_min, dtype=dtype),
            )
            self.register_buffer(
                "minmaxjoint_max",
                torch.tensor(minmaxjoint_max, dtype=dtype),
            )
            self.register_buffer(
                "minmaxjoint_weight",
                (kLimitWeight * torch.tensor(minmaxjoint_weight, dtype=dtype))
                .sqrt()
                .detach(),
            )

        if self.has_linear:
            self.register_buffer(
                "linear_refidx", torch.tensor(linear_refidx, dtype=torch.int32)
            )
            self.register_buffer(
                "linear_targetidx",
                torch.tensor(linear_targetidx, dtype=torch.int32),
            )
            self.register_buffer(
                "linear_scale", torch.tensor(linear_scale, dtype=dtype)
            )
            self.register_buffer(
                "linear_offset",
                torch.tensor(linear_offset, dtype=dtype),
            )
            self.register_buffer(
                "linear_weight",
                (kLimitWeight * torch.tensor(linear_weight, dtype=dtype))
                .sqrt()
                .detach(),
            )
            self.register_buffer(
                "linear_range_min",
                torch.tensor(
                    [-float("inf") if x is None else x for x in linear_range_min],
                    dtype=dtype,
                ),
            )
            self.register_buffer(
                "linear_range_max",
                torch.tensor(
                    [float("inf") if x is None else x for x in linear_range_max],
                    dtype=dtype,
                ),
            )

        if self.has_linear_joint:
            self.register_buffer(
                "linear_joint_refidx",
                torch.tensor(linear_joint_refidx, dtype=torch.int32),
            )
            self.register_buffer(
                "linear_joint_targetidx",
                torch.tensor(linear_joint_targetidx, dtype=torch.int32),
            )
            self.register_buffer(
                "linear_joint_scale", torch.tensor(linear_joint_scale, dtype=dtype)
            )
            self.register_buffer(
                "linear_joint_offset",
                torch.tensor(linear_joint_offset, dtype=dtype),
            )
            self.register_buffer(
                "linear_joint_weight",
                (kLimitWeight * torch.tensor(linear_joint_weight, dtype=dtype))
                .sqrt()
                .detach(),
            )
            self.register_buffer(
                "linear_joint_range_min",
                torch.tensor(
                    [-float("inf") if x is None else x for x in linear_joint_range_min],
                    dtype=dtype,
                ),
            )
            self.register_buffer(
                "linear_joint_range_max",
                torch.tensor(
                    [float("inf") if x is None else x for x in linear_joint_range_max],
                    dtype=dtype,
                ),
            )

        if self.has_halfplane:
            self.register_buffer(
                "halfplane_param1_idx",
                torch.tensor(halfplane_param1_idx, dtype=torch.int32),
            )
            self.register_buffer(
                "halfplane_param2_idx",
                torch.tensor(halfplane_param2_idx, dtype=torch.int32),
            )
            self.register_buffer(
                "halfplane_normal",
                torch.tensor(halfplane_normal, dtype=dtype),
            )
            self.register_buffer(
                "halfplane_offset", torch.tensor(halfplane_offset, dtype=dtype)
            )
            self.register_buffer(
                "halfplane_weight",
                (kLimitWeight * torch.tensor(halfplane_weight, dtype=dtype))
                .sqrt()
                .detach(),
            )

        if self.has_ellipsoid:
            self.register_buffer(
                "ellipsoid_parent", torch.tensor(ellipsoid_parent, dtype=torch.int32)
            )
            self.register_buffer(
                "ellipsoid_ellipsoid_parent",
                torch.tensor(ellipsoid_ellipsoid_parent, dtype=torch.int32),
            )
            self.register_buffer(
                "ellipsoid_offset", torch.stack(ellipsoid_offset, dim=0)
            )
            self.register_buffer(
                "ellipsoid_ellipsoid", torch.stack(ellipsoid_ellipsoid, dim=0)
            )
            self.register_buffer(
                "ellipsoid_ellipsoid_inv", torch.stack(ellipsoid_ellipsoid_inv, dim=0)
            )
            self.register_buffer(
                "ellipsoid_weight",
                (
                    kLimitWeight
                    * kPositionWeight
                    * torch.tensor(ellipsoid_weight, dtype=dtype)
                )
                .sqrt()
                .detach(),
            )

    def evaluate_minmax_error(self, model_params: torch.Tensor) -> torch.Tensor:
        if not self.has_minmax:
            return torch.zeros(
                (model_params.shape[0], 0),
                dtype=model_params.dtype,
                device=model_params.device,
            )

        selected_model_params = model_params[..., self.minmax_parameter_index]

        return self.minmax_weight * (
            torch.where(
                selected_model_params < self.minmax_min,
                self.minmax_min - selected_model_params,
                torch.zeros_like(selected_model_params),
            )
            + torch.where(
                selected_model_params > self.minmax_max,
                selected_model_params - self.minmax_max,
                torch.zeros_like(selected_model_params),
            )
        )

    def evaluate_minmaxjoint_error(self, joint_params: torch.Tensor) -> torch.Tensor:
        if not self.has_minmaxjoint:
            return torch.zeros(
                (joint_params.shape[0], 0),
                dtype=joint_params.dtype,
                device=joint_params.device,
            )

        selected_joint_params = _squeeze_joint_params(joint_params)[
            ..., self.minmaxjoint_index
        ]

        return self.minmaxjoint_weight * (
            torch.where(
                selected_joint_params < self.minmaxjoint_min,
                self.minmaxjoint_min - selected_joint_params,
                torch.zeros_like(selected_joint_params),
            )
            + torch.where(
                selected_joint_params > self.minmaxjoint_max,
                selected_joint_params - self.minmaxjoint_max,
                torch.zeros_like(selected_joint_params),
            )
        )

    def evaluate_linear_error(self, model_params: torch.Tensor) -> torch.Tensor:
        if not self.has_linear:
            return torch.zeros(
                (model_params.shape[0], 0),
                dtype=model_params.dtype,
                device=model_params.device,
            )

        ref_params = model_params[:, self.linear_refidx]
        target_params = model_params[:, self.linear_targetidx]

        is_in_range = torch.logical_and(
            target_params >= self.linear_range_min,
            target_params < self.linear_range_max,
        )

        linear_res = self.linear_weight * (
            self.linear_scale * target_params - self.linear_offset - ref_params
        )

        return torch.where(is_in_range, linear_res, torch.zeros_like(linear_res))

    def evaluate_linear_joint_error(self, joint_params: torch.Tensor) -> torch.Tensor:
        if not self.has_linear_joint:
            return torch.zeros(
                (joint_params.shape[0], 0),
                dtype=joint_params.dtype,
                device=joint_params.device,
            )

        ref_joint_params = joint_params[:, self.linear_joint_refidx]
        target_joint_params = joint_params[:, self.linear_joint_targetidx]

        is_in_range = torch.logical_and(
            target_joint_params >= self.linear_joint_range_min,
            target_joint_params < self.linear_joint_range_max,
        )

        linear_res = self.linear_joint_weight * (
            self.linear_joint_scale * target_joint_params
            - self.linear_joint_offset
            - ref_joint_params
        )

        return torch.where(is_in_range, linear_res, torch.zeros_like(linear_res))

    def evaluate_halfplane_error(self, model_params: torch.Tensor) -> torch.Tensor:
        if not self.has_halfplane:
            return torch.zeros(
                (model_params.shape[0], 0),
                dtype=model_params.dtype,
                device=model_params.device,
            )

        # n_batch x n_constraints
        param1 = model_params[:, self.halfplane_param1_idx]
        param2 = model_params[:, self.halfplane_param2_idx]

        fn_val = (
            param1 * self.halfplane_normal[None, :, 0]
            + param2 * self.halfplane_normal[None, :, 1]
            - self.halfplane_offset[None, :]
        )

        return self.halfplane_weight * torch.where(
            fn_val < 0, fn_val, torch.zeros_like(fn_val)
        )

    def _transform_points(
        self, transforms: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        return (
            torch.einsum("...ij,...j->...i", transforms[:, :, 0:3, 0:3], points)
            + transforms[:, :, 0:3, 3]
        )

    def evaluate_ellipsoid_error(self, skel_state: torch.Tensor) -> torch.Tensor:
        if not self.has_ellipsoid:
            return torch.zeros(
                (skel_state.shape[0], 0),
                dtype=skel_state.dtype,
                device=skel_state.device,
            )

        skel_state_inv = pym_skel_state.inverse(skel_state)

        n_batch = skel_state.shape[0]
        n_constraints = self.ellipsoid_parent.shape[0]

        # Following the code here: https://www.internalfb.com/code/fbsource/arvr/libraries/momentum/character_solver/limit_error_function.cpp?lines=433

        # get the constraint position in global space
        position: torch.Tensor = pym_skel_state.transform_points(
            skel_state[:, self.ellipsoid_parent],
            self.ellipsoid_offset[None, ...].expand(n_batch, n_constraints, 3),
        )
        # get the constraint position in local ellipsoid space
        local_position: torch.Tensor = pym_skel_state.transform_points(
            skel_state_inv[:, self.ellipsoid_ellipsoid_parent], position
        )
        # calculate constraint position in ellipsoid space
        ellipsoid_position: torch.Tensor = self._transform_points(
            self.ellipsoid_ellipsoid_inv[None, :, :], local_position
        )
        # project onto closest surface point
        normalized_position = ellipsoid_position / torch.norm(
            ellipsoid_position, dim=-1, keepdim=True
        )
        # go back to ellipsoid frame
        projected_position = self._transform_points(
            self.ellipsoid_ellipsoid[None, :, :], normalized_position
        )
        # calculate the difference between projected position and actual position
        diff = position - pym_skel_state.transform_points(
            skel_state[:, self.ellipsoid_ellipsoid_parent], projected_position
        )
        result = (self.ellipsoid_weight[None, :, None] * diff).flatten(-2)
        return result

    def forward(
        self,
        model_parameters: torch.Tensor,
        joint_parameters: torch.Tensor,
        skel_state: torch.Tensor,
    ) -> torch.Tensor:
        minmax_error = self.evaluate_minmax_error(model_parameters)
        minmaxjoint_error = self.evaluate_minmaxjoint_error(joint_parameters)
        linear_error = self.evaluate_linear_error(model_parameters)
        linear_joint_error = self.evaluate_linear_joint_error(joint_parameters)
        halfplane_error = self.evaluate_halfplane_error(model_parameters)
        ellipsoid_error = self.evaluate_ellipsoid_error(skel_state)

        return torch.cat(
            [
                minmax_error,
                minmaxjoint_error,
                linear_error,
                linear_joint_error,
                halfplane_error,
                ellipsoid_error,
            ],
            -1,
        )
