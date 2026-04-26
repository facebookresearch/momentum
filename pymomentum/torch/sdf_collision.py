# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Pure PyTorch SDF collision geometry for torch_character.

Provides differentiable SDF evaluation matching the behavior of
:class:`momentum.SDFColliderT`. All operations use pure PyTorch ops,
enabling gradient computation through autograd including higher-order
gradients (gradient-of-gradient).

Typical usage::

    import pymomentum.geometry as pym_geometry
    from pymomentum.torch.sdf_collision import SDFCollider, SDFCollisionGeometry

    # Create from pymomentum colliders
    pym_colliders = [pym_geometry.SDFCollider(sdf=sdf, parent=joint_idx)]
    collision_geo = SDFCollisionGeometry.from_pymomentum(pym_colliders)

    # Evaluate: skel_state from Skeleton.forward(), points in world space
    distances = collision_geo.evaluate(skel_state, world_points)
"""

from __future__ import annotations

import numpy as np
import pymomentum.quaternion as pym_quaternion
import torch


def _trilinear_interpolate(
    sdf_data: torch.Tensor,
    grid_coords: torch.Tensor,
) -> torch.Tensor:
    """Trilinear interpolation of a 3D SDF grid in pure PyTorch.

    Matches the interpolation in ``axel::SignedDistanceField::sample()``.
    Grid coordinates must already be clamped to ``[0, res-1]``.

    :param sdf_data: 3D SDF grid values, shape ``(nz, ny, nx)`` (DHW order).
        ``sdf_data[k, j, i]`` is the signed distance at grid point
        ``(i, j, k)``.  This layout matches ``F.grid_sample`` input
        convention and avoids a permute when using it.
    :param grid_coords: Clamped continuous grid coordinates, shape ``(..., 3)``
        as ``(gx, gy, gz)`` where each component is in ``[0, res-1]``.
    :return: Interpolated SDF values, shape ``(...)``.
    """
    nz, ny, nx = sdf_data.shape
    gx = grid_coords[..., 0]
    gy = grid_coords[..., 1]
    gz = grid_coords[..., 2]

    # Integer lower-corner indices (detach so floor doesn't affect gradient graph)
    i0 = gx.detach().floor().long().clamp(0, nx - 1)
    j0 = gy.detach().floor().long().clamp(0, ny - 1)
    k0 = gz.detach().floor().long().clamp(0, nz - 1)

    # Upper corner indices (clamped to grid bounds)
    i1 = (i0 + 1).clamp(max=nx - 1)
    j1 = (j0 + 1).clamp(max=ny - 1)
    k1 = (k0 + 1).clamp(max=nz - 1)

    # Fractional interpolation weights (differentiable w.r.t. grid_coords)
    fx = gx - i0.to(gx.dtype)
    fy = gy - j0.to(gy.dtype)
    fz = gz - k0.to(gz.dtype)

    # Gather 8 corner values — stored as (nz, ny, nx) so index [k, j, i]
    c000 = sdf_data[k0, j0, i0]
    c100 = sdf_data[k0, j0, i1]
    c010 = sdf_data[k0, j1, i0]
    c110 = sdf_data[k0, j1, i1]
    c001 = sdf_data[k1, j0, i0]
    c101 = sdf_data[k1, j0, i1]
    c011 = sdf_data[k1, j1, i0]
    c111 = sdf_data[k1, j1, i1]

    # Blend along x, then y, then z (same order as C++ sample())
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx

    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    return c0 * (1 - fz) + c1 * fz


def evaluate_sdf(
    sdf_data: torch.Tensor,
    bounds_min: torch.Tensor,
    voxel_size: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Evaluate SDF distances at query points in SDF local space.

    Performs world-to-grid conversion, boundary clamping, trilinear
    interpolation, and out-of-bounds distance computation, matching
    ``axel::SignedDistanceField::sample()``.

    All operations are differentiable via PyTorch autograd. For out-of-bounds
    points, the Euclidean distance to the nearest grid boundary is added to
    the interpolated boundary value, providing correct gradient behavior:

    - Inside the grid: gradient comes from trilinear interpolation derivatives.
    - Outside the grid: gradient points away from the boundary (increasing
      distance).

    :param sdf_data: 3D SDF grid values, shape ``(nz, ny, nx)`` (DHW order).
    :param bounds_min: SDF bounding box minimum corner, shape ``(3,)``.
    :param voxel_size: Size of each voxel in world units, shape ``(3,)``.
    :param points: Query points in SDF local space, shape ``(..., 3)``.
    :return: Signed distance values, shape ``(...)``.
    """
    nz, ny, nx = sdf_data.shape

    # Convert world-space positions to continuous grid coordinates
    grid_coords = (points - bounds_min) / voxel_size

    # Clamp to valid grid range [0, res-1] for interpolation
    gx_clamped = grid_coords[..., 0].clamp(0.0, nx - 1)
    gy_clamped = grid_coords[..., 1].clamp(0.0, ny - 1)
    gz_clamped = grid_coords[..., 2].clamp(0.0, nz - 1)
    clamped = torch.stack([gx_clamped, gy_clamped, gz_clamped], dim=-1)

    # Trilinear interpolation at clamped grid position
    value = _trilinear_interpolate(sdf_data, clamped)

    # For out-of-bounds points, add distance from query point to nearest
    # grid boundary. Inside the grid this is exactly zero.
    clamped_world = clamped * voxel_size + bounds_min
    offset_dist = (points - clamped_world).norm(dim=-1)

    return value + offset_dist


class SDFCollider(torch.nn.Module):
    """A signed distance field volume attached to a skeleton joint.

    Mirrors :class:`momentum.SDFColliderT` with evaluation implemented
    in pure PyTorch for full autograd differentiability.

    The SDF data is stored as a 3D tensor in ``(nz, ny, nx)`` (DHW) order:
    ``sdf_data[k, j, i]`` is the signed distance at grid point
    ``(i, j, k)``.  This layout is compatible with ``F.grid_sample``.

    :param sdf_data: 3D SDF grid values, shape ``(nz, ny, nx)`` (DHW order).
    :param bounds_min: SDF bounding box minimum corner, shape ``(3,)``.
    :param voxel_size: Voxel dimensions in world units, shape ``(3,)``.
    :param local_translation: Translation offset from parent joint, shape ``(3,)``.
    :param local_rotation: Rotation matrix relative to parent joint, shape ``(3, 3)``.
    :param parent_joint: Parent joint index. Use ``-1`` for world-space SDFs.
    """

    parent_joint: int

    def __init__(
        self,
        sdf_data: torch.Tensor,
        bounds_min: torch.Tensor,
        voxel_size: torch.Tensor,
        local_translation: torch.Tensor,
        local_rotation: torch.Tensor,
        parent_joint: int = -1,
    ) -> None:
        super().__init__()
        self.register_buffer("sdf_data", sdf_data)
        self.register_buffer("bounds_min", bounds_min)
        self.register_buffer("voxel_size", voxel_size)
        self.register_buffer("local_translation", local_translation)
        self.register_buffer("local_rotation", local_rotation)
        self.parent_joint = parent_joint

    @staticmethod
    def from_pymomentum(
        collider: object,
        dtype: torch.dtype = torch.float32,
    ) -> SDFCollider:
        """Create an :class:`SDFCollider` from a :class:`pymomentum.geometry.SDFCollider`.

        Converts the SDF grid data to a ``(nz, ny, nx)`` PyTorch tensor
        (DHW order, ready for ``F.grid_sample``) and the collider's local
        transform to a rotation matrix.

        :param collider: A :class:`pymomentum.geometry.SDFCollider` instance.
        :param dtype: Tensor dtype (default: ``torch.float32``).
        :return: A new :class:`SDFCollider` module.
        """
        sdf = collider.sdf  # pyre-ignore[16]
        res = sdf.resolution
        nx, ny, nz = int(res[0]), int(res[1]), int(res[2])

        # The buffer protocol exposes a Fortran-order (nx, ny, nz) view
        # matching the C++ linearIndex(i,j,k) = k*nx*ny + j*nx + i storage.
        # ravel(order='F') gives the flat data in memory order (x fastest).
        # Reshaping to (nz, ny, nx) C-contiguous gives tensor[k,j,i] ==
        # C++ at(i,j,k), which is the DHW layout for F.grid_sample.
        sdf_data = torch.tensor(np.asarray(sdf).ravel(order="F"), dtype=dtype).reshape(
            nz, ny, nx
        )

        bounds_min = torch.tensor(np.asarray(sdf.bounds.min), dtype=dtype)
        voxel_size = torch.tensor(np.asarray(sdf.voxel_size), dtype=dtype)

        local_translation = torch.tensor(
            np.asarray(collider.translation),  # pyre-ignore[16]
            dtype=dtype,
        )
        rotation_quat = torch.tensor(
            np.asarray(collider.rotation),  # pyre-ignore[16]
            dtype=dtype,
        )
        local_rotation: torch.Tensor = pym_quaternion.to_rotation_matrix(
            rotation_quat.unsqueeze(0)
        ).squeeze(0)

        return SDFCollider(
            sdf_data=sdf_data,
            bounds_min=bounds_min,
            voxel_size=voxel_size,
            local_translation=local_translation,
            local_rotation=local_rotation,
            parent_joint=collider.parent,  # pyre-ignore[16]
        )

    def evaluate(
        self,
        skel_state: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate SDF distances for world-space query points.

        Transforms ``points`` from world space into the SDF's local coordinate
        system using the joint transform from ``skel_state``, then evaluates
        via trilinear interpolation.

        :param skel_state: Global skeleton state from forward kinematics,
            shape ``(..., n_joints, 8)`` with per-joint format
            ``[tx, ty, tz, qx, qy, qz, qw, scale]``.
            Ignored when ``parent_joint == -1`` (world-space SDF).
        :param points: World-space query points, shape ``(..., n_points, 3)``.
        :return: SDF distances at each query point, shape ``(..., n_points)``.
        """
        if self.parent_joint < 0:
            # World-space SDF: local transform IS the world transform
            R_world = self.local_rotation
            t_world = self.local_translation
        else:
            # Joint-space SDF: compose joint transform with local transform
            joint_t = skel_state[..., self.parent_joint, 0:3]
            joint_q = skel_state[..., self.parent_joint, 3:7]
            joint_R = pym_quaternion.to_rotation_matrix(joint_q)

            # R_world = R_joint @ R_local
            R_world = joint_R @ self.local_rotation
            # t_world = R_joint @ t_local + t_joint
            t_world = (
                torch.einsum("...ij,j->...i", joint_R, self.local_translation) + joint_t
            )

        # Transform points to SDF local space:
        #   p_local = R_world^T @ (p_world - t_world)
        # In row-vector form: (p - t) @ R  (since p @ R == R^T @ p)
        p_centered = points - t_world[..., None, :]
        p_local = torch.einsum("...pi,...ij->...pj", p_centered, R_world)

        return evaluate_sdf(
            self.sdf_data,
            self.bounds_min,
            self.voxel_size,
            p_local,
        )


class SDFCollisionGeometry(torch.nn.Module):
    """Collection of SDF colliders for character collision detection.

    Mirrors :class:`momentum.SDFCollisionGeometryT` — a list of
    :class:`SDFCollider` modules attached to skeleton joints.

    :param colliders: List of :class:`SDFCollider` modules.
    """

    def __init__(self, colliders: list[SDFCollider]) -> None:
        super().__init__()
        self.colliders: torch.nn.ModuleList = torch.nn.ModuleList(colliders)

    @staticmethod
    def from_pymomentum(
        colliders: list[object],
        dtype: torch.dtype = torch.float32,
    ) -> SDFCollisionGeometry:
        """Create from a list of :class:`pymomentum.geometry.SDFCollider` objects.

        :param colliders: List of :class:`pymomentum.geometry.SDFCollider` instances.
        :param dtype: Tensor dtype (default: ``torch.float32``).
        :return: A new :class:`SDFCollisionGeometry` module.
        """
        return SDFCollisionGeometry(
            [SDFCollider.from_pymomentum(c, dtype=dtype) for c in colliders]
        )

    def evaluate(
        self,
        skel_state: torch.Tensor,
        points: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Evaluate all colliders and return per-collider SDF distances.

        :param skel_state: Global skeleton state, shape ``(..., n_joints, 8)``.
        :param points: World-space query points, shape ``(..., n_points, 3)``.
        :return: List of distance tensors, one per collider, each shape
            ``(..., n_points)``.
        """
        return [c.evaluate(skel_state, points) for c in self.colliders]

    def evaluate_min(
        self,
        skel_state: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Minimum SDF distance across all colliders for each point.

        Useful for collision detection: negative values indicate penetration
        into at least one collider.

        :param skel_state: Global skeleton state, shape ``(..., n_joints, 8)``.
        :param points: World-space query points, shape ``(..., n_points, 3)``.
        :return: Minimum signed distance per point, shape ``(..., n_points)``.
        """
        distances = self.evaluate(skel_state, points)
        return torch.stack(distances, dim=0).min(dim=0).values
