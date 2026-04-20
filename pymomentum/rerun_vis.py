# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Rerun Visualization Helpers
============================

Helper functions for visualizing momentum characters using the
`Rerun <https://www.rerun.io/>`_ viewer, providing the Python equivalent of
the C++ ``momentum::logCharacter`` family of functions in
``momentum/gui/rerun/logger.h``.

The primary entry point is :func:`log_character`, which logs the skeleton,
mesh, and locators of a character in a structured entity tree::

    import rerun as rr
    import pymomentum.geometry as geo
    from pymomentum.rerun_vis import log_character

    character = geo.Character.load_gltf("character.glb")
    char, motion, identity, fps = geo.Character.load_gltf_with_motion("motion.glb")
    skel_state = geo.joint_parameters_to_skeleton_state(char, motion[0])
    posed_mesh = char.pose_mesh(motion[0])

    rec = rr.init("pymomentum_viewer", spawn=True)
    log_character(rec, "character", char, skel_state, posed_mesh=posed_mesh)

Individual logging functions are also available for finer control:

- :func:`log_mesh` — log a :class:`~pymomentum.geometry.Mesh`
- :func:`log_joints` — log skeleton bones and per-joint transforms
- :func:`log_locators` — log locator world-space positions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymomentum.quaternion_np import (
    rotate_vector_assume_normalized,
    to_rotation_matrix_assume_normalized,
)

if TYPE_CHECKING:
    import rerun as rr
    from pymomentum.geometry import Character, Mesh  # @manual=:geometry


def log_mesh(
    rec: rr.RecordingStream,
    entity_path: str,
    mesh: Mesh,
    *,
    color: tuple[int, int, int] | None = None,
) -> None:
    """Log a :class:`~pymomentum.geometry.Mesh` to Rerun as a 3D mesh.

    :param rec: The Rerun recording stream.
    :param entity_path: Entity path in the Rerun viewer.
    :param mesh: The mesh to log (typically a posed mesh from
        :meth:`~pymomentum.geometry.Character.pose_mesh`).
    :param color: Optional RGB color tuple ``(r, g, b)`` applied as an
        albedo factor.  If *None*, per-vertex colors from the mesh are used
        when available.
    """
    import rerun as rr

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    kwargs: dict[str, object] = {
        "vertex_positions": vertices,
        "triangle_indices": faces,
    }

    normals = np.asarray(mesh.normals, dtype=np.float32)
    if normals.size > 0:
        kwargs["vertex_normals"] = normals

    if color is not None:
        kwargs["albedo_factor"] = rr.AlbedoFactor(color)
    else:
        colors = np.asarray(mesh.colors, dtype=np.uint8)
        if colors.size > 0:
            kwargs["vertex_colors"] = colors

    rec.log(entity_path, rr.Mesh3D(**kwargs))


def log_joints(
    rec: rr.RecordingStream,
    entity_path: str,
    character: Character,
    skel_state: np.ndarray,
) -> None:
    """Log skeleton bones and per-joint transforms to Rerun.

    Bones are drawn as grey :class:`rerun.LineStrips3D` connecting parent–child
    joint pairs.  Each joint also gets a :class:`rerun.Transform3D` logged as a
    child entity so that the Rerun viewer can display its local coordinate
    frame.

    :param rec: The Rerun recording stream.
    :param entity_path: Entity path for the skeleton entity.
    :param character: The character whose skeleton is logged.
    :param skel_state: Skeleton state array of shape ``(n_joints, 8)`` with
        per-joint ``[tx, ty, tz, rx, ry, rz, rw, scale]``.
    """
    import rerun as rr

    skeleton = character.skeleton
    joint_names: list[str] = skeleton.joint_names
    joint_parents: list[int] = list(skeleton.joint_parents)
    n_joints: int = len(joint_names)

    lines: list[list[list[float]]] = []
    labels: list[str] = []

    for i in range(n_joints):
        parent_idx = joint_parents[i]
        if 0 <= parent_idx < n_joints:
            parent_pos: list[float] = skel_state[parent_idx, :3].tolist()
            child_pos: list[float] = skel_state[i, :3].tolist()
            lines.append([parent_pos, child_pos])
            labels.append(joint_names[i])

        # Log per-joint transform
        rot_mat = to_rotation_matrix_assume_normalized(skel_state[i, 3:7])
        translation = skel_state[i, :3]
        rec.log(
            f"{entity_path}/{joint_names[i]}",
            rr.Transform3D(
                translation=translation,
                mat3x3=rot_mat,
            ),
        )

    if lines:
        rec.log(
            entity_path,
            rr.LineStrips3D(
                lines,
                radii=0.2,
                colors=(180, 180, 180),
                labels=labels,
            ),
        )


def log_locators(
    rec: rr.RecordingStream,
    entity_path: str,
    character: Character,
    skel_state: np.ndarray,
) -> None:
    """Log locator positions to Rerun as colored 3D points.

    Locators whose name contains ``"Floor_"`` are shown in green; others are
    shown in a grey whose intensity is proportional to the locator weight.

    :param rec: The Rerun recording stream.
    :param entity_path: Entity path for the locators entity.
    :param character: The character whose locators are logged.
    :param skel_state: Skeleton state array of shape ``(n_joints, 8)``.
    """
    import rerun as rr

    locators = character.locators
    if not locators:
        return

    positions: list[list[float]] = []
    colors: list[tuple[int, int, int]] = []
    labels: list[str] = []

    for loc in locators:
        parent_idx: int = loc.parent
        q = skel_state[parent_idx, 3:7]
        t = skel_state[parent_idx, :3]
        s = float(skel_state[parent_idx, 7])
        offset = np.asarray(loc.offset, dtype=np.float32)
        world_pos: np.ndarray = t + rotate_vector_assume_normalized(q, offset * s)

        positions.append(world_pos.tolist())
        labels.append(loc.name)

        if "Floor_" in loc.name:
            colors.append((100, 255, 100))
        else:
            intensity = int(255.0 * loc.weight * 0.6)
            intensity = max(0, min(255, intensity))
            colors.append(
                (intensity, intensity, max(0, min(255, int(intensity * 0.5))))
            )

    rec.log(
        entity_path,
        rr.Points3D(
            positions=positions,
            radii=0.5,
            colors=colors,
            labels=labels,
        ),
    )


def log_character(
    rec: rr.RecordingStream,
    entity_path: str,
    character: Character,
    skel_state: np.ndarray,
    *,
    posed_mesh: Mesh | None = None,
    color: tuple[int, int, int] = (200, 200, 200),
) -> None:
    """Log a complete character visualization to Rerun.

    This is the top-level convenience function that logs the skeleton,
    optionally the mesh, and locators as a structured entity tree:

    - ``{entity_path}/mesh`` — posed mesh (if *posed_mesh* is provided)
    - ``{entity_path}/joints`` — skeleton bones and per-joint transforms
    - ``{entity_path}/locators`` — locator positions (if character has locators)

    This is the Python equivalent of the C++ ``momentum::logCharacter`` function.

    :param rec: The Rerun recording stream.
    :param entity_path: Root entity path for the character.
    :param character: The character to visualize.
    :param skel_state: Skeleton state array of shape ``(n_joints, 8)`` with
        per-joint ``[tx, ty, tz, rx, ry, rz, rw, scale]``.  Obtain via
        :func:`~pymomentum.geometry.joint_parameters_to_skeleton_state` or
        :func:`~pymomentum.geometry.model_parameters_to_skeleton_state`.
    :param posed_mesh: An optional pre-posed mesh (from
        :meth:`~pymomentum.geometry.Character.pose_mesh`).  If *None*, no mesh
        is logged.
    :param color: RGB color tuple for the mesh albedo.  Defaults to light grey
        ``(200, 200, 200)``.
    """
    if posed_mesh is not None:
        log_mesh(rec, f"{entity_path}/mesh", posed_mesh, color=color)

    log_joints(rec, f"{entity_path}/joints", character, skel_state)

    if character.locators:
        log_locators(rec, f"{entity_path}/locators", character, skel_state)
