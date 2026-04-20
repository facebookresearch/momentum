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

For multi-frame animation, use :func:`log_animation` which sends all frames
using :func:`rerun.send_columns` for dramatically better throughput::

    skel_states = geo.joint_parameters_to_skeleton_state(char, joint_params)
    posed_meshes = [char.pose_mesh(jp) for jp in joint_params]
    log_animation(rec, "character", char, skel_states,
                  posed_meshes=posed_meshes, fps=fps)

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
    from collections.abc import Sequence

    import rerun as rr  # @manual=fbsource//third-party/pypi/rerun-sdk:rerun-sdk
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
    n_joints: int = len(joint_names)
    parents = np.array(skeleton.joint_parents, dtype=np.intp)

    translations = skel_state[:, :3]
    rot_mats = to_rotation_matrix_assume_normalized(skel_state[:, 3:7])

    valid = (parents >= 0) & (parents < n_joints)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) > 0:
        lines = np.stack(
            [translations[parents[valid_indices]], translations[valid_indices]], axis=1
        )
        labels = [joint_names[i] for i in valid_indices]
        rec.log(
            entity_path,
            rr.LineStrips3D(
                lines,
                radii=0.2,
                colors=(180, 180, 180),
                labels=labels,
            ),
        )

    for i in range(n_joints):
        rec.log(
            f"{entity_path}/{joint_names[i]}",
            rr.Transform3D(
                translation=translations[i],
                mat3x3=rot_mats[i],
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

    n_locs = len(locators)
    parent_indices = np.array([loc.parent for loc in locators], dtype=np.intp)
    offsets = np.array([loc.offset for loc in locators], dtype=np.float32)

    q = skel_state[parent_indices, 3:7]
    t = skel_state[parent_indices, :3]
    s = skel_state[parent_indices, 7:8]
    positions = t + rotate_vector_assume_normalized(q, offsets * s)

    colors = np.empty((n_locs, 3), dtype=np.uint8)
    labels: list[str] = []
    for i, loc in enumerate(locators):
        labels.append(loc.name)
        if "Floor_" in loc.name:
            colors[i] = (100, 255, 100)
        else:
            intensity = int(255.0 * loc.weight * 0.6)
            intensity = max(0, min(255, intensity))
            colors[i] = (intensity, intensity, max(0, min(255, int(intensity * 0.5))))

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


def _make_time_columns(
    n_frames: int,
    fps: float,
    frame_offset: int,
) -> list:
    """Return ``[TimeColumn("frame_index", …), TimeColumn("time", …)]``."""
    import rerun as rr

    frame_indices = np.arange(frame_offset, frame_offset + n_frames)
    return [
        rr.TimeColumn("frame_index", sequence=frame_indices),
        rr.TimeColumn("time", duration=frame_indices / fps),
    ]


def _send_mesh_columns(
    rec: rr.RecordingStream,
    entity_path: str,
    posed_meshes: Sequence[Mesh],
    time_cols: list,
    color: tuple[int, int, int] | None,
    fps: float,
    frame_offset: int,
) -> None:
    """Send mesh vertex data for multiple frames via :meth:`rec.send_columns`."""
    import rerun as rr

    n_frames = len(posed_meshes)
    if n_frames == 0:
        return

    first = posed_meshes[0]
    vertices = np.asarray(first.vertices, dtype=np.float32)
    n_verts = vertices.shape[0]
    faces = np.asarray(first.faces, dtype=np.uint32)

    # Log a complete mesh at the first time point to establish constant
    # components (triangle_indices, albedo_factor).  Rerun's "latest at"
    # query makes them persist for subsequent frames.
    rec.set_time("frame_index", sequence=frame_offset)
    rec.set_time("time", duration=frame_offset / fps)
    first_kwargs: dict[str, object] = {
        "vertex_positions": vertices,
        "triangle_indices": faces,
    }
    normals = np.asarray(first.normals, dtype=np.float32)
    if normals.size > 0:
        first_kwargs["vertex_normals"] = normals
    if color is not None:
        first_kwargs["albedo_factor"] = rr.AlbedoFactor(color)
    else:
        first_colors = np.asarray(first.colors, dtype=np.uint8)
        if first_colors.size > 0:
            first_kwargs["vertex_colors"] = first_colors
    rec.log(entity_path, rr.Mesh3D(**first_kwargs))
    rec.reset_time()

    if n_frames <= 1:
        return

    # Send dynamic vertex positions (and normals) for remaining frames.
    remaining_time = _make_time_columns(n_frames - 1, fps, frame_offset + 1)
    cols_kwargs: dict[str, object] = {
        "vertex_positions": np.concatenate(
            [np.asarray(m.vertices, dtype=np.float32) for m in posed_meshes[1:]]
        ),
    }
    if normals.size > 0:
        cols_kwargs["vertex_normals"] = np.concatenate(
            [np.asarray(m.normals, dtype=np.float32) for m in posed_meshes[1:]]
        )

    mesh_cols = rr.Mesh3D.columns(**cols_kwargs)
    mesh_cols = mesh_cols.partition(np.full(n_frames - 1, n_verts))
    rec.send_columns(entity_path, indexes=remaining_time, columns=mesh_cols)


def _send_joints_columns(
    rec: rr.RecordingStream,
    entity_path: str,
    character: Character,
    skel_states: np.ndarray,
    time_cols: list,
    fps: float,
    frame_offset: int,
) -> None:
    """Send skeleton bones and per-joint transforms via :meth:`rec.send_columns`."""
    import rerun as rr

    skeleton = character.skeleton
    joint_names: list[str] = skeleton.joint_names
    n_joints = len(joint_names)
    n_frames = skel_states.shape[0]
    parents = np.array(skeleton.joint_parents, dtype=np.intp)

    translations = skel_states[:, :, :3]

    valid = (parents >= 0) & (parents < n_joints)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) > 0:
        n_bones = len(valid_indices)
        parent_pos = translations[:, parents[valid_indices], :]
        child_pos = translations[:, valid_indices, :]
        all_lines = np.stack([parent_pos, child_pos], axis=2).reshape(-1, 2, 3)

        # Radii and colors are constant; send once at the first time point.
        first_time = _make_time_columns(1, fps, frame_offset)
        static_cols = rr.LineStrips3D.columns(
            radii=np.full(n_bones, 0.2, dtype=np.float32),
            colors=np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (n_bones, 1)),
        )
        static_cols = static_cols.partition(np.array([n_bones]))
        rec.send_columns(entity_path, indexes=first_time, columns=static_cols)

        line_cols = rr.LineStrips3D.columns(strips=all_lines)
        line_cols = line_cols.partition(np.full(n_frames, n_bones))
        rec.send_columns(entity_path, indexes=time_cols, columns=line_cols)

    rot_mats = to_rotation_matrix_assume_normalized(
        skel_states[:, :, 3:7].reshape(-1, 4)
    ).reshape(n_frames, n_joints, 3, 3)

    for j in range(n_joints):
        transform_cols = rr.Transform3D.columns(
            translation=translations[:, j, :],
            mat3x3=rot_mats[:, j, :, :],
        )
        rec.send_columns(
            f"{entity_path}/{joint_names[j]}",
            indexes=time_cols,
            columns=transform_cols,
        )


def _send_locators_columns(
    rec: rr.RecordingStream,
    entity_path: str,
    character: Character,
    skel_states: np.ndarray,
    time_cols: list,
    fps: float,
    frame_offset: int,
) -> None:
    """Send locator positions for multiple frames via :meth:`rec.send_columns`."""
    import rerun as rr

    locators = character.locators
    if not locators:
        return

    n_locs = len(locators)
    n_frames = skel_states.shape[0]
    parent_indices = np.array([loc.parent for loc in locators], dtype=np.intp)
    offsets = np.array([loc.offset for loc in locators], dtype=np.float32)

    q = skel_states[:, parent_indices, 3:7]
    t = skel_states[:, parent_indices, :3]
    s = skel_states[:, parent_indices, 7:8]
    scaled_offsets = offsets[np.newaxis, :, :] * s
    rotated = rotate_vector_assume_normalized(
        q.reshape(-1, 4), scaled_offsets.reshape(-1, 3)
    )
    positions = t.reshape(-1, 3) + rotated

    # Colors and radii are constant; send once at the first time point.
    first_time = _make_time_columns(1, fps, frame_offset)

    colors_per_loc = np.empty((n_locs, 3), dtype=np.uint8)
    for i, loc in enumerate(locators):
        if "Floor_" in loc.name:
            colors_per_loc[i] = (100, 255, 100)
        else:
            intensity = int(255.0 * loc.weight * 0.6)
            intensity = max(0, min(255, intensity))
            colors_per_loc[i] = (
                intensity,
                intensity,
                max(0, min(255, int(intensity * 0.5))),
            )

    static_cols = rr.Points3D.columns(
        radii=np.full(n_locs, 0.5, dtype=np.float32),
        colors=colors_per_loc,
    )
    static_cols = static_cols.partition(np.array([n_locs]))
    rec.send_columns(entity_path, indexes=first_time, columns=static_cols)

    # Positions change per frame.
    pos_cols = rr.Points3D.columns(positions=positions)
    pos_cols = pos_cols.partition(np.full(n_frames, n_locs))
    rec.send_columns(entity_path, indexes=time_cols, columns=pos_cols)


def log_animation(
    rec: rr.RecordingStream,
    entity_path: str,
    character: Character,
    skel_states: np.ndarray,
    *,
    posed_meshes: Sequence[Mesh] | None = None,
    fps: float = 30.0,
    color: tuple[int, int, int] = (200, 200, 200),
    frame_offset: int = 0,
) -> None:
    """Log a multi-frame character animation using columnar batch sends.

    This sends all frames at once using :func:`rerun.send_columns`, which is
    dramatically faster than per-frame :func:`log_character` calls for long
    animations.

    For memory-bounded processing of very long animations, call this function
    in a loop with chunks of frames and matching *frame_offset* values::

        chunk_size = 1000
        for start in range(0, n_frames, chunk_size):
            end = min(start + chunk_size, n_frames)
            chunk_meshes = [char.pose_mesh(jp) for jp in joint_params[start:end]]
            log_animation(rec, path, char, skel_states[start:end],
                          posed_meshes=chunk_meshes, fps=fps, frame_offset=start)

    :param rec: The Rerun recording stream.
    :param entity_path: Root entity path for the character.
    :param character: The character to visualize.
    :param skel_states: Skeleton state array of shape ``(n_frames, n_joints, 8)``
        with per-joint ``[tx, ty, tz, rx, ry, rz, rw, scale]``.  Use
        :func:`~pymomentum.geometry.joint_parameters_to_skeleton_state` with
        a batched ``(n_frames, n_joint_params)`` input to compute this
        efficiently in a single C++ call.
    :param posed_meshes: Optional sequence of pre-posed meshes, one per frame.
        If *None*, no mesh is logged.
    :param fps: Frames per second for the time column.
    :param color: RGB color for the mesh albedo factor.
    :param frame_offset: Starting frame index for the time columns.  Use when
        sending multiple chunks of the same animation.
    """
    n_frames = skel_states.shape[0]
    time_cols = _make_time_columns(n_frames, fps, frame_offset)

    if posed_meshes is not None:
        _send_mesh_columns(
            rec,
            f"{entity_path}/mesh",
            posed_meshes,
            time_cols,
            color,
            fps,
            frame_offset,
        )

    _send_joints_columns(
        rec,
        f"{entity_path}/joints",
        character,
        skel_states,
        time_cols,
        fps,
        frame_offset,
    )

    if character.locators:
        _send_locators_columns(
            rec,
            f"{entity_path}/locators",
            character,
            skel_states,
            time_cols,
            fps,
            frame_offset,
        )
