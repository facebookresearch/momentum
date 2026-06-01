# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Viser Visualization Helpers
============================

Helper functions for visualizing momentum characters in a
`viser <https://viser.studio/>`_ web viewer, providing a live interactive
3D scene with per-frame updates.

Unlike the Rerun helpers (:mod:`pymomentum.rerun_vis`), viser is a
**live server** — you add geometry once and then update transforms each
frame.  The typical usage pattern is::

    import viser
    import pymomentum.geometry as geo
    from pymomentum.viser_vis import add_character, update_character

    server = viser.ViserServer(host="::", port=8080)
    character = geo.Character.load_urdf("robot.urdf")
    skel_state = geo.joint_parameters_to_skeleton_state(character, joint_params)

    handles = add_character(server, "character", character, skel_state)

    # In your simulation loop:
    for frame in frames:
        new_skel_state = geo.joint_parameters_to_skeleton_state(character, params)
        update_character(server, handles, character, new_skel_state)

Individual functions are also available:

- :func:`add_mesh` / :func:`update_mesh` — manage a :class:`~pymomentum.geometry.Mesh`
- :func:`add_skinned_mesh` / :func:`update_skinned_mesh` — upload mesh data once
  and update only bone transforms
- :func:`add_joints` / :func:`update_joints` — manage skeleton joint frames
- :func:`add_collision_geometry` / :func:`update_collision_geometry` — manage
  collision proxy capsules, ellipsoids, and boxes
- :func:`add_character` / :func:`update_character` / :func:`remove_character` —
  manage a complete character

Mesh wireframe and collision proxy visibility are controlled through viser's
built-in scene-tree panel (each is its own scene node), so no bespoke GUI
toggle is provided — a single control avoids the desync between a custom
checkbox and the scene tree's client-side visibility override.

GUI widget helpers (optional, for apps building interactive viewers):

- :func:`add_character_param_sliders` — one slider per model parameter
- :func:`add_log_panel` — scrollable HTML log strip (returns a :class:`LogPanel`)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING, TypeAlias

import numpy as np
from pymomentum.quaternion_np import (
    align_z_with,
    from_rotation_matrix,
    rotate_vector_assume_normalized,
    to_rotation_matrix_assume_normalized,
)

if TYPE_CHECKING:
    import viser
    from pymomentum.geometry import Character, Mesh  # @manual=:geometry


@dataclass
class _CollisionCapsuleHandles:
    """Handles for the three primitives that render one capsule."""

    cylinder: Any
    start_sphere: Any
    end_sphere: Any


@dataclass
class _CollisionEllipsoidHandles:
    """Handle for the single icosphere that renders one ellipsoid."""

    ellipsoid: Any


@dataclass
class _CollisionBoxHandles:
    """Handle for the single box mesh that renders one box."""

    box: Any


# Union of all collision primitive handle types stored in
# :attr:`CharacterHandles.collision_handles`, in collision-geometry order.
_CollisionPrimitiveHandles: TypeAlias = (
    _CollisionCapsuleHandles | _CollisionEllipsoidHandles | _CollisionBoxHandles
)


@dataclass
class CharacterHandles:
    """Opaque handle bag returned by :func:`add_character`.

    Pass this to :func:`update_character` or :func:`remove_character`.
    """

    entity_path: str = ""
    joint_frames: dict[str, Any] = field(default_factory=dict)
    bones_handle: Any | None = None
    mesh_handle: Any | None = None
    wireframe_handle: Any | None = None
    collision_root_handle: Any | None = None
    collision_handles: list[_CollisionPrimitiveHandles] = field(default_factory=list)


@dataclass
class _ColoredMeshPart:
    """A uniformly colored submesh derived from a per-vertex colored mesh."""

    handle: Any
    vertex_indices: np.ndarray


_ColoredMeshGroups: TypeAlias = list[
    tuple[tuple[int, int, int], np.ndarray, np.ndarray]
]


#: Default scale factor to convert momentum units (cm) to meters for viser.
CM_TO_M: float = 0.01

# Matches the Rerun viewer and the C++ logger so the same proxy looks identical
# across visualization backends.
_COLLISION_PROXY_COLOR: tuple[int, int, int] = (128, 64, 64)

# Cached unit-sphere mesh used to render ellipsoid collision proxies. Viser has no
# ellipsoid primitive, and the icosphere/box per-axis ``scale`` prop is shadowed by
# the handle's dataclass default in this viser version (it does not reflect on the
# handle). So we bake the per-axis half-sizes directly into a unit-sphere mesh's
# vertices instead.
_UNIT_SPHERE: tuple[np.ndarray, np.ndarray] | None = None


def _unit_sphere_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Return cached (vertices, faces) for a radius-1 UV sphere centered at origin."""
    global _UNIT_SPHERE
    if _UNIT_SPHERE is not None:
        return _UNIT_SPHERE

    n_lat = 16  # number of stacks (excluding the two poles)
    n_lon = 24  # number of slices around the equator
    thetas = np.linspace(0.0, math.pi, n_lat + 2)[1:-1]  # exclude poles
    phis = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)

    ring_x = np.outer(np.sin(thetas), np.cos(phis))
    ring_y = np.outer(np.sin(thetas), np.sin(phis))
    ring_z = np.outer(np.cos(thetas), np.ones_like(phis))
    ring_vertices = np.stack([ring_x.ravel(), ring_y.ravel(), ring_z.ravel()], axis=1)
    top = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    bottom = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    vertices = np.concatenate([ring_vertices, top, bottom], axis=0).astype(np.float32)

    top_index = ring_vertices.shape[0]
    bottom_index = top_index + 1
    faces: list[tuple[int, int, int]] = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            a = i * n_lon + j
            b = i * n_lon + j_next
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + j_next
            faces.append((a, c, b))
            faces.append((b, c, d))
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append((top_index, j, j_next))
        last_ring = (n_lat - 1) * n_lon
        faces.append((bottom_index, last_ring + j_next, last_ring + j))

    _UNIT_SPHERE = (vertices, np.asarray(faces, dtype=np.int32))
    return _UNIT_SPHERE


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from ``[x, y, z, w]`` (pymomentum) to ``[w, x, y, z]`` (viser)."""
    q_arr = np.asarray(q, dtype=np.float64)
    return np.concatenate((q_arr[..., 3:4], q_arr[..., :3]), axis=-1)


def _get_vertex_colors(mesh: Mesh, n_vertices: int) -> np.ndarray | None:
    colors = np.asarray(mesh.colors, dtype=np.uint8)
    if colors.size == 0:
        return None
    if colors.ndim != 2 or colors.shape != (n_vertices, 3):
        return None
    return colors


def _color_for_face(face_colors: np.ndarray) -> tuple[int, int, int]:
    if np.all(face_colors == face_colors[0]):
        color = face_colors[0]
    else:
        color = np.rint(np.mean(face_colors, axis=0)).astype(np.uint8)
    return (int(color[0]), int(color[1]), int(color[2]))


def _add_colored_mesh_parts(
    server: viser.ViserServer,
    entity_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
) -> list[_ColoredMeshPart]:
    parts: list[_ColoredMeshPart] = []
    for color_index, (color, vertex_indices, part_faces) in enumerate(
        _colored_mesh_groups(faces, colors)
    ):
        handle = server.scene.add_mesh_simple(
            f"{entity_path}/color_{color_index}",
            vertices=vertices[vertex_indices],
            faces=part_faces,
            color=color,
        )
        parts.append(_ColoredMeshPart(handle=handle, vertex_indices=vertex_indices))
    return parts


def _colored_mesh_groups(faces: np.ndarray, colors: np.ndarray) -> _ColoredMeshGroups:
    face_indices_by_color: dict[tuple[int, int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        color = _color_for_face(colors[face])
        face_indices_by_color.setdefault(color, []).append(face_index)

    groups: _ColoredMeshGroups = []
    for color, face_indices in face_indices_by_color.items():
        part_faces_original = faces[np.asarray(face_indices, dtype=np.int64)]
        vertex_indices, inverse_faces = np.unique(
            part_faces_original.reshape(-1), return_inverse=True
        )
        part_faces = inverse_faces.reshape((-1, 3)).astype(np.int32)
        groups.append((color, vertex_indices, part_faces))
    return groups


def _solid_mesh_handles(mesh_handle: Any | None) -> list[Any]:
    if mesh_handle is None:
        return []
    if isinstance(mesh_handle, list):
        return [part.handle for part in mesh_handle]
    return [mesh_handle]


def _remove_solid_mesh(mesh_handle: Any | None) -> None:
    for handle in _solid_mesh_handles(mesh_handle):
        handle.remove()


def _skinned_mesh_handles(mesh_handle: Any | None) -> list[Any]:
    if mesh_handle is None:
        return []
    if hasattr(mesh_handle, "bones"):
        return [mesh_handle]
    if isinstance(mesh_handle, list):
        handles = [part.handle for part in mesh_handle]
        if all(hasattr(handle, "bones") for handle in handles):
            return handles
    return []


def add_mesh(
    server: viser.ViserServer,
    entity_path: str,
    mesh: Mesh,
    *,
    color: tuple[int, int, int] = (200, 200, 200),
    scale: float = CM_TO_M,
) -> tuple[Any, Any]:
    """Add a :class:`~pymomentum.geometry.Mesh` to the viser scene.

    Creates both a solid mesh and a hidden wireframe mesh for toggling.

    :param server: The viser server instance.
    :param entity_path: Scene-graph path for the mesh node.
    :param mesh: The mesh to add (typically from
        :meth:`~pymomentum.geometry.Character.pose_mesh`).
    :param color: RGB color tuple ``(r, g, b)`` in 0–255.
    :param scale: Unit conversion factor applied to vertex positions.
        Defaults to :data:`CM_TO_M` (0.01) to convert momentum's cm to meters.
    :return: A tuple of (solid_handle, wireframe_handle). The solid handle is a list
        of uniformly colored submesh handles when per-vertex colors are present.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertex_colors = _get_vertex_colors(mesh, vertices.shape[0])
    solid: Any
    if vertex_colors is None:
        solid = server.scene.add_mesh_simple(
            entity_path,
            vertices=vertices,
            faces=faces,
            color=color,
        )
    else:
        solid = _add_colored_mesh_parts(
            server, entity_path, vertices, faces, vertex_colors
        )
    wireframe = server.scene.add_mesh_simple(
        f"{entity_path}_wireframe",
        vertices=vertices,
        faces=faces,
        color=(80, 80, 80),
        wireframe=True,
        visible=False,
    )
    return solid, wireframe


def add_floor(
    server: viser.ViserServer,
    entity_path: str = "/ground",
    *,
    height: float = 0.0,
    up: str = "z",
    width: float = 20.0,
    cell_size: float = 0.5,
    shadow_opacity: float = 0.3,
    plane_opacity: float = 0.5,
) -> Any:
    """Add a ground plane grid with shadow receiving to the viser scene.

    :param server: The viser server instance.
    :param entity_path: Scene-graph path for the grid node.
    :param height: Position along the up axis for the ground plane.
    :param up: The up axis — ``"y"`` for Y-up (glTF/FBX convention) or
        ``"z"`` for Z-up (URDF/robotics convention).
    :param width: Width and depth of the grid (in scene units).
    :param cell_size: Size of each grid cell.
    :param shadow_opacity: Shadow darkness (0 = none, 1 = black).
    :param plane_opacity: Ground plane opacity (0 = transparent, 1 = opaque).
    :return: A viser grid handle.
    """
    if up == "z":
        plane = "xy"
        position = (0.0, 0.0, height)
    else:
        plane = "xz"
        position = (0.0, height, 0.0)

    return server.scene.add_grid(
        entity_path,
        width=width,
        height=width,
        plane=plane,
        cell_size=cell_size,
        section_size=cell_size * 4,
        position=position,
        shadow_opacity=shadow_opacity,
        plane_color=(255, 255, 255),
        plane_opacity=plane_opacity,
    )


def update_mesh(
    mesh_handle: Any,
    wireframe_handle: Any | None,
    mesh: Mesh,
    *,
    scale: float = CM_TO_M,
) -> None:
    """Update vertex positions of existing mesh handles.

    :param mesh_handle: The solid mesh handle from :func:`add_mesh`.
    :param wireframe_handle: The wireframe handle from :func:`add_mesh`.
    :param mesh: The new posed mesh.
    :param scale: Unit conversion factor.  Defaults to :data:`CM_TO_M`.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale
    if isinstance(mesh_handle, list):
        for part in mesh_handle:
            part.handle.vertices = vertices[part.vertex_indices]
    else:
        mesh_handle.vertices = vertices
    if wireframe_handle is not None:
        wireframe_handle.vertices = vertices


def _max_nonzero_skin_influences(
    skin_weights: np.ndarray,
    *,
    weight_threshold: float = 1e-8,
) -> int:
    """Return the maximum number of nonzero skinning influences on any vertex."""
    if skin_weights.shape[0] == 0:
        return 0
    return int(
        np.max(np.count_nonzero(np.abs(skin_weights) > weight_threshold, axis=1))
    )


def _normalize_skin_weights(skin_weights: np.ndarray) -> np.ndarray:
    """Renormalize skin weights without dropping any nonzero influences."""
    result = skin_weights.copy()
    row_sums = np.sum(result, axis=1, keepdims=True)
    return np.divide(
        result,
        row_sums,
        out=np.zeros_like(result),
        where=row_sums > 0.0,
    )


def max_skin_influences(character: Character) -> int:
    """Return the maximum nonzero skinning influences per vertex for *character*.

    :param character: Character to inspect.
    :return: The largest number of nonzero skin weights on any mesh vertex, or
        zero if the character has no skinned mesh.
    """
    if not character.has_mesh:
        return 0
    skin_weights = np.asarray(
        character.skin_weights.to_dense(character.skeleton.size), dtype=np.float32
    )
    return _max_nonzero_skin_influences(skin_weights)


def can_use_skinned_mesh(character: Character) -> bool:
    """Return whether Viser client-side skinning can represent *character*.

    Viser uses at most four skinning influences per vertex. Characters with
    more influences should use the posed-vertex mesh path to preserve the
    original deformation.

    :param character: Character to inspect.
    :return: ``True`` if the character has a mesh and no vertex has more than
        four nonzero skin weights.
    """
    return character.has_mesh and max_skin_influences(character) <= 4


def add_skinned_mesh(
    server: viser.ViserServer,
    entity_path: str,
    character: Character,
    skel_state: np.ndarray,
    *,
    color: tuple[int, int, int] = (200, 200, 200),
    scale: float = CM_TO_M,
) -> Any:
    """Add a client-side skinned mesh to the viser scene.

    Unlike :func:`add_mesh`, this sends rest vertices, faces, skin weights, and
    bind-pose bones once. Subsequent animation only needs
    :func:`update_skinned_mesh`, which updates per-joint bone transforms rather
    than re-uploading all posed vertex positions.

    :param server: The viser server instance.
    :param entity_path: Scene-graph path for the mesh node.
    :param character: Character with a mesh, skin weights, and bind pose.
    :param skel_state: Initial skeleton state array of shape ``(n_joints, 8)``.
    :param color: RGB color tuple ``(r, g, b)`` in 0–255.
    :param scale: Unit conversion factor applied to vertices and bone
        translations. Defaults to :data:`CM_TO_M`.
    :return: The viser skinned mesh handle. When per-vertex colors are present,
        returns a list of uniformly colored skinned submesh handles.
    """
    if not character.has_mesh:
        raise ValueError("add_skinned_mesh() requires a character with mesh skinning")

    skeleton_size = character.skeleton.size
    mesh = character.mesh
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale
    faces = np.asarray(mesh.faces, dtype=np.int32)
    skin_weights = np.asarray(
        character.skin_weights.to_dense(skeleton_size), dtype=np.float32
    )
    if skin_weights.shape != (vertices.shape[0], skeleton_size):
        raise ValueError(
            "Expected dense skin weights with shape "
            f"({vertices.shape[0]}, {skeleton_size}); got {skin_weights.shape}"
        )
    max_influences = _max_nonzero_skin_influences(skin_weights)
    if max_influences > 4:
        raise ValueError(
            "Viser skinned meshes support at most 4 skinning influences per "
            f"vertex; this character has {max_influences}"
        )
    skin_weights = _normalize_skin_weights(skin_weights)

    bind_pose = np.asarray(character.bind_pose, dtype=np.float32)
    expected_bind_pose_shape = (skeleton_size, 4, 4)
    if bind_pose.shape != expected_bind_pose_shape:
        raise ValueError(
            f"Expected bind pose shape {expected_bind_pose_shape}; got {bind_pose.shape}"
        )

    bone_positions = bind_pose[:, :3, 3].astype(np.float32) * scale
    bone_wxyzs = _xyzw_to_wxyz(from_rotation_matrix(bind_pose[:, :3, :3])).astype(
        np.float32
    )
    if skeleton_size < 4:
        # Viser's skinned mesh path always extracts four influences per vertex,
        # so characters with fewer than four joints need zero-weight padding.
        pad_count = 4 - skeleton_size
        skin_weights = np.pad(skin_weights, ((0, 0), (0, pad_count)))
        bone_positions = np.concatenate(
            (bone_positions, np.zeros((pad_count, 3), dtype=np.float32)), axis=0
        )
        bone_wxyzs = np.concatenate(
            (
                bone_wxyzs,
                np.tile(
                    np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    (pad_count, 1),
                ),
            ),
            axis=0,
        )
    vertex_colors = _get_vertex_colors(mesh, vertices.shape[0])
    skinned_mesh_handle: Any
    if vertex_colors is None:
        skinned_mesh_handle = server.scene.add_mesh_skinned(
            entity_path,
            vertices=vertices,
            faces=faces,
            bone_wxyzs=bone_wxyzs,
            bone_positions=bone_positions,
            skin_weights=skin_weights,
            color=color,
            scale=1.0,
        )
    else:
        skinned_mesh_handle = _add_colored_skinned_mesh_parts(
            server,
            entity_path,
            vertices,
            faces,
            skin_weights,
            bone_wxyzs,
            bone_positions,
            vertex_colors,
        )
    update_skinned_mesh(skinned_mesh_handle, skel_state, scale=scale)
    return skinned_mesh_handle


def _add_colored_skinned_mesh_parts(
    server: viser.ViserServer,
    entity_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    skin_weights: np.ndarray,
    bone_wxyzs: np.ndarray,
    bone_positions: np.ndarray,
    colors: np.ndarray,
) -> list[_ColoredMeshPart]:
    parts: list[_ColoredMeshPart] = []
    for color_index, (color, vertex_indices, part_faces) in enumerate(
        _colored_mesh_groups(faces, colors)
    ):
        handle = server.scene.add_mesh_skinned(
            f"{entity_path}/color_{color_index}",
            vertices=vertices[vertex_indices],
            faces=part_faces,
            bone_wxyzs=bone_wxyzs,
            bone_positions=bone_positions,
            skin_weights=skin_weights[vertex_indices],
            color=color,
            scale=1.0,
        )
        parts.append(_ColoredMeshPart(handle=handle, vertex_indices=vertex_indices))
    return parts


def update_skinned_mesh(
    skinned_mesh_handle: Any,
    skel_state: np.ndarray,
    *,
    scale: float = CM_TO_M,
) -> None:
    """Update a skinned mesh by sending only bone transforms.

    :param skinned_mesh_handle: The handle returned by :func:`add_skinned_mesh`.
        This may be a list of uniformly colored skinned submesh handles.
    :param skel_state: Skeleton state array of shape ``(n_joints, 8)``.
    :param scale: Unit conversion factor for bone translations.
    """
    handles = _skinned_mesh_handles(skinned_mesh_handle)
    if not handles:
        raise ValueError("update_skinned_mesh() requires skinned mesh handle(s)")

    positions = skel_state[:, :3].astype(np.float64) * scale
    wxyzs = _xyzw_to_wxyz(skel_state[:, 3:7])
    for handle in handles:
        _update_skinned_mesh_handle(handle, positions, wxyzs, skel_state.shape[0])


def _update_skinned_mesh_handle(
    skinned_mesh_handle: Any,
    positions: np.ndarray,
    wxyzs: np.ndarray,
    n_joints: int,
) -> None:
    bones = skinned_mesh_handle.bones
    if len(bones) < n_joints:
        raise ValueError(
            f"Skinned mesh has {len(bones)} bones but got {n_joints} joints"
        )

    for bone, position, wxyz in zip(bones[:n_joints], positions, wxyzs):
        bone.position = position
        bone.wxyz = wxyz


def _build_bone_segments(
    character: Character,
    skel_state: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray | None:
    """Build an (N, 2, 3) array of bone line segments from parent-child joints."""
    joint_parents: list[int] = list(character.skeleton.joint_parents)
    n_joints = len(joint_parents)
    segments: list[list[np.ndarray]] = []
    for i in range(n_joints):
        parent_idx = joint_parents[i]
        if 0 <= parent_idx < n_joints:
            segments.append([skel_state[parent_idx, :3], skel_state[i, :3]])
    if not segments:
        return None
    return np.array(segments, dtype=np.float32) * scale


def _wxyz_from_z_axis(direction: np.ndarray) -> np.ndarray:
    """Return a wxyz quaternion that rotates +Z onto *direction*."""
    return _xyzw_to_wxyz(align_z_with(direction))


def _is_capsule(primitive: Any) -> bool:
    return hasattr(primitive, "length")


def _is_ellipsoid(primitive: Any) -> bool:
    return hasattr(primitive, "radii")


def _is_box(primitive: Any) -> bool:
    return hasattr(primitive, "half_extents")


def _collision_primitive_world_transform(
    primitive: Any,
    skel_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compose a primitive's world transform from its parent joint state.

    Mirrors momentum's C++ collision convention
    (``CollisionGeometryStateT::update`` in ``collision_geometry_state.cpp``):
    ``transform = parentTransform * primitive.transformation``, with the center
    taken from the composed translation and the orientation from the composed
    rotation. World-space primitives without a parent joint
    (``parent == kInvalidIndex``) use an identity parent transform.

    :return: A tuple of (center, world xyzw quaternion, parent joint scale).
        Ellipsoid radii and box half-extents are scaled by the parent joint
        scale only, matching the C++ state update.
    """
    local = np.asarray(primitive.transformation, dtype=np.float64)
    parent_scale = 1.0
    parent_transform = np.eye(4, dtype=np.float64)
    parent = int(primitive.parent)
    if 0 <= parent < skel_state.shape[0]:
        parent_scale = float(skel_state[parent, 7])
        parent_rotation = to_rotation_matrix_assume_normalized(
            skel_state[parent, 3:7][None, :]
        )[0]
        parent_transform[:3, :3] = parent_rotation * parent_scale
        parent_transform[:3, 3] = skel_state[parent, :3]

    world = parent_transform @ local
    world_scale = float(np.linalg.norm(world[:3, 0]))
    if world_scale <= 1e-8:
        world_scale = 1.0
    rotation = world[:3, :3] / world_scale
    world_quat = from_rotation_matrix(rotation[None, :, :])[0].astype(np.float64)
    return world[:3, 3], world_quat, parent_scale


def _ellipsoid_render_state(
    primitive: Any,
    skel_state: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (center, xyzw quaternion, half_sizes) for one ellipsoid in scene units."""
    center, quaternion, parent_scale = _collision_primitive_world_transform(
        primitive, skel_state
    )
    half_sizes = np.asarray(primitive.radii, dtype=np.float64) * parent_scale * scale
    return center * scale, quaternion, np.maximum(half_sizes, 1e-8)


def _box_render_state(
    primitive: Any,
    skel_state: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (center, xyzw quaternion, half_sizes) for one box in scene units."""
    center, quaternion, parent_scale = _collision_primitive_world_transform(
        primitive, skel_state
    )
    half_sizes = (
        np.asarray(primitive.half_extents, dtype=np.float64) * parent_scale * scale
    )
    return center * scale, quaternion, np.maximum(half_sizes, 1e-8)


def _capsule_render_segment(
    primitive: Any,
    skel_state: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (start, end, radius) for one capsule centerline in scene units.

    Uses the same parent-scale convention as the C++ collision state: the
    centerline runs along the local X axis (scaled by parent joint scale), and
    the rendered radius is the larger tapered endpoint radius scaled by the
    parent joint scale.
    """
    transform = np.asarray(primitive.transformation, dtype=np.float64)
    length = float(primitive.length)
    radius = float(np.max(np.asarray(primitive.radius, dtype=np.float64)))

    parent = int(primitive.parent)
    if 0 <= parent < skel_state.shape[0]:
        parent_t = skel_state[parent, :3].astype(np.float64)
        parent_q = skel_state[parent, 3:7].astype(np.float64)
        parent_scale = float(skel_state[parent, 7])
    else:
        parent_t = np.zeros(3, dtype=np.float64)
        parent_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        parent_scale = 1.0

    local_start = transform[:3, 3]
    local_end = local_start + transform[:3, 0] * length
    start = (
        parent_t + rotate_vector_assume_normalized(parent_q, local_start * parent_scale)
    ) * scale
    end = (
        parent_t + rotate_vector_assume_normalized(parent_q, local_end * parent_scale)
    ) * scale
    rendered_radius = max(radius * abs(parent_scale) * scale, 1e-8)
    return start, end, rendered_radius


def add_collision_geometry(
    server: viser.ViserServer,
    entity_path: str,
    character: Character,
    skel_state: np.ndarray,
    *,
    scale: float = CM_TO_M,
    color: tuple[int, int, int] = _COLLISION_PROXY_COLOR,
    opacity: float = 0.45,
    visible: bool = True,
) -> tuple[Any, list[_CollisionPrimitiveHandles]]:
    """Add collision proxy primitives to the viser scene.

    Momentum stores tapered capsules, ellipsoids, and boxes. Capsules are
    rendered as a cylinder plus two end icospheres using the larger endpoint
    radius (viser does not expose tapered capsules); ellipsoids are rendered as a
    scaled icosphere and boxes as a box mesh.

    :param server: The viser server instance.
    :param entity_path: Scene-graph path for the collision proxy group.
    :param character: The character whose collision geometry is visualized.
    :param skel_state: Skeleton state array of shape ``(n_joints, 8)``.
    :param scale: Unit conversion factor. Defaults to :data:`CM_TO_M`.
    :param color: RGB color tuple for the collision proxies.
    :param opacity: Material opacity for the proxies.
    :param visible: Initial visibility for the collision proxy group.
    :return: A tuple of (root_handle, collision primitive handle groups), one
        handle group per primitive in collision-geometry order.
    """
    root_handle = server.scene.add_frame(
        entity_path,
        show_axes=False,
        visible=visible,
    )
    collision_handles: list[_CollisionPrimitiveHandles] = []
    for i, primitive in enumerate(character.collision_geometry):
        primitive_path = f"{entity_path}/{i}"
        if _is_capsule(primitive):
            collision_handles.append(
                _add_collision_capsule(
                    server, primitive_path, primitive, skel_state, scale, color, opacity
                )
            )
        elif _is_ellipsoid(primitive):
            collision_handles.append(
                _add_collision_ellipsoid(
                    server, primitive_path, primitive, skel_state, scale, color, opacity
                )
            )
        elif _is_box(primitive):
            collision_handles.append(
                _add_collision_box(
                    server, primitive_path, primitive, skel_state, scale, color, opacity
                )
            )
        else:
            raise ValueError(
                f"Unsupported collision primitive at index {i}: "
                f"{type(primitive).__name__}"
            )

    return root_handle, collision_handles


def _add_collision_capsule(
    server: viser.ViserServer,
    primitive_path: str,
    primitive: Any,
    skel_state: np.ndarray,
    scale: float,
    color: tuple[int, int, int],
    opacity: float,
) -> _CollisionCapsuleHandles:
    start, end, rendered_radius = _capsule_render_segment(primitive, skel_state, scale)
    direction = end - start
    height = max(float(np.linalg.norm(direction)), 1e-8)

    cylinder = server.scene.add_cylinder(
        f"{primitive_path}/body",
        radius=rendered_radius,
        height=height,
        color=color,
        radial_segments=16,
        opacity=opacity,
        side="double",
        cast_shadow=False,
        receive_shadow=False,
        position=(start + end) * 0.5,
        wxyz=_wxyz_from_z_axis(direction),
        visible=True,
    )
    start_sphere = server.scene.add_icosphere(
        f"{primitive_path}/start",
        radius=rendered_radius,
        color=color,
        subdivisions=1,
        opacity=opacity,
        cast_shadow=False,
        receive_shadow=False,
        position=start,
        visible=True,
    )
    end_sphere = server.scene.add_icosphere(
        f"{primitive_path}/end",
        radius=rendered_radius,
        color=color,
        subdivisions=1,
        opacity=opacity,
        cast_shadow=False,
        receive_shadow=False,
        position=end,
        visible=True,
    )
    return _CollisionCapsuleHandles(
        cylinder=cylinder,
        start_sphere=start_sphere,
        end_sphere=end_sphere,
    )


def _add_collision_ellipsoid(
    server: viser.ViserServer,
    primitive_path: str,
    primitive: Any,
    skel_state: np.ndarray,
    scale: float,
    color: tuple[int, int, int],
    opacity: float,
) -> _CollisionEllipsoidHandles:
    center, quaternion, half_sizes = _ellipsoid_render_state(
        primitive, skel_state, scale
    )
    # Bake the per-axis half-sizes into the mesh vertices: viser has no ellipsoid
    # primitive, and the icosphere ``scale`` prop does not reflect on the handle.
    unit_vertices, faces = _unit_sphere_mesh()
    ellipsoid = server.scene.add_mesh_simple(
        f"{primitive_path}/body",
        vertices=(unit_vertices * half_sizes).astype(np.float32),
        faces=faces,
        color=color,
        opacity=opacity,
        cast_shadow=False,
        receive_shadow=False,
        wxyz=_xyzw_to_wxyz(quaternion),
        position=center,
        visible=True,
    )
    return _CollisionEllipsoidHandles(ellipsoid=ellipsoid)


def _add_collision_box(
    server: viser.ViserServer,
    primitive_path: str,
    primitive: Any,
    skel_state: np.ndarray,
    scale: float,
    color: tuple[int, int, int],
    opacity: float,
) -> _CollisionBoxHandles:
    center, quaternion, half_sizes = _box_render_state(primitive, skel_state, scale)
    box = server.scene.add_box(
        f"{primitive_path}/body",
        dimensions=tuple(float(v) for v in 2.0 * half_sizes),
        color=color,
        opacity=opacity,
        cast_shadow=False,
        receive_shadow=False,
        wxyz=_xyzw_to_wxyz(quaternion),
        position=center,
        visible=True,
    )
    return _CollisionBoxHandles(box=box)


def update_collision_geometry(
    collision_handles: list[_CollisionPrimitiveHandles],
    character: Character,
    skel_state: np.ndarray,
    *,
    scale: float = CM_TO_M,
) -> None:
    """Update collision proxy positions and sizes.

    Always repositions every primitive from the current ``skel_state``,
    regardless of visibility, so the proxies stay in sync with the model.
    Visibility is a separate concern controlled through viser's scene-tree panel;
    gating updates on visibility would leave proxies stranded at a stale pose when
    later revealed through a path that does not re-run this.

    :param collision_handles: Collision primitive handles returned by
        :func:`add_collision_geometry`.
    :param character: The character whose collision geometry is visualized.
    :param skel_state: Skeleton state array of shape ``(n_joints, 8)``.
    :param scale: Unit conversion factor. Defaults to :data:`CM_TO_M`.
    :return: None.
    """
    if not collision_handles:
        return

    primitives = character.collision_geometry
    if len(primitives) != len(collision_handles):
        raise ValueError(
            "Collision geometry primitive count does not match existing handles"
        )

    for primitive_handles, primitive in zip(collision_handles, primitives):
        if isinstance(primitive_handles, _CollisionCapsuleHandles):
            _update_collision_capsule(primitive_handles, primitive, skel_state, scale)
        elif isinstance(primitive_handles, _CollisionEllipsoidHandles):
            center, quaternion, half_sizes = _ellipsoid_render_state(
                primitive, skel_state, scale
            )
            unit_vertices, _faces = _unit_sphere_mesh()
            primitive_handles.ellipsoid.position = center
            primitive_handles.ellipsoid.wxyz = _xyzw_to_wxyz(quaternion)
            primitive_handles.ellipsoid.vertices = (unit_vertices * half_sizes).astype(
                np.float32
            )
        elif isinstance(primitive_handles, _CollisionBoxHandles):
            center, quaternion, half_sizes = _box_render_state(
                primitive, skel_state, scale
            )
            primitive_handles.box.position = center
            primitive_handles.box.wxyz = _xyzw_to_wxyz(quaternion)
            primitive_handles.box.dimensions = tuple(float(v) for v in 2.0 * half_sizes)


def _update_collision_capsule(
    capsule_handles: _CollisionCapsuleHandles,
    primitive: Any,
    skel_state: np.ndarray,
    scale: float,
) -> None:
    start, end, rendered_radius = _capsule_render_segment(primitive, skel_state, scale)
    direction = end - start
    height = max(float(np.linalg.norm(direction)), 1e-8)

    capsule_handles.cylinder.position = (start + end) * 0.5
    capsule_handles.cylinder.wxyz = _wxyz_from_z_axis(direction)
    capsule_handles.cylinder.height = height
    capsule_handles.cylinder.radius = rendered_radius
    capsule_handles.start_sphere.position = start
    capsule_handles.start_sphere.radius = rendered_radius
    capsule_handles.end_sphere.position = end
    capsule_handles.end_sphere.radius = rendered_radius


def _collision_primitive_parts(
    primitive_handles: _CollisionPrimitiveHandles,
) -> tuple[Any, ...]:
    if isinstance(primitive_handles, _CollisionCapsuleHandles):
        return (
            primitive_handles.cylinder,
            primitive_handles.start_sphere,
            primitive_handles.end_sphere,
        )
    if isinstance(primitive_handles, _CollisionEllipsoidHandles):
        return (primitive_handles.ellipsoid,)
    return (primitive_handles.box,)


def _remove_collision_geometry(
    collision_root_handle: Any | None,
    collision_handles: list[_CollisionPrimitiveHandles],
) -> None:
    # Idempotent: clear the list as we remove so a second call (e.g. accidental
    # double-remove during cleanup) is a no-op rather than touching freed handles.
    while collision_handles:
        primitive_handles = collision_handles.pop()
        for part in _collision_primitive_parts(primitive_handles):
            part.remove()
    if collision_root_handle is not None:
        collision_root_handle.remove()


def add_joints(
    server: viser.ViserServer,
    entity_path: str,
    character: Character,
    skel_state: np.ndarray,
    *,
    scale: float = CM_TO_M,
    axes_length: float = 0.05,
    axes_radius: float = 0.002,
    bone_line_width: float = 4.0,
    bone_color: tuple[int, int, int] = (180, 180, 180),
) -> tuple[dict[str, Any], Any | None]:
    """Add per-joint coordinate frames and bone lines to the viser scene.

    Each joint is rendered as a small set of RGB axes at its world-space
    position and orientation.  Parent-child joints are connected by grey
    line segments.

    :param server: The viser server instance.
    :param entity_path: Base scene-graph path; joints are added as children.
    :param character: The character whose skeleton is visualized.
    :param skel_state: Skeleton state array of shape ``(n_joints, 8)`` with
        per-joint ``[tx, ty, tz, rx, ry, rz, rw, scale]``.
    :param axes_length: Length of the axis indicators (in scene units).
    :param axes_radius: Radius of the axis indicators.
    :param bone_line_width: Width of the bone line segments.
    :param bone_color: RGB color tuple for the bone lines.
    :return: A tuple of (joint_handle_dict, bones_handle_or_None).
    """
    skeleton = character.skeleton
    joint_names: list[str] = skeleton.joint_names
    n_joints: int = len(joint_names)
    handles: dict[str, Any] = {}

    for i in range(n_joints):
        name = joint_names[i]
        pos = skel_state[i, :3].astype(np.float64) * scale
        wxyz = _xyzw_to_wxyz(skel_state[i, 3:7])
        handle = server.scene.add_frame(
            f"{entity_path}/{name}",
            position=pos,
            wxyz=wxyz,
            axes_length=axes_length,
            axes_radius=axes_radius,
        )
        handles[name] = handle

    # Add bone line segments connecting parent-child joints
    bones_handle = None
    segments = _build_bone_segments(character, skel_state, scale=scale)
    if segments is not None:
        bones_handle = server.scene.add_line_segments(
            f"{entity_path}/bones",
            points=segments,
            colors=np.array(bone_color, dtype=np.uint8),
            line_width=bone_line_width,
        )

    return handles, bones_handle


def update_joints(
    handles: dict[str, Any],
    bones_handle: Any | None,
    character: Character,
    skel_state: np.ndarray,
    *,
    scale: float = CM_TO_M,
) -> None:
    """Update joint frame positions, orientations, and bone lines.

    :param handles: The joint handle dict returned by :func:`add_joints`.
    :param bones_handle: The bones handle returned by :func:`add_joints`.
    :param character: The character whose skeleton is being updated.
    :param skel_state: New skeleton state array of shape ``(n_joints, 8)``.
    :param scale: Unit conversion factor.  Defaults to :data:`CM_TO_M`.
    """
    joint_names: list[str] = character.skeleton.joint_names
    for i, name in enumerate(joint_names):
        handle = handles.get(name)
        if handle is None:
            continue
        handle.position = skel_state[i, :3].astype(np.float64) * scale
        handle.wxyz = _xyzw_to_wxyz(skel_state[i, 3:7])

    if bones_handle is not None:
        segments = _build_bone_segments(character, skel_state, scale=scale)
        if segments is not None:
            bones_handle.points = segments


def add_character(
    server: viser.ViserServer,
    entity_path: str,
    character: Character,
    skel_state: np.ndarray,
    *,
    posed_mesh: Mesh | None = None,
    use_skinned_mesh: bool = False,
    color: tuple[int, int, int] = (200, 200, 200),
    scale: float = CM_TO_M,
    axes_length: float = 0.05,
    axes_radius: float = 0.002,
    collision_geometry: bool = False,
    collision_geometry_visible: bool = True,
) -> CharacterHandles:
    """Add a complete character visualization to the viser scene.

    Creates joint frames and optionally a mesh, returning handles for
    subsequent updates via :func:`update_character`.

    - ``{entity_path}/joints/{name}`` — per-joint coordinate frames
    - ``{entity_path}/mesh`` — posed or skinned mesh
    - ``{entity_path}/collision_proxies`` — collision capsules, ellipsoids, and
      boxes, if requested

    :param server: The viser server instance.
    :param entity_path: Root scene-graph path for the character.
    :param character: The character to visualize.
    :param skel_state: Skeleton state array of shape ``(n_joints, 8)``.
    :param posed_mesh: An optional pre-posed mesh for the legacy mesh path.  If
        *None* and *use_skinned_mesh* is false, no mesh is added.
    :param use_skinned_mesh: If true and the character has a skinned mesh, add
        a client-side skinned mesh instead of uploading posed vertices.
    :param color: RGB color tuple for the mesh.
    :param axes_length: Length of joint axis indicators.
    :param axes_radius: Radius of joint axis indicators.
    :param collision_geometry: If true, add collision proxy capsules,
        ellipsoids, and boxes.
    :param collision_geometry_visible: Initial visibility for collision proxies.
    :return: A :class:`CharacterHandles` for use with :func:`update_character`.
    """
    handles = CharacterHandles(entity_path=entity_path)

    handles.joint_frames, handles.bones_handle = add_joints(
        server,
        f"{entity_path}/joints",
        character,
        skel_state,
        scale=scale,
        axes_length=axes_length,
        axes_radius=axes_radius,
    )

    if use_skinned_mesh and character.has_mesh:
        handles.mesh_handle = add_skinned_mesh(
            server,
            f"{entity_path}/mesh",
            character,
            skel_state,
            color=color,
            scale=scale,
        )
    elif posed_mesh is not None:
        handles.mesh_handle, handles.wireframe_handle = add_mesh(
            server, f"{entity_path}/mesh", posed_mesh, color=color, scale=scale
        )

    if collision_geometry:
        (
            handles.collision_root_handle,
            handles.collision_handles,
        ) = add_collision_geometry(
            server,
            f"{entity_path}/collision_proxies",
            character,
            skel_state,
            scale=scale,
            visible=collision_geometry_visible,
        )

    return handles


def update_character(
    server: viser.ViserServer,
    handles: CharacterHandles,
    character: Character,
    skel_state: np.ndarray,
    *,
    posed_mesh: Mesh | None = None,
    scale: float = CM_TO_M,
) -> None:
    """Update all character transforms in a single atomic batch.

    Wraps updates in ``server.atomic()`` to avoid partial-frame rendering
    artifacts, following the pattern used in ``rl_robot``'s viser renderer.

    :param server: The viser server instance.
    :param handles: The handles returned by :func:`add_character`.
    :param character: The character being updated.
    :param skel_state: New skeleton state array of shape ``(n_joints, 8)``.
    :param posed_mesh: If provided, updates mesh vertex positions.
    :param scale: Unit conversion factor.  Defaults to :data:`CM_TO_M`.
    """
    with server.atomic():
        update_joints(
            handles.joint_frames,
            handles.bones_handle,
            character,
            skel_state,
            scale=scale,
        )

        if _skinned_mesh_handles(handles.mesh_handle):
            update_skinned_mesh(handles.mesh_handle, skel_state, scale=scale)
        elif posed_mesh is not None and handles.mesh_handle is not None:
            update_mesh(
                handles.mesh_handle, handles.wireframe_handle, posed_mesh, scale=scale
            )

        update_collision_geometry(
            handles.collision_handles,
            character,
            skel_state,
            scale=scale,
        )

        server.flush()


def remove_character(
    handles: CharacterHandles,
    server: viser.ViserServer | None = None,
) -> None:
    """Remove all character scene nodes from the viser scene.

    :param handles: The handles returned by :func:`add_character`.
    :param server: If provided, removes the entire scene subtree by name
        (cleans up parent nodes from viser's scene tree).
    """
    if server is not None and handles.entity_path:
        server.scene.remove_by_name(handles.entity_path)
    else:
        for handle in handles.joint_frames.values():
            handle.remove()
        if handles.bones_handle is not None:
            handles.bones_handle.remove()
        _remove_solid_mesh(handles.mesh_handle)
        if handles.wireframe_handle is not None:
            handles.wireframe_handle.remove()
        _remove_collision_geometry(
            handles.collision_root_handle,
            handles.collision_handles,
        )

    handles.joint_frames.clear()
    handles.bones_handle = None
    handles.mesh_handle = None
    handles.wireframe_handle = None
    handles.collision_root_handle = None
    handles.collision_handles.clear()


# =============================================================================
# Optional GUI widget helpers
# =============================================================================


def _normalize_param_limit(name: str, lo: float, hi: float) -> tuple[float, float]:
    """Clamp unbounded model-parameter limits to viewer-friendly slider ranges.

    Translation parameters (``_tx``/``_ty``/``_tz`` in the name) are clamped to
    ±200 cm; everything else is clamped to ±π radians.
    """
    if lo > -1e30 and hi < 1e30:
        return lo, hi
    if "_tx" in name or "_ty" in name or "_tz" in name:
        return max(lo, -200.0), min(hi, 200.0)
    return max(lo, -math.pi), min(hi, math.pi)


def add_character_param_sliders(
    server: viser.ViserServer,
    character: Character,
    *,
    on_change: Callable[[Any], None] | None = None,
    initial_values: np.ndarray | None = None,
) -> list[Any]:
    """Build one slider per model parameter of *character*.

    Unbounded parameter limits are clamped via :func:`_normalize_param_limit` so
    the sliders stay usable. The caller reads slider values from the returned
    list (e.g. ``np.array([s.value for s in sliders])``) and converts them back
    into model parameters.

    :param server: The viser server instance.
    :param character: The character whose model parameters drive the sliders.
    :param on_change: Optional callback wired to every slider's
        ``on_update`` event. The callback receives a single positional argument
        (the viser event object), which it can ignore.
    :param initial_values: Optional length-``n_params`` array of initial slider
        values. Each value is clamped to its slider's range. If *None*, all
        sliders start at 0 (or the closest in-range value).
    :return: The list of slider handles, in parameter-index order.
    """
    param_names: list[str] = character.parameter_transform.names
    min_limits, max_limits = character.model_parameter_limits
    n_params = character.parameter_transform.size
    sliders: list[Any] = []
    for i in range(n_params):
        lo, hi = _normalize_param_limit(
            param_names[i], float(min_limits[i]), float(max_limits[i])
        )
        initial = 0.0 if initial_values is None else float(initial_values[i])
        slider = server.gui.add_slider(
            param_names[i],
            min=lo,
            max=hi,
            step=(hi - lo) / 200.0,
            initial_value=max(lo, min(hi, initial)),
        )
        if on_change is not None:
            slider.on_update(on_change)
        sliders.append(slider)
    return sliders


class LogPanel:
    """Scrollable HTML log strip created by :func:`add_log_panel`.

    Call :meth:`log` to append a line. The panel keeps the last ``max_lines``
    entries and auto-scrolls to the latest.
    """

    def __init__(
        self,
        html_handle: Any,
        *,
        height_px: int = 80,
        max_lines: int = 50,
        echo_to_stdout: bool = True,
    ) -> None:
        self._handle: Any = html_handle
        self._height_px: int = height_px
        self._max_lines: int = max_lines
        self._echo: bool = echo_to_stdout
        self._lines: list[str] = []

    def log(self, msg: str, level: str = "info") -> None:
        """Append a log line. *level* is one of ``"info"``, ``"warning"``, ``"error"``."""
        icon = {"error": "🔴", "warning": "🟡"}.get(level, "🟢")
        self._lines.append(f"{icon} {msg}")
        body = "<br>".join(self._lines[-self._max_lines :])
        self._handle.content = (
            f'<div style="height:{self._height_px}px; overflow-y:auto; '
            f"font-size:12px; padding:4px; background:#f8f8f8; "
            f'border-radius:4px; border:1px solid #ddd;">{body}</div>'
        )
        if self._echo:
            print(f"  [{level.upper()}] {msg}")


def add_log_panel(
    server: viser.ViserServer,
    *,
    height_px: int = 80,
    max_lines: int = 50,
    echo_to_stdout: bool = True,
) -> LogPanel:
    """Add a scrollable HTML log strip to the GUI and return its controller.

    :param server: The viser server instance.
    :param height_px: Fixed height of the log strip.
    :param max_lines: Maximum number of lines retained on screen.
    :param echo_to_stdout: If ``True`` (default), each ``log()`` call also
        prints to ``stdout`` for terminal visibility.
    :return: A :class:`LogPanel` whose ``.log(msg, level)`` method appends lines.
    """
    handle = server.gui.add_html("")
    return LogPanel(
        handle,
        height_px=height_px,
        max_lines=max_lines,
        echo_to_stdout=echo_to_stdout,
    )
