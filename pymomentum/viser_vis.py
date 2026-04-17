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
- :func:`add_joints` / :func:`update_joints` — manage skeleton joint frames
- :func:`add_character` / :func:`update_character` / :func:`remove_character` —
  manage a complete character

GUI widget helpers (optional, for apps building interactive viewers):

- :func:`add_character_param_sliders` — one slider per model parameter
- :func:`add_wireframe_toggle` — checkbox that toggles a character's mesh/wireframe
- :func:`add_log_panel` — scrollable HTML log strip (returns a :class:`LogPanel`)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import viser
    from pymomentum.geometry import Character, Mesh


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


#: Default scale factor to convert momentum units (cm) to meters for viser.
CM_TO_M: float = 0.01


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from ``[x, y, z, w]`` (pymomentum) to ``[w, x, y, z]`` (viser)."""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


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
    :return: A tuple of (solid_handle, wireframe_handle).
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale
    faces = np.asarray(mesh.faces, dtype=np.int32)
    solid = server.scene.add_mesh_simple(
        entity_path,
        vertices=vertices,
        faces=faces,
        color=color,
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
    mesh_handle.vertices = vertices
    if wireframe_handle is not None:
        wireframe_handle.vertices = vertices


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
    color: tuple[int, int, int] = (200, 200, 200),
    scale: float = CM_TO_M,
    axes_length: float = 0.05,
    axes_radius: float = 0.002,
) -> CharacterHandles:
    """Add a complete character visualization to the viser scene.

    Creates joint frames and optionally a mesh, returning handles for
    subsequent updates via :func:`update_character`.

    - ``{entity_path}/joints/{name}`` — per-joint coordinate frames
    - ``{entity_path}/mesh`` — posed mesh (if *posed_mesh* is provided)

    :param server: The viser server instance.
    :param entity_path: Root scene-graph path for the character.
    :param character: The character to visualize.
    :param skel_state: Skeleton state array of shape ``(n_joints, 8)``.
    :param posed_mesh: An optional pre-posed mesh.  If *None*, no mesh
        is added.
    :param color: RGB color tuple for the mesh.
    :param axes_length: Length of joint axis indicators.
    :param axes_radius: Radius of joint axis indicators.
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

    if posed_mesh is not None:
        handles.mesh_handle, handles.wireframe_handle = add_mesh(
            server, f"{entity_path}/mesh", posed_mesh, color=color, scale=scale
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

        if posed_mesh is not None and handles.mesh_handle is not None:
            update_mesh(
                handles.mesh_handle, handles.wireframe_handle, posed_mesh, scale=scale
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
        if handles.mesh_handle is not None:
            handles.mesh_handle.remove()
        if handles.wireframe_handle is not None:
            handles.wireframe_handle.remove()

    handles.joint_frames.clear()
    handles.bones_handle = None
    handles.mesh_handle = None
    handles.wireframe_handle = None


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


def add_wireframe_toggle(
    server: viser.ViserServer,
    handles: CharacterHandles,
    *,
    label: str = "Wireframe",
    initial_value: bool = False,
) -> Any:
    """Add a checkbox that toggles between solid and wireframe for *handles*.

    :func:`add_mesh` (and therefore :func:`add_character`) creates both a solid
    and a wireframe handle; this widget flips their ``.visible`` flags so only
    one is shown at a time. Sets initial visibility on both handles to match
    *initial_value*.

    :param server: The viser server instance.
    :param handles: The character handles whose mesh/wireframe to toggle.
    :param label: Checkbox label shown in the GUI.
    :param initial_value: ``True`` shows the wireframe initially, ``False``
        shows the solid mesh.
    :return: The viser checkbox handle.
    """
    cb = server.gui.add_checkbox(label, initial_value=initial_value)
    if handles.mesh_handle is not None:
        handles.mesh_handle.visible = not initial_value
    if handles.wireframe_handle is not None:
        handles.wireframe_handle.visible = initial_value

    @cb.on_update
    def _(_: object) -> None:
        if handles.mesh_handle is not None:
            handles.mesh_handle.visible = not cb.value
        if handles.wireframe_handle is not None:
            handles.wireframe_handle.visible = cb.value

    return cb


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
