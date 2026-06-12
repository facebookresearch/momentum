# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest
from typing import Any

import numpy as np
import pymomentum.geometry as geo  # @manual=:geometry
import pymomentum.geometry_test_utils as test_utils  # @manual=:geometry_test_utils
from pymomentum.quaternion_np import to_rotation_matrix_assume_normalized

try:
    import viser

    _HAS_VISER = True
except ImportError:
    viser = None  # pyre-ignore[9]
    _HAS_VISER = False


# The TestViserVis class is gated behind `_HAS_VISER` (rather than skipped via
# load_tests() alone) so that pytest — which does not honor unittest's
# load_tests protocol — cannot discover and instantiate it when viser is
# unavailable. This keeps the OSS pytest path silent on missing viser while
# still letting BUCK exercise the suite normally (viser is a hard dep there).
if _HAS_VISER:  # noqa: C901 — gating block intentionally wraps the whole suite
    from pymomentum.viser_vis import (
        _CollisionBoxHandles,
        _CollisionCapsuleHandles,
        _CollisionEllipsoidHandles,
        _max_nonzero_skin_influences,
        _normalize_param_limit,
        _normalize_skin_weights,
        _xyzw_to_wxyz,
        add_character,
        add_character_param_sliders,
        add_collision_geometry,
        add_joints,
        add_log_panel,
        add_mesh,
        add_skinned_mesh,
        CharacterHandles,
        remove_character,
        update_character,
        update_collision_geometry,
        update_joints,
        update_joints_transforms,
        update_mesh,
        update_skinned_mesh,
        update_skinned_mesh_transforms,
    )

    class TestViserVis(unittest.TestCase):
        @classmethod
        def setUpClass(cls) -> None:
            # Ephemeral port; the server runs locally only for handle creation.
            cls.server: viser.ViserServer = viser.ViserServer(host="127.0.0.1", port=0)

        @classmethod
        def tearDownClass(cls) -> None:
            cls.server.stop()

        def _make_character_and_skel_state(
            self,
        ) -> tuple[geo.Character, np.ndarray]:
            character = test_utils.create_test_character()
            self.assertGreater(character.skeleton.size, 0)
            self.assertTrue(
                character.has_mesh,
                "create_test_character() is documented to return a mesh",
            )
            model_params = np.zeros(
                character.parameter_transform.size, dtype=np.float32
            )
            skel_state = geo.model_parameters_to_skeleton_state(character, model_params)
            self.assertEqual(skel_state.shape, (character.skeleton.size, 8))
            return character, skel_state

        def _make_collision_character_and_skel_state(
            self,
        ) -> tuple[geo.Character, np.ndarray]:
            character, _ = self._make_character_and_skel_state()
            capsule = geo.TaperedCapsule(
                parent=0,
                transformation=np.eye(4, dtype=np.float32),
                radius=np.array([1.0, 2.0], dtype=np.float32),
                length=10.0,
            )
            character = character.with_collision_geometry([capsule])
            self.assertEqual(len(character.collision_geometry), 1)

            skel_state = np.zeros((character.skeleton.size, 8), dtype=np.float32)
            skel_state[:, 6] = 1.0
            skel_state[:, 7] = 1.0
            return character, skel_state

        def _make_mixed_collision_character_and_skel_state(
            self,
        ) -> tuple[geo.Character, np.ndarray]:
            """Build a character with one capsule, one ellipsoid, and one box.

            Each primitive has an identity local transform and a distinct parent
            joint, so its world placement is fully determined by that joint's
            state. The skel_state gives each joint a distinct translation, an
            identity rotation, and a distinct (non-unit) scale so parent-scale
            paths are exercised and independently verifiable.
            """
            character, _ = self._make_character_and_skel_state()
            character = character.with_collision_geometry(
                [
                    geo.TaperedCapsule(
                        parent=0,
                        transformation=np.eye(4, dtype=np.float32),
                        radius=np.array([1.0, 2.0], dtype=np.float32),
                        length=10.0,
                    ),
                    geo.Ellipsoid(
                        parent=1,
                        transformation=np.eye(4, dtype=np.float32),
                        radii=np.array([3.0, 2.0, 1.0], dtype=np.float32),
                    ),
                    geo.Box(
                        parent=2,
                        transformation=np.eye(4, dtype=np.float32),
                        half_extents=np.array([4.0, 3.0, 2.0], dtype=np.float32),
                    ),
                ]
            )
            self.assertEqual(len(character.collision_geometry), 3)
            self.assertGreaterEqual(character.skeleton.size, 3)

            skel_state = np.zeros((character.skeleton.size, 8), dtype=np.float32)
            skel_state[:, 0] = np.arange(character.skeleton.size, dtype=np.float32)
            skel_state[:, 6] = 1.0
            skel_state[:, 7] = 1.0 + 0.5 * np.arange(
                character.skeleton.size, dtype=np.float32
            )
            return character, skel_state

        def _posed_mesh(self, character: geo.Character) -> geo.Mesh:
            joint_params = np.zeros(character.skeleton.size * 7, dtype=np.float32)
            return character.pose_mesh(joint_params)

        def _cleanup_collision_handles(self, root: Any, handles: list[Any]) -> None:
            for primitive_handles in handles:
                if isinstance(primitive_handles, _CollisionCapsuleHandles):
                    primitive_handles.cylinder.remove()
                    primitive_handles.start_sphere.remove()
                    primitive_handles.end_sphere.remove()
                elif isinstance(primitive_handles, _CollisionEllipsoidHandles):
                    primitive_handles.ellipsoid.remove()
                else:
                    primitive_handles.box.remove()
            root.remove()

        def _make_uncolored_character(self, character: geo.Character) -> geo.Character:
            mesh = geo.Mesh(
                vertices=character.mesh.vertices,
                faces=character.mesh.faces,
                normals=character.mesh.normals,
            )
            return character.with_mesh_and_skin_weights(mesh, character.skin_weights)

        def _make_colored_skinned_character_and_skel_state(
            self,
        ) -> tuple[geo.Character, np.ndarray]:
            character, _ = self._make_character_and_skel_state()
            vertices = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
            colors = np.array(
                [
                    [255, 0, 0],
                    [255, 0, 0],
                    [255, 0, 0],
                    [0, 0, 255],
                    [0, 0, 255],
                    [0, 0, 255],
                ],
                dtype=np.uint8,
            )
            mesh = geo.Mesh(vertices=vertices, faces=faces, colors=colors)
            skin_weights = geo.SkinWeights(
                index=np.zeros((vertices.shape[0], 1), dtype=np.int32),
                weights=np.ones((vertices.shape[0], 1), dtype=np.float32),
            )
            character = character.with_mesh_and_skin_weights(mesh, skin_weights)
            model_params = np.zeros(
                character.parameter_transform.size, dtype=np.float32
            )
            skel_state = geo.model_parameters_to_skeleton_state(character, model_params)
            return character, skel_state

        def test_add_and_update_joints(self) -> None:
            character, skel_state = self._make_character_and_skel_state()
            handles, bones_handle = add_joints(
                self.server, "test_joints", character, skel_state
            )
            self.assertEqual(set(handles.keys()), set(character.skeleton.joint_names))
            # The test character has parent-child joints, so a bone handle exists.
            self.assertIsNotNone(bones_handle)

            # Move every joint by +1 cm along x; positions in handles should follow.
            moved = skel_state.copy()
            moved[:, 0] += 1.0
            update_joints(handles, bones_handle, character, moved, scale=1.0)
            for i, name in enumerate(character.skeleton.joint_names):
                np.testing.assert_allclose(
                    np.asarray(handles[name].position),
                    moved[i, :3],
                    atol=1e-5,
                )

            positions = np.asarray(moved[:, :3] * 2.0, dtype=np.float32)
            wxyzs = np.empty((moved.shape[0], 4), dtype=np.float32)
            wxyzs[:, 0] = moved[:, 6]
            wxyzs[:, 1:] = moved[:, 3:6]
            expected_segments: list[list[np.ndarray]] = []
            for child, parent in enumerate(character.skeleton.joint_parents):
                parent_index = int(parent)
                if 0 <= parent_index < positions.shape[0]:
                    expected_segments.append(
                        [positions[parent_index], positions[child]]
                    )
            bone_segments = np.asarray(expected_segments, dtype=np.float32)
            update_joints_transforms(
                handles,
                bones_handle,
                character,
                positions,
                wxyzs,
                bone_segments,
            )
            for i, name in enumerate(character.skeleton.joint_names):
                np.testing.assert_allclose(
                    np.asarray(handles[name].position),
                    positions[i],
                    atol=1e-5,
                )
                np.testing.assert_allclose(
                    np.asarray(handles[name].wxyz),
                    wxyzs[i],
                    atol=1e-5,
                )
            np.testing.assert_allclose(
                np.asarray(bones_handle.points), bone_segments, atol=1e-5
            )

            inferred_positions = np.asarray(moved[:, :3] * 3.0, dtype=np.float32)
            update_joints_transforms(
                handles,
                bones_handle,
                character,
                inferred_positions,
                wxyzs,
            )
            expected_inferred_segments = []
            for child, parent in enumerate(character.skeleton.joint_parents):
                parent_index = int(parent)
                if 0 <= parent_index < inferred_positions.shape[0]:
                    expected_inferred_segments.append(
                        [inferred_positions[parent_index], inferred_positions[child]]
                    )
            np.testing.assert_allclose(
                np.asarray(bones_handle.points),
                np.asarray(expected_inferred_segments, dtype=np.float32),
                atol=1e-5,
            )

            for h in handles.values():
                h.remove()
            bones_handle.remove()

        def test_add_and_update_mesh(self) -> None:
            character, _ = self._make_character_and_skel_state()
            posed_mesh = self._posed_mesh(character)
            self.assertGreater(len(posed_mesh.vertices), 0)
            posed_mesh = geo.Mesh(
                vertices=posed_mesh.vertices,
                faces=posed_mesh.faces,
                normals=posed_mesh.normals,
            )
            solid_handle, wireframe_handle = add_mesh(
                self.server, "test_mesh", posed_mesh
            )

            # update_mesh should propagate new vertices through both handles.
            scale = 2.0
            update_mesh(solid_handle, wireframe_handle, posed_mesh, scale=scale)
            expected = np.asarray(posed_mesh.vertices, dtype=np.float32) * scale
            np.testing.assert_allclose(np.asarray(solid_handle.vertices), expected)
            np.testing.assert_allclose(np.asarray(wireframe_handle.vertices), expected)

            solid_handle.remove()
            wireframe_handle.remove()

        def test_add_and_update_skinned_mesh(self) -> None:
            character, skel_state = self._make_character_and_skel_state()
            character = self._make_uncolored_character(character)
            skinned_handle = add_skinned_mesh(
                self.server, "test_skinned_mesh", character, skel_state
            )
            self.assertGreaterEqual(len(skinned_handle.bones), character.skeleton.size)

            moved = skel_state.copy()
            moved[:, 0] += 1.0
            scale = 2.0
            update_skinned_mesh(skinned_handle, moved, scale=scale)
            for i, bone in enumerate(skinned_handle.bones[: character.skeleton.size]):
                np.testing.assert_allclose(
                    np.asarray(bone.position),
                    moved[i, :3] * scale,
                    atol=1e-5,
                )

            direct_positions = np.asarray(
                moved[:, :3] * (scale + 1.0), dtype=np.float32
            )
            direct_wxyzs = np.empty((moved.shape[0], 4), dtype=np.float32)
            direct_wxyzs[:, 0] = moved[:, 6]
            direct_wxyzs[:, 1:] = moved[:, 3:6]
            update_skinned_mesh_transforms(
                skinned_handle,
                direct_positions,
                direct_wxyzs,
            )
            for i, bone in enumerate(skinned_handle.bones[: character.skeleton.size]):
                np.testing.assert_allclose(
                    np.asarray(bone.position),
                    direct_positions[i],
                    atol=1e-5,
                )
                np.testing.assert_allclose(
                    np.asarray(bone.wxyz),
                    direct_wxyzs[i],
                    atol=1e-5,
                )

            skinned_handle.remove()

        def test_add_and_update_skinned_mesh_with_vertex_colors(self) -> None:
            character, skel_state = (
                self._make_colored_skinned_character_and_skel_state()
            )
            skinned_handle = add_skinned_mesh(
                self.server, "test_color_skinned_mesh", character, skel_state
            )
            self.assertIsInstance(skinned_handle, list)
            self.assertEqual(len(skinned_handle), 2)
            self.assertEqual(
                {tuple(part.handle.color) for part in skinned_handle},
                {(255, 0, 0), (0, 0, 255)},
            )

            moved = skel_state.copy()
            moved[:, 0] += 1.0
            scale = 2.0
            update_skinned_mesh(skinned_handle, moved, scale=scale)
            for part in skinned_handle:
                self.assertGreaterEqual(len(part.handle.bones), character.skeleton.size)
                for i, bone in enumerate(part.handle.bones[: character.skeleton.size]):
                    np.testing.assert_allclose(
                        np.asarray(bone.position),
                        moved[i, :3] * scale,
                        atol=1e-5,
                    )

            for part in skinned_handle:
                part.handle.remove()

        def test_add_and_update_mesh_with_vertex_colors(self) -> None:
            vertices = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
            colors = np.array(
                [
                    [255, 0, 0],
                    [255, 0, 0],
                    [255, 0, 0],
                    [0, 0, 255],
                    [0, 0, 255],
                    [0, 0, 255],
                ],
                dtype=np.uint8,
            )
            mesh = geo.Mesh(vertices=vertices, faces=faces, colors=colors)

            solid_handle, wireframe_handle = add_mesh(
                self.server, "test_color_mesh", mesh
            )
            self.assertIsInstance(solid_handle, list)
            self.assertEqual(len(solid_handle), 2)
            self.assertEqual(
                {tuple(part.handle.color) for part in solid_handle},
                {(255, 0, 0), (0, 0, 255)},
            )

            moved_mesh = geo.Mesh(
                vertices=vertices + np.array([1.0, 2.0, 3.0], dtype=np.float32),
                faces=faces,
                colors=colors,
            )
            update_mesh(solid_handle, wireframe_handle, moved_mesh, scale=1.0)
            moved_vertices = np.asarray(moved_mesh.vertices, dtype=np.float32)
            for part in solid_handle:
                np.testing.assert_allclose(
                    np.asarray(part.handle.vertices),
                    moved_vertices[part.vertex_indices],
                )
            np.testing.assert_allclose(
                np.asarray(wireframe_handle.vertices), moved_vertices
            )

            for part in solid_handle:
                part.handle.remove()
            wireframe_handle.remove()

        def test_add_character_with_and_without_mesh(self) -> None:
            character, skel_state = self._make_character_and_skel_state()
            posed_mesh = self._posed_mesh(character)

            # With mesh: handle bag should include both mesh and wireframe handles.
            full = add_character(
                self.server,
                "test_char_full",
                character,
                skel_state,
                posed_mesh=posed_mesh,
            )
            self.assertIsInstance(full, CharacterHandles)
            self.assertEqual(len(full.joint_frames), character.skeleton.size)
            self.assertIsNotNone(full.mesh_handle)
            self.assertIsNotNone(full.wireframe_handle)
            # update_character should run without raising; verify the joint
            # positions follow the new skel state.
            moved = skel_state.copy()
            moved[:, 0] += 0.5
            update_character(self.server, full, character, moved, posed_mesh=posed_mesh)
            for i, name in enumerate(character.skeleton.joint_names):
                np.testing.assert_allclose(
                    np.asarray(full.joint_frames[name].position),
                    moved[i, :3] * 0.01,  # default CM_TO_M scale
                    atol=1e-5,
                )
            remove_character(full)
            self.assertEqual(len(full.joint_frames), 0)
            self.assertIsNone(full.mesh_handle)

            # Without mesh: mesh/wireframe handles should be None.
            no_mesh = add_character(self.server, "test_char_nm", character, skel_state)
            self.assertIsNone(no_mesh.mesh_handle)
            self.assertIsNone(no_mesh.wireframe_handle)
            self.assertGreater(len(no_mesh.joint_frames), 0)
            remove_character(no_mesh)

            # Skinned mesh: no secondary wireframe handle is needed because the
            # skinned mesh handle exposes a wireframe property directly.
            uncolored_character = self._make_uncolored_character(character)
            skinned = add_character(
                self.server,
                "test_char_skinned",
                uncolored_character,
                skel_state,
                use_skinned_mesh=True,
            )
            self.assertIsNotNone(skinned.mesh_handle)
            self.assertIsNone(skinned.wireframe_handle)
            self.assertGreaterEqual(
                len(skinned.mesh_handle.bones), uncolored_character.skeleton.size
            )
            moved[:, 1] += 0.25
            update_character(self.server, skinned, uncolored_character, moved)
            for i, bone in enumerate(
                skinned.mesh_handle.bones[: uncolored_character.skeleton.size]
            ):
                np.testing.assert_allclose(
                    np.asarray(bone.position),
                    moved[i, :3] * 0.01,
                    atol=1e-5,
                )
            remove_character(skinned)

        def test_collision_proxies_track_model_revealed_via_scene_tree(self) -> None:
            # Regression: proxy visibility is driven solely by viser's scene-tree
            # panel, which flips the root frame's visibility directly. Because
            # update_character repositions proxies regardless of visibility, a
            # proxy revealed after the model moved is already at the current pose
            # rather than stranded at the rest pose.
            character, skel_state = self._make_collision_character_and_skel_state()
            handles = add_character(
                self.server,
                "scene_tree_char",
                character,
                skel_state,
                collision_geometry=True,
                collision_geometry_visible=False,
            )
            root = handles.collision_root_handle
            self.assertIsNotNone(root)
            assert root is not None
            self.assertFalse(root.visible)
            capsule_handles = handles.collision_handles[0]

            # Move the model while the proxies remain hidden.
            moved = skel_state.copy()
            moved[0, 0] = 10.0
            update_character(self.server, handles, character, moved)

            # Reveal via the scene-tree path: flip root visibility directly, with
            # no checkbox/on_show refresh.
            root.visible = True

            # Proxy is at the moved pose, not stranded at the rest pose.
            np.testing.assert_allclose(
                np.asarray(capsule_handles.start_sphere.position),
                [0.1, 0.0, 0.0],
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.asarray(capsule_handles.end_sphere.position),
                [0.2, 0.0, 0.0],
                atol=1e-5,
            )
            remove_character(handles)

        def test_add_collision_geometry_mixed_primitives(self) -> None:
            character, skel_state = (
                self._make_mixed_collision_character_and_skel_state()
            )
            ellipsoid_rotation = np.array(
                [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32
            )
            box_rotation = np.array(
                [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32
            )
            skel_state[1, 3:7] = ellipsoid_rotation
            skel_state[2, 3:7] = box_rotation
            # scale=1.0 so scene units equal momentum units for easy verification.
            root, handles = add_collision_geometry(
                self.server, "mixed_collision", character, skel_state, scale=1.0
            )
            self.assertIsNotNone(root)
            self.assertEqual(len(handles), 3)
            self.assertIsInstance(handles[0], _CollisionCapsuleHandles)
            self.assertIsInstance(handles[1], _CollisionEllipsoidHandles)
            self.assertIsInstance(handles[2], _CollisionBoxHandles)

            # Ellipsoid (parent joint 1, identity local transform): centered at
            # the parent translation. The ellipsoid is a unit-sphere mesh whose
            # vertices are baked to the per-axis half-sizes (radii scaled by the
            # parent joint scale), so its per-axis extent equals those half-sizes.
            ellipsoid = handles[1].ellipsoid
            np.testing.assert_allclose(
                np.asarray(ellipsoid.position), skel_state[1, :3], atol=1e-5
            )
            # The UV-sphere tessellation samples discrete longitudes, so the
            # measured per-axis extent is at most the true half-size and within a
            # couple percent of it; the proportions still encode the radii.
            ellipsoid_extent = np.max(np.abs(np.asarray(ellipsoid.vertices)), axis=0)
            expected_radii = np.array([3.0, 2.0, 1.0]) * skel_state[1, 7]
            np.testing.assert_array_less(ellipsoid_extent, expected_radii + 1e-4)
            np.testing.assert_allclose(ellipsoid_extent, expected_radii, rtol=0.02)
            np.testing.assert_allclose(
                np.asarray(ellipsoid.wxyz),
                [ellipsoid_rotation[3], *ellipsoid_rotation[:3]],
                atol=1e-5,
            )

            # Box (parent joint 2): centered at the parent translation, with
            # full dimensions = 2 * half_extents * parent joint scale.
            box = handles[2].box
            np.testing.assert_allclose(
                np.asarray(box.position), skel_state[2, :3], atol=1e-5
            )
            np.testing.assert_allclose(
                np.asarray(box.dimensions),
                np.array([8.0, 6.0, 4.0]) * skel_state[2, 7],
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.asarray(box.wxyz), [box_rotation[3], *box_rotation[:3]], atol=1e-5
            )

            self._cleanup_collision_handles(root, handles)

        def test_update_collision_geometry_mixed_primitives(self) -> None:
            character, skel_state = (
                self._make_mixed_collision_character_and_skel_state()
            )
            _root, handles = add_collision_geometry(
                self.server, "mixed_collision_update", character, skel_state, scale=1.0
            )

            ellipsoid = handles[1].ellipsoid
            box = handles[2].box
            ellipsoid_before = np.asarray(ellipsoid.position).copy()
            box_before = np.asarray(box.position).copy()

            # Shift every joint by a known amount; primitives must follow.
            shift = 7.0
            moved = skel_state.copy()
            moved[:, 0] += shift
            update_collision_geometry(handles, character, moved, scale=1.0)

            np.testing.assert_allclose(
                np.asarray(ellipsoid.position),
                ellipsoid_before + np.array([shift, 0.0, 0.0]),
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.asarray(box.position),
                box_before + np.array([shift, 0.0, 0.0]),
                atol=1e-5,
            )

            self._cleanup_collision_handles(_root, handles)

        def test_add_collision_geometry_world_space_ellipsoid_and_box(self) -> None:
            # World-fixed (parent=-1) ellipsoid and box with non-identity local rotation
            # and translation. Complements test_add_collision_geometry_mixed_primitives
            # (which parents the primitives to joints): this exercises the identity-parent
            # fallback in viser_vis._collision_primitive_world_transform (world = identity
            # @ local, parent scale 1.0), which the parented test does not cover.
            character, skel_state = self._make_character_and_skel_state()

            ellipsoid_rotation = np.array(
                [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32
            )
            box_rotation = np.array(
                [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32
            )
            ellipsoid_translation = np.array([3.0, 4.0, 5.0], dtype=np.float32)
            box_translation = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
            ellipsoid_radii = np.array([3.0, 2.0, 1.0], dtype=np.float32)
            box_half_extents = np.array([4.0, 3.0, 2.0], dtype=np.float32)

            ellipsoid_transform = np.eye(4, dtype=np.float32)
            ellipsoid_transform[:3, :3] = to_rotation_matrix_assume_normalized(
                ellipsoid_rotation[None, :]
            )[0]
            ellipsoid_transform[:3, 3] = ellipsoid_translation
            box_transform = np.eye(4, dtype=np.float32)
            box_transform[:3, :3] = to_rotation_matrix_assume_normalized(
                box_rotation[None, :]
            )[0]
            box_transform[:3, 3] = box_translation

            character = character.with_collision_geometry(
                [
                    geo.Ellipsoid(
                        parent=-1,
                        transformation=ellipsoid_transform,
                        radii=ellipsoid_radii,
                    ),
                    geo.Box(
                        parent=-1,
                        transformation=box_transform,
                        half_extents=box_half_extents,
                    ),
                ]
            )

            # scale=1.0 so scene units equal momentum units for easy verification.
            root, handles = add_collision_geometry(
                self.server, "world_collision", character, skel_state, scale=1.0
            )
            self.assertEqual(len(handles), 2)
            self.assertIsInstance(handles[0], _CollisionEllipsoidHandles)
            self.assertIsInstance(handles[1], _CollisionBoxHandles)

            # Identity parent transform: world center == local translation, the per-axis
            # extent encodes the raw radii (parent scale 1.0), and the handle orientation
            # equals the local rotation.
            ellipsoid = handles[0].ellipsoid
            np.testing.assert_allclose(
                np.asarray(ellipsoid.position), ellipsoid_translation, atol=1e-5
            )
            ellipsoid_extent = np.max(np.abs(np.asarray(ellipsoid.vertices)), axis=0)
            np.testing.assert_array_less(ellipsoid_extent, ellipsoid_radii + 1e-4)
            np.testing.assert_allclose(ellipsoid_extent, ellipsoid_radii, rtol=0.02)
            np.testing.assert_allclose(
                np.asarray(ellipsoid.wxyz),
                [ellipsoid_rotation[3], *ellipsoid_rotation[:3]],
                atol=1e-5,
            )

            # Box: full dimensions = 2 * half_extents (parent joint scale 1.0).
            box = handles[1].box
            np.testing.assert_allclose(
                np.asarray(box.position), box_translation, atol=1e-5
            )
            np.testing.assert_allclose(
                np.asarray(box.dimensions), 2.0 * box_half_extents, atol=1e-5
            )
            np.testing.assert_allclose(
                np.asarray(box.wxyz), [box_rotation[3], *box_rotation[:3]], atol=1e-5
            )

            self._cleanup_collision_handles(root, handles)

        def test_add_character_includes_mixed_collision_geometry(self) -> None:
            character, skel_state = (
                self._make_mixed_collision_character_and_skel_state()
            )
            handles = add_character(
                self.server,
                "char_mixed_collision",
                character,
                skel_state,
                collision_geometry=True,
            )
            self.assertEqual(len(handles.collision_handles), 3)
            self.assertIsInstance(
                handles.collision_handles[1], _CollisionEllipsoidHandles
            )
            self.assertIsInstance(handles.collision_handles[2], _CollisionBoxHandles)

            box = handles.collision_handles[2].box
            box_before = float(np.asarray(box.position)[0])
            moved = skel_state.copy()
            moved[:, 0] += 5.0
            update_character(self.server, handles, character, moved)
            expected_shift = 5.0 * 0.01  # default CM_TO_M scale
            np.testing.assert_allclose(
                float(np.asarray(box.position)[0]),
                box_before + expected_shift,
                atol=1e-5,
            )

            remove_character(handles)
            self.assertEqual(handles.collision_handles, [])

        def test_xyzw_to_wxyz(self) -> None:
            # xyzw [0.1, 0.2, 0.3, 0.9] -> wxyz [0.9, 0.1, 0.2, 0.3]
            q_xyzw = np.array([0.1, 0.2, 0.3, 0.9])
            np.testing.assert_allclose(
                _xyzw_to_wxyz(q_xyzw), [0.9, 0.1, 0.2, 0.3], atol=1e-6
            )

        def test_normalize_param_limit(self) -> None:
            # Bounded limits pass through unchanged.
            self.assertEqual(_normalize_param_limit("foo", -1.5, 1.5), (-1.5, 1.5))
            # Translation params get clamped to ±200 cm.
            self.assertEqual(
                _normalize_param_limit("root_tx", -1e35, 1e35), (-200.0, 200.0)
            )
            # Rotation params get clamped to ±π.
            lo, hi = _normalize_param_limit("hip_rx", -1e35, 1e35)
            self.assertAlmostEqual(lo, -math.pi)
            self.assertAlmostEqual(hi, math.pi)

        def test_skin_weight_influence_helpers(self) -> None:
            weights = np.array(
                [
                    [0.40, 0.30, 0.20, 0.05, 0.05],
                    [0.00, 0.00, 0.00, 0.00, 0.00],
                ],
                dtype=np.float32,
            )
            result = _normalize_skin_weights(weights)

            self.assertEqual(_max_nonzero_skin_influences(weights), 5)
            np.testing.assert_allclose(result[0], weights[0], atol=1e-6)
            np.testing.assert_allclose(result[0].sum(), 1.0, atol=1e-6)
            np.testing.assert_allclose(result[1], np.zeros(5, dtype=np.float32))

        def test_add_character_param_sliders(self) -> None:
            character, _ = self._make_character_and_skel_state()
            n_params = character.parameter_transform.size
            events: list[object] = []

            def _cb(evt: object) -> None:
                events.append(evt)

            sliders = add_character_param_sliders(self.server, character, on_change=_cb)
            self.assertEqual(len(sliders), n_params)
            # Verify the callback is wired by mutating one slider's value (which
            # triggers on_update synchronously in viser).
            sliders[0].value = sliders[0].min
            self.assertGreater(len(events), 0)
            for s in sliders:
                s.remove()

        def test_add_log_panel(self) -> None:
            panel = add_log_panel(self.server, echo_to_stdout=False)
            panel.log("hello")
            panel.log("uh oh", level="error")
            # The panel renders into an HTML handle; assert the HTML content
            # picked up both messages and the error icon.
            # pyre-ignore[16]: _handle is set in __init__
            content = panel._handle.content
            self.assertIn("hello", content)
            self.assertIn("uh oh", content)
            self.assertIn("🔴", content)
            # pyre-ignore[16]
            panel._handle.remove()


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Skip when viser is unavailable.

    The class-level `if _HAS_VISER` gate above is what protects the OSS pytest
    path (pytest does not honor this protocol). This hook is what BUCK's TPX
    runner uses to enumerate tests when ``supports_static_listing = False`` —
    without it, BUCK reports 0 tests even when the class is defined.
    """
    if not _HAS_VISER:
        return unittest.TestSuite()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestViserVis))
    return suite
