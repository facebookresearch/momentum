# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pymomentum.geometry as geo  # @manual=:geometry
import pymomentum.geometry_test_utils as test_utils  # @manual=:geometry_test_utils
from pymomentum.quaternion_np import to_rotation_matrix_assume_normalized
from pymomentum.rerun_vis import (
    log_animation,
    log_character,
    log_collision_geometry,
    log_joints,
    log_locators,
    log_mesh,
)

try:
    import rerun as rr

    _HAS_RERUN = True
except ImportError:
    rr = None  # pyre-ignore[9]
    _HAS_RERUN = False


def _is_tsan_active() -> bool:
    # rerun-sdk's Rust internals (RecordingStream + ChunkBatcher background
    # threads) trigger ThreadSanitizer warnings on its own shared state. The
    # warnings happen inside the SDK and cannot be fixed from Python; skipping
    # the whole module under TSan keeps the binary from exiting non-zero.
    # Detect TSan by inspecting the loaded shared libraries — TSAN_OPTIONS is
    # not set by Meta's TSan build mode.
    try:
        return "tsan" in Path("/proc/self/maps").read_text()
    except OSError:
        return False


# The TestRerunVis class is gated behind `_HAS_RERUN` (rather than skipped via
# load_tests() alone) so that pytest — which does not honor unittest's
# load_tests protocol — cannot discover and instantiate it when rerun is
# unavailable. load_tests is kept for the TSan skip path (BUCK / unittest only).
if _HAS_RERUN:  # noqa: C901 — gating block intentionally wraps the whole suite

    class TestRerunVis(unittest.TestCase):
        @classmethod
        def setUpClass(cls) -> None:
            # Reuse one RecordingStream across all log tests. Each new stream spins
            # up its own RecordingStream + ChunkBatcher background threads; sharing
            # one stream keeps thread count bounded and the test fast.
            cls._rec: rr.RecordingStream = rr.RecordingStream("test_rerun_vis")

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
            capsule_transform = np.eye(4, dtype=np.float32)
            capsule = geo.TaperedCapsule(
                parent=0,
                transformation=capsule_transform,
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
            identity rotation, and a distinct (non-unit) scale so the parent-scale
            convention is exercised and independently verifiable.
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
            # Distinct translation per joint and identity rotation.
            skel_state[:, 0] = np.arange(character.skeleton.size, dtype=np.float32)
            skel_state[:, 6] = 1.0
            # Distinct, non-unit per-joint scale to exercise parent-scale paths.
            skel_state[:, 7] = 1.0 + 0.5 * np.arange(
                character.skeleton.size, dtype=np.float32
            )
            return character, skel_state

        def _posed_mesh(self, character: geo.Character) -> geo.Mesh:
            joint_params = np.zeros(character.skeleton.size * 7, dtype=np.float32)
            return character.pose_mesh(joint_params)

        def _component_array(self, component_batch: Any) -> np.ndarray:
            self.assertIsNotNone(component_batch)
            return np.asarray(component_batch.as_arrow_array().to_pylist())

        def test_log_joints_uses_skeleton_state(self) -> None:
            character, skel_state = self._make_character_and_skel_state()
            # Smoke check: the call should consume the full skel_state and not
            # raise. There is no public API to introspect what was logged, so we
            # also re-run with an obviously-bad shape to confirm input validation
            # propagates a real error rather than silently corrupting the stream.
            log_joints(self._rec, "test/joints", character, skel_state)
            with self.assertRaises((ValueError, IndexError, AssertionError)):
                log_joints(self._rec, "test/joints_bad", character, np.zeros((1, 8)))

        def test_log_mesh_default_and_custom_color(self) -> None:
            character, _ = self._make_character_and_skel_state()
            posed_mesh = self._posed_mesh(character)
            self.assertGreater(len(posed_mesh.vertices), 0)
            self.assertGreater(len(posed_mesh.faces), 0)

            mock_rec = MagicMock()
            log_mesh(mock_rec, "m", posed_mesh)
            mock_rec.log.assert_called_once()
            path, archetype = mock_rec.log.call_args.args
            self.assertEqual(path, "m")
            self.assertIsInstance(archetype, rr.Mesh3D)

            mock_rec = MagicMock()
            log_mesh(mock_rec, "m_color", posed_mesh, color=(255, 0, 0))
            path, archetype = mock_rec.log.call_args.args
            self.assertEqual(path, "m_color")
            self.assertIsInstance(archetype, rr.Mesh3D)

        def test_log_locators(self) -> None:
            character, skel_state = self._make_character_and_skel_state()
            self.assertGreater(len(character.locators), 0)

            mock_rec = MagicMock()
            log_locators(mock_rec, "locs", character, skel_state)
            mock_rec.log.assert_called_once()
            path, archetype = mock_rec.log.call_args.args
            self.assertEqual(path, "locs")
            self.assertIsInstance(archetype, rr.Points3D)

        def test_log_collision_geometry(self) -> None:
            character, skel_state = self._make_collision_character_and_skel_state()

            mock_rec = MagicMock()
            log_collision_geometry(mock_rec, "collisions", character, skel_state)

            mock_rec.log.assert_called_once()
            path, archetype = mock_rec.log.call_args.args
            self.assertEqual(path, "collisions")
            self.assertIsInstance(archetype, rr.Capsules3D)
            self.assertIsNotNone(archetype.colors)
            assert archetype.colors is not None
            expected_color = (128 << 24) | (64 << 16) | (64 << 8) | 0xFF
            self.assertEqual(
                archetype.colors.as_arrow_array().to_pylist(), [expected_color]
            )
            self.assertIsNotNone(archetype.fill_mode)
            assert archetype.fill_mode is not None
            self.assertEqual(
                archetype.fill_mode.as_arrow_array().to_pylist(),
                [rr.components.FillMode.Solid.value],
            )

        def test_log_collision_geometry_world_space_capsule(self) -> None:
            character, skel_state = self._make_character_and_skel_state()
            capsule_transform = np.eye(4, dtype=np.float32)
            capsule_transform[:3, 3] = np.array([3.0, 4.0, 5.0], dtype=np.float32)
            capsule = geo.TaperedCapsule(
                parent=-1,
                transformation=capsule_transform,
                radius=np.array([0.5, 0.75], dtype=np.float32),
                length=2.0,
            )
            character = character.with_collision_geometry([capsule])

            mock_rec = MagicMock()
            log_collision_geometry(mock_rec, "collisions", character, skel_state)

            mock_rec.log.assert_called_once()
            path, archetype = mock_rec.log.call_args.args
            self.assertEqual(path, "collisions")
            self.assertIsInstance(archetype, rr.Capsules3D)
            np.testing.assert_allclose(
                self._component_array(archetype.translations),
                np.array([[3.0, 4.0, 5.0]], dtype=np.float32),
                atol=1e-5,
            )
            np.testing.assert_allclose(
                self._component_array(archetype.lengths),
                np.array([2.0], dtype=np.float32),
                atol=1e-5,
            )
            np.testing.assert_allclose(
                self._component_array(archetype.radii),
                np.array([0.75], dtype=np.float32),
                atol=1e-5,
            )

        def test_log_collision_geometry_ellipsoid_and_box(self) -> None:
            character, skel_state = (
                self._make_mixed_collision_character_and_skel_state()
            )
            ellipsoid_parent = 1
            box_parent = 2
            ellipsoid_rotation = np.array(
                [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32
            )
            box_rotation = np.array(
                [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32
            )
            skel_state[ellipsoid_parent, 3:7] = ellipsoid_rotation
            skel_state[box_parent, 3:7] = box_rotation

            mock_rec = MagicMock()
            log_collision_geometry(mock_rec, "collisions", character, skel_state)
            logged = {c.args[0]: c.args[1] for c in mock_rec.log.call_args_list}

            # Capsules log at the root path; ellipsoids/boxes at sub-paths.
            self.assertIn("collisions", logged)
            self.assertIsInstance(logged["collisions"], rr.Capsules3D)
            ellipsoids = logged["collisions/ellipsoids"]
            boxes = logged["collisions/boxes"]
            self.assertIsInstance(ellipsoids, rr.Ellipsoids3D)
            self.assertIsInstance(boxes, rr.Boxes3D)

            # The ellipsoid's parent joint is index 1; the box's is index 2.
            # Both have an identity local transform, so the world center equals
            # the parent joint translation and the orientation equals the
            # parent joint rotation. Half-sizes scale by the parent joint scale only.
            ellipsoid_scale = float(skel_state[ellipsoid_parent, 7])
            box_scale = float(skel_state[box_parent, 7])

            ellipsoid_centers = self._component_array(ellipsoids.centers)
            self.assertEqual(ellipsoid_centers.shape, (1, 3))
            np.testing.assert_allclose(
                ellipsoid_centers[0], skel_state[ellipsoid_parent, :3], atol=1e-5
            )
            np.testing.assert_allclose(
                self._component_array(ellipsoids.half_sizes),
                np.array([[3.0, 2.0, 1.0]], dtype=np.float32) * ellipsoid_scale,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                self._component_array(ellipsoids.quaternions),
                ellipsoid_rotation[None, :],
                atol=1e-5,
            )

            box_centers = self._component_array(boxes.centers)
            self.assertEqual(box_centers.shape, (1, 3))
            np.testing.assert_allclose(
                box_centers[0], skel_state[box_parent, :3], atol=1e-5
            )
            np.testing.assert_allclose(
                self._component_array(boxes.half_sizes),
                np.array([[4.0, 3.0, 2.0]], dtype=np.float32) * box_scale,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                self._component_array(boxes.quaternions),
                box_rotation[None, :],
                atol=1e-5,
            )

            # Ellipsoids and boxes carry the same collision color and solid fill.
            expected_color = (128 << 24) | (64 << 16) | (64 << 8) | 0xFF
            for archetype in (ellipsoids, boxes):
                self.assertEqual(
                    self._component_array(archetype.colors).tolist(), [expected_color]
                )
                self.assertEqual(
                    self._component_array(archetype.fill_mode).tolist(),
                    [rr.components.FillMode.Solid.value],
                )

        def test_log_collision_geometry_world_space_ellipsoid_and_box(self) -> None:
            # World-fixed (parent=-1) ellipsoid and box with non-identity local rotation
            # and translation. Complements test_log_collision_geometry_ellipsoid_and_box
            # (which parents the primitives to joints): this exercises the identity-parent
            # fallback in _collision_primitive_world_transform (world = identity @ local,
            # parent scale 1.0) and the from_rotation_matrix orientation extraction for
            # these primitive types, which the parented test does not cover.
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

            mock_rec = MagicMock()
            log_collision_geometry(mock_rec, "collisions", character, skel_state)
            logged = {c.args[0]: c.args[1] for c in mock_rec.log.call_args_list}

            ellipsoids = logged["collisions/ellipsoids"]
            boxes = logged["collisions/boxes"]
            self.assertIsInstance(ellipsoids, rr.Ellipsoids3D)
            self.assertIsInstance(boxes, rr.Boxes3D)

            # Identity parent transform: world center == local translation, half sizes ==
            # raw radii/half-extents (parent scale 1.0), and the logged quaternion
            # round-trips the local rotation.
            np.testing.assert_allclose(
                self._component_array(ellipsoids.centers)[0],
                ellipsoid_translation,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                self._component_array(ellipsoids.half_sizes),
                ellipsoid_radii[None, :],
                atol=1e-5,
            )
            np.testing.assert_allclose(
                self._component_array(ellipsoids.quaternions),
                ellipsoid_rotation[None, :],
                atol=1e-5,
            )

            np.testing.assert_allclose(
                self._component_array(boxes.centers)[0], box_translation, atol=1e-5
            )
            np.testing.assert_allclose(
                self._component_array(boxes.half_sizes),
                box_half_extents[None, :],
                atol=1e-5,
            )
            np.testing.assert_allclose(
                self._component_array(boxes.quaternions),
                box_rotation[None, :],
                atol=1e-5,
            )

        def test_log_character_includes_ellipsoid_and_box(self) -> None:
            character, skel_state = (
                self._make_mixed_collision_character_and_skel_state()
            )

            mock_rec = MagicMock()
            log_character(
                mock_rec, "ch_mixed", character, skel_state, collision_geometry=True
            )
            paths = [c.args[0] for c in mock_rec.log.call_args_list]
            self.assertIn("ch_mixed/collision_proxies", paths)
            self.assertIn("ch_mixed/collision_proxies/ellipsoids", paths)
            self.assertIn("ch_mixed/collision_proxies/boxes", paths)

        def test_log_character_with_and_without_mesh(self) -> None:
            character, skel_state = self._make_character_and_skel_state()
            posed_mesh = self._posed_mesh(character)
            n_joints = character.skeleton.size

            mock_rec = MagicMock()
            log_character(mock_rec, "ch", character, skel_state, posed_mesh=posed_mesh)
            paths = [c.args[0] for c in mock_rec.log.call_args_list]
            self.assertIn("ch/mesh", paths)
            self.assertIn("ch/joints", paths)
            # The test character always has locators, so they must be logged.
            self.assertGreater(len(character.locators), 0)
            self.assertIn("ch/locators", paths)
            joint_transform_paths = [p for p in paths if p.startswith("ch/joints/")]
            self.assertEqual(len(joint_transform_paths), n_joints)

            mock_rec = MagicMock()
            log_character(mock_rec, "ch_nm", character, skel_state)
            paths_no_mesh = [c.args[0] for c in mock_rec.log.call_args_list]
            self.assertNotIn("ch_nm/mesh", paths_no_mesh)
            self.assertIn("ch_nm/joints", paths_no_mesh)

        def test_log_character_collision_geometry_is_opt_in(self) -> None:
            character, skel_state = self._make_collision_character_and_skel_state()

            mock_rec = MagicMock()
            log_character(mock_rec, "ch", character, skel_state)
            paths = [c.args[0] for c in mock_rec.log.call_args_list]
            self.assertNotIn("ch/collision_proxies", paths)

            mock_rec = MagicMock()
            log_character(
                mock_rec, "ch", character, skel_state, collision_geometry=True
            )
            paths = [c.args[0] for c in mock_rec.log.call_args_list]
            self.assertIn("ch/collision_proxies", paths)

        def _make_animation_inputs(
            self, n_frames: int
        ) -> tuple[geo.Character, np.ndarray, np.ndarray, list[geo.Mesh]]:
            """Build (character, joint_params, skel_states, posed_meshes) for n_frames."""
            character, _ = self._make_character_and_skel_state()
            model_params = np.zeros(
                (n_frames, character.parameter_transform.size), dtype=np.float32
            )
            joint_params = character.parameter_transform.apply(model_params)
            skel_states = geo.joint_parameters_to_skeleton_state(
                character, joint_params
            )
            posed_meshes = [
                character.pose_mesh(joint_params[i]) for i in range(n_frames)
            ]
            return character, joint_params, skel_states, posed_meshes

        def test_log_animation_sends_columns(self) -> None:
            character, _, skel_states, posed_meshes = self._make_animation_inputs(5)

            mock_rec = MagicMock()
            log_animation(
                mock_rec,
                "test/anim",
                character,
                skel_states,
                posed_meshes=posed_meshes,
                fps=30.0,
            )

            # First-frame mesh log establishes triangle_indices/albedo at frame 0.
            log_paths = [c.args[0] for c in mock_rec.log.call_args_list]
            self.assertIn("test/anim/mesh", log_paths)

            # send_columns is used for: vertex positions (4 remaining frames),
            # bones static + dynamic, per-joint transforms, locators static + dynamic.
            send_paths = [c.args[0] for c in mock_rec.send_columns.call_args_list]
            self.assertIn("test/anim/mesh", send_paths)
            self.assertIn("test/anim/joints", send_paths)
            # Per-joint transforms: one send_columns per joint.
            joint_subpaths = [
                p for p in send_paths if p.startswith("test/anim/joints/")
            ]
            self.assertEqual(len(joint_subpaths), character.skeleton.size)
            # The test character always has locators, so they must be animated.
            self.assertGreater(len(character.locators), 0)
            self.assertIn("test/anim/locators", send_paths)

        def test_log_animation_with_collision_geometry(self) -> None:
            character, skel_state = self._make_collision_character_and_skel_state()
            moved_skel_state = skel_state.copy()
            moved_skel_state[0, 0] = 5.0
            skel_states = np.stack([skel_state, moved_skel_state], axis=0)

            mock_rec = MagicMock()
            log_animation(
                mock_rec,
                "test/collision_anim",
                character,
                skel_states,
                fps=30.0,
                collision_geometry=True,
            )

            send_paths = [c.args[0] for c in mock_rec.send_columns.call_args_list]
            self.assertIn("test/collision_anim/collision_proxies", send_paths)
            collision_calls = [
                c
                for c in mock_rec.send_columns.call_args_list
                if c.args[0] == "test/collision_anim/collision_proxies"
            ]
            self.assertGreaterEqual(len(collision_calls), 1)
            static_columns = {
                column.component_descriptor().component: column.as_arrow_array().to_pylist()
                for column in collision_calls[0].kwargs["columns"]
            }
            self.assertEqual(
                static_columns["Capsules3D:fill_mode"],
                [[rr.components.FillMode.Solid.value]],
            )

        def test_log_animation_sends_ellipsoid_and_box_columns(self) -> None:
            character, skel_state = (
                self._make_mixed_collision_character_and_skel_state()
            )
            moved = skel_state.copy()
            moved[:, 0] += 5.0
            skel_states = np.stack([skel_state, moved], axis=0)

            mock_rec = MagicMock()
            log_animation(
                mock_rec,
                "test/mixed_anim",
                character,
                skel_states,
                fps=30.0,
                collision_geometry=True,
            )

            send_paths = [c.args[0] for c in mock_rec.send_columns.call_args_list]
            # Capsules send to the root proxy path; ellipsoids/boxes to sub-paths.
            self.assertIn("test/mixed_anim/collision_proxies", send_paths)
            self.assertIn("test/mixed_anim/collision_proxies/ellipsoids", send_paths)
            self.assertIn("test/mixed_anim/collision_proxies/boxes", send_paths)
            ellipsoid_calls = [
                c
                for c in mock_rec.send_columns.call_args_list
                if c.args[0] == "test/mixed_anim/collision_proxies/ellipsoids"
            ]
            box_calls = [
                c
                for c in mock_rec.send_columns.call_args_list
                if c.args[0] == "test/mixed_anim/collision_proxies/boxes"
            ]
            self.assertGreaterEqual(len(ellipsoid_calls), 1)
            self.assertGreaterEqual(len(box_calls), 1)
            ellipsoid_static_columns = {
                column.component_descriptor().component: column.as_arrow_array().to_pylist()
                for column in ellipsoid_calls[0].kwargs["columns"]
            }
            box_static_columns = {
                column.component_descriptor().component: column.as_arrow_array().to_pylist()
                for column in box_calls[0].kwargs["columns"]
            }
            self.assertEqual(
                ellipsoid_static_columns["Ellipsoids3D:fill_mode"],
                [[rr.components.FillMode.Solid.value]],
            )
            self.assertEqual(
                box_static_columns["Boxes3D:fill_mode"],
                [[rr.components.FillMode.Solid.value]],
            )

        def test_log_animation_without_collision_geometry_does_not_send(self) -> None:
            # Regression: a character with no collision capsules but with
            # collision_geometry=True should silently no-op (no send_columns call to
            # collision_proxies), not crash.
            character, _, skel_states, _ = self._make_animation_inputs(2)
            character = character.with_collision_geometry([])
            self.assertEqual(len(character.collision_geometry), 0)

            mock_rec = MagicMock()
            log_animation(
                mock_rec,
                "test/no_collision",
                character,
                skel_states,
                fps=30.0,
                collision_geometry=True,
            )
            send_paths = [c.args[0] for c in mock_rec.send_columns.call_args_list]
            self.assertNotIn("test/no_collision/collision_proxies", send_paths)

        def test_log_animation_without_mesh(self) -> None:
            character, _, skel_states, _ = self._make_animation_inputs(3)

            mock_rec = MagicMock()
            log_animation(mock_rec, "test/no_mesh", character, skel_states, fps=30.0)

            # No mesh means no log() for the mesh entity and no send_columns to it.
            log_paths = [c.args[0] for c in mock_rec.log.call_args_list]
            self.assertNotIn("test/no_mesh/mesh", log_paths)
            send_paths = [c.args[0] for c in mock_rec.send_columns.call_args_list]
            self.assertNotIn("test/no_mesh/mesh", send_paths)
            # Joints are still logged.
            self.assertIn("test/no_mesh/joints", send_paths)

        def test_log_animation_with_frame_offset(self) -> None:
            character, _, skel_states, _ = self._make_animation_inputs(4)
            offset = 100

            mock_rec = MagicMock()
            log_animation(
                mock_rec,
                "test/offset",
                character,
                skel_states,
                fps=30.0,
                frame_offset=offset,
            )

            # Time columns passed to send_columns must start at frame_offset.
            for call in mock_rec.send_columns.call_args_list:
                for col in call.kwargs["indexes"]:
                    if col.timeline_name == "frame_index":
                        first_idx = int(np.asarray(col.times)[0])
                        self.assertGreaterEqual(first_idx, offset)

        def test_log_animation_single_frame_with_mesh(self) -> None:
            # Regression test: 1-frame case used to trigger
            # "num_positions % 3 == 0" on the viewer side because rec.log and
            # send_columns both wrote vertex_positions at the same time point.
            character, _, skel_states, posed_meshes = self._make_animation_inputs(1)

            mock_rec = MagicMock()
            log_animation(
                mock_rec,
                "test/one",
                character,
                skel_states,
                posed_meshes=posed_meshes,
                fps=30.0,
            )

            # First-frame mesh comes from rec.log; no send_columns to the mesh
            # entity (would overlap at frame 0 and break the viewer).
            log_paths = [c.args[0] for c in mock_rec.log.call_args_list]
            self.assertIn("test/one/mesh", log_paths)
            mesh_send_calls = [
                c
                for c in mock_rec.send_columns.call_args_list
                if c.args[0] == "test/one/mesh"
            ]
            self.assertEqual(mesh_send_calls, [])

        def test_log_animation_single_frame_without_mesh(self) -> None:
            character, _, skel_states, _ = self._make_animation_inputs(1)

            mock_rec = MagicMock()
            log_animation(
                mock_rec, "test/one_no_mesh", character, skel_states, fps=30.0
            )

            send_paths = [c.args[0] for c in mock_rec.send_columns.call_args_list]
            self.assertIn("test/one_no_mesh/joints", send_paths)

        def test_log_animation_chunked(self) -> None:
            """Simulate chunked sending as the viewer does for long animations."""
            character, joint_params, skel_states, _ = self._make_animation_inputs(10)
            chunk_size = 4

            mock_rec = MagicMock()
            for start in range(0, 10, chunk_size):
                end = min(start + chunk_size, 10)
                chunk_meshes = [
                    character.pose_mesh(joint_params[i]) for i in range(start, end)
                ]
                log_animation(
                    mock_rec,
                    "test/chunked",
                    character,
                    skel_states[start:end],
                    posed_meshes=chunk_meshes,
                    fps=30.0,
                    frame_offset=start,
                )

            # 3 chunks (4+4+2 frames) → each re-logs the first-frame mesh.
            mesh_log_calls = [
                c
                for c in mock_rec.log.call_args_list
                if c.args[0] == "test/chunked/mesh"
            ]
            self.assertEqual(len(mesh_log_calls), 3)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Skip when rerun-sdk is unavailable or ThreadSanitizer is active.

    Note: pytest does not honor this protocol — the rerun-availability skip is
    enforced by gating the TestRerunVis class definition above. This hook
    remains for the TSan skip path under unittest (BUCK).
    """
    if not _HAS_RERUN or _is_tsan_active():
        return unittest.TestSuite()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRerunVis))
    return suite
