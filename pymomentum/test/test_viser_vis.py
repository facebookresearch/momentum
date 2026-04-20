# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import pymomentum.geometry as geo
import pymomentum.geometry_test_utils as test_utils

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
        _normalize_param_limit,
        _xyzw_to_wxyz,
        add_character,
        add_character_param_sliders,
        add_joints,
        add_log_panel,
        add_mesh,
        add_wireframe_toggle,
        CharacterHandles,
        remove_character,
        update_character,
        update_joints,
        update_mesh,
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

        def _posed_mesh(self, character: geo.Character) -> geo.Mesh:
            joint_params = np.zeros(character.skeleton.size * 7, dtype=np.float32)
            return character.pose_mesh(joint_params)

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

            for h in handles.values():
                h.remove()
            bones_handle.remove()

        def test_add_and_update_mesh(self) -> None:
            character, _ = self._make_character_and_skel_state()
            posed_mesh = self._posed_mesh(character)
            self.assertGreater(len(posed_mesh.vertices), 0)
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
            import math

            lo, hi = _normalize_param_limit("hip_rx", -1e35, 1e35)
            self.assertAlmostEqual(lo, -math.pi)
            self.assertAlmostEqual(hi, math.pi)

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

        def test_add_wireframe_toggle(self) -> None:
            character, skel_state = self._make_character_and_skel_state()
            posed_mesh = self._posed_mesh(character)
            handles = add_character(
                self.server, "test_wf", character, skel_state, posed_mesh=posed_mesh
            )
            self.assertIsNotNone(handles.mesh_handle)
            self.assertIsNotNone(handles.wireframe_handle)

            # Default initial_value=False: solid visible, wireframe hidden.
            cb = add_wireframe_toggle(self.server, handles)
            self.assertTrue(handles.mesh_handle.visible)
            self.assertFalse(handles.wireframe_handle.visible)

            # Flip the checkbox; visibility should swap.
            cb.value = True
            self.assertFalse(handles.mesh_handle.visible)
            self.assertTrue(handles.wireframe_handle.visible)

            cb.remove()
            remove_character(handles)

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
