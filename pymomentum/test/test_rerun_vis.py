# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pymomentum.geometry as geo
import pymomentum.geometry_test_utils as test_utils
from pymomentum.rerun_vis import log_character, log_joints, log_locators, log_mesh

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
if _HAS_RERUN:

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

        def _posed_mesh(self, character: geo.Character) -> geo.Mesh:
            joint_params = np.zeros(character.skeleton.size * 7, dtype=np.float32)
            return character.pose_mesh(joint_params)

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

        def test_log_character_with_and_without_mesh(self) -> None:
            character, skel_state = self._make_character_and_skel_state()
            posed_mesh = self._posed_mesh(character)
            n_joints = character.skeleton.size

            mock_rec = MagicMock()
            log_character(mock_rec, "ch", character, skel_state, posed_mesh=posed_mesh)
            paths = [c.args[0] for c in mock_rec.log.call_args_list]
            self.assertIn("ch/mesh", paths)
            self.assertIn("ch/joints", paths)
            if character.locators:
                self.assertIn("ch/locators", paths)
            joint_transform_paths = [p for p in paths if p.startswith("ch/joints/")]
            self.assertEqual(len(joint_transform_paths), n_joints)

            mock_rec = MagicMock()
            log_character(mock_rec, "ch_nm", character, skel_state)
            paths_no_mesh = [c.args[0] for c in mock_rec.log.call_args_list]
            self.assertNotIn("ch_nm/mesh", paths_no_mesh)
            self.assertIn("ch_nm/joints", paths_no_mesh)


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
