# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import tempfile
import unittest

import numpy as np
import pymomentum.geometry as pym_geometry
from pymomentum.marker_tracking import (
    calibrate_markers,
    CalibrationConfig,
    ModelOptions,
    process_markers,
    refine_motion,
    RefineConfig,
    save_motion,
    TrackingConfig,
)


class TestMarkerTracker(unittest.TestCase):
    def test_configs(self) -> None:
        # test values different than default

        # CalibrationConfig
        calib_config = CalibrationConfig(
            min_vis_percent=0.1,
            loss_alpha=1.0,
            max_iter=10,
            debug=True,
            calib_frames=20,
            major_iter=2,
            global_scale_only=True,
            locators_only=True,
        )
        self.assertTrue(
            math.isclose(calib_config.min_vis_percent, 0.1, abs_tol=1e-6)
            and math.isclose(calib_config.loss_alpha, 1.0, abs_tol=1e-6)
            and calib_config.max_iter == 10
            and calib_config.debug
            and calib_config.calib_frames == 20
            and calib_config.major_iter == 2
            and calib_config.global_scale_only
            and calib_config.locators_only
        )

        # TrackingConfig
        tracking_config = TrackingConfig(
            min_vis_percent=0.2,
            loss_alpha=1.5,
            max_iter=20,
            debug=True,
            smoothing=0.1,
            collision_error_weight=0.2,
        )
        self.assertTrue(
            math.isclose(tracking_config.min_vis_percent, 0.2, abs_tol=1e-6)
            and math.isclose(tracking_config.loss_alpha, 1.5, abs_tol=1e-6)
            and tracking_config.max_iter == 20
            and tracking_config.debug
            and math.isclose(tracking_config.smoothing, 0.1, abs_tol=1e-6)
            and math.isclose(tracking_config.collision_error_weight, 0.2, abs_tol=1e-6)
        )

        # RefineConfig
        refine_config = RefineConfig(
            min_vis_percent=0.2,
            loss_alpha=1.5,
            max_iter=20,
            debug=True,
            smoothing=0.1,
            collision_error_weight=0.2,
            regularizer=0.1,
            calib_id=True,
            calib_locators=True,
        )
        self.assertTrue(
            math.isclose(refine_config.min_vis_percent, 0.2, abs_tol=1e-6)
            and math.isclose(refine_config.loss_alpha, 1.5, abs_tol=1e-6)
            and refine_config.max_iter == 20
            and refine_config.debug
            and math.isclose(refine_config.smoothing, 0.1, abs_tol=1e-6)
            and math.isclose(refine_config.collision_error_weight, 0.2, abs_tol=1e-6)
            and math.isclose(refine_config.regularizer, 0.1, abs_tol=1e-6)
            and refine_config.calib_id
            and refine_config.calib_locators
        )

        model_options = ModelOptions("blueman.glb", "blueman.model", "blueman.locators")

        self.assertTrue(
            model_options.model == "blueman.glb"
            and model_options.parameters == "blueman.model"
            and model_options.locators == "blueman.locators"
        )

    def test_refine_config(self) -> None:
        refine_config_default = RefineConfig()
        self.assertTrue(refine_config_default.smoothing_weights.size == 0)

        refine_config = RefineConfig(smoothing_weights=np.asarray([1.0, 0, 1.0]))
        self.assertEqual(refine_config.smoothing_weights.shape, (3,))
        self.assertTrue(
            np.allclose(refine_config.smoothing_weights, [1.0, 0, 1.0], atol=1e-6)
        )

    def test_tracking_config(self) -> None:
        tracking_config = TrackingConfig(smoothing_weights=np.asarray([1.0, 0, 1.0]))
        self.assertEqual(tracking_config.smoothing_weights.shape, (3,))
        self.assertTrue(
            np.allclose(tracking_config.smoothing_weights, [1.0, 0, 1.0], atol=1e-6)
        )

    def test_motion_matrix(self) -> None:
        character = pym_geometry.create_test_character()
        tracking_config = TrackingConfig()
        calibration_config = CalibrationConfig()
        refine_config = RefineConfig()
        identity = np.zeros(0, dtype=np.float32)

        marker_data = []
        num_frames = 30
        for frame_i in range(num_frames):
            markers = []
            for loc in character.locators:
                markers.append(
                    pym_geometry.Marker(
                        loc.name, np.array([frame_i, 0.0, 0.0] + loc.offset), False
                    )
                )
            marker_data.append(markers)

        motion = process_markers(
            character, identity, marker_data, tracking_config, calibration_config
        )
        # check that process_markers returns correct motion matrix
        self.assertEqual(motion.shape[0], num_frames)
        self.assertEqual(motion.shape[1], character.parameter_transform.size)

        ref_motion = refine_motion(
            character, identity, motion, marker_data, refine_config
        )
        # check that refine_motion returns correct motion matrix
        self.assertEqual(ref_motion.shape[0], num_frames)
        self.assertEqual(ref_motion.shape[1], character.parameter_transform.size)

        with tempfile.NamedTemporaryFile() as temp_file:
            save_motion(
                temp_file.name,
                character,
                identity,
                ref_motion,
                marker_data,
                30,
                True,
            )
            # check that save_motion returns correct motion matrix
            self.assertEqual(ref_motion.shape[0], num_frames)
            self.assertEqual(ref_motion.shape[1], character.parameter_transform.size)

    def test_calibrate_markers(self) -> None:
        """Test that calibrate_markers returns valid identity parameters."""
        character = pym_geometry.create_test_character()
        calibration_config = CalibrationConfig(calib_frames=10)
        identity = np.zeros(0, dtype=np.float32)

        # Create marker data for calibration
        marker_data = []
        num_frames = 30
        for frame_i in range(num_frames):
            markers = []
            for loc in character.locators:
                markers.append(
                    pym_geometry.Marker(
                        loc.name, np.array([frame_i, 0.0, 0.0] + loc.offset), False
                    )
                )
            marker_data.append(markers)

        # Run calibration only (no tracking)
        calibrated_identity = calibrate_markers(
            character, identity, marker_data, calibration_config
        )

        # Check that calibrate_markers returns the correct identity vector size
        self.assertEqual(
            calibrated_identity.shape[0], character.parameter_transform.size
        )

        # Verify we can use the calibrated identity with process_markers
        tracking_config = TrackingConfig()
        motion = process_markers(
            character,
            calibrated_identity,
            marker_data,
            tracking_config,
            calibration_config,
            calibrate=False,
        )
        self.assertEqual(motion.shape[0], num_frames)
        self.assertEqual(motion.shape[1], character.parameter_transform.size)


if __name__ == "__main__":
    unittest.main()
