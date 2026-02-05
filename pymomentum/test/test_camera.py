# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import pymomentum.camera as pym_camera


class TestCamera(unittest.TestCase):
    def test_pinhole_intrinsics_model_creation(self) -> None:
        """Test that PinholeIntrinsicsModel can be created."""
        image_width, image_height = 640, 480
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=image_width,
            image_height=image_height,
        )
        self.assertEqual(intrinsics.image_width, image_width)
        self.assertEqual(intrinsics.image_height, image_height)
        # Check default focal length is computed
        self.assertGreater(intrinsics.fx, 0)
        self.assertGreater(intrinsics.fy, 0)
        # Check default principal point is at center
        self.assertAlmostEqual(intrinsics.cx, image_width / 2.0)
        self.assertAlmostEqual(intrinsics.cy, image_height / 2.0)

    def test_pinhole_intrinsics_model_with_custom_params(self) -> None:
        """Test PinholeIntrinsicsModel with custom focal length and principal point."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=800,
            image_height=600,
            fx=500.0,
            fy=500.0,
            cx=400.0,
            cy=300.0,
        )
        self.assertEqual(intrinsics.image_width, 800)
        self.assertEqual(intrinsics.image_height, 600)
        self.assertAlmostEqual(intrinsics.fx, 500.0)
        self.assertAlmostEqual(intrinsics.fy, 500.0)
        self.assertAlmostEqual(intrinsics.cx, 400.0)
        self.assertAlmostEqual(intrinsics.cy, 300.0)

    def test_opencv_distortion_parameters(self) -> None:
        """Test OpenCVDistortionParameters creation and modification."""
        params = pym_camera.OpenCVDistortionParameters()
        # Check default values are zero
        self.assertAlmostEqual(params.k1, 0.0)
        self.assertAlmostEqual(params.k2, 0.0)
        self.assertAlmostEqual(params.p1, 0.0)
        self.assertAlmostEqual(params.p2, 0.0)

        # Modify values
        params.k1 = 0.1
        params.k2 = 0.2
        self.assertAlmostEqual(params.k1, 0.1)
        self.assertAlmostEqual(params.k2, 0.2)

    def test_opencv_intrinsics_model(self) -> None:
        """Test OpenCVIntrinsicsModel creation with distortion."""
        distortion = pym_camera.OpenCVDistortionParameters()
        distortion.k1 = 0.1

        intrinsics = pym_camera.OpenCVIntrinsicsModel(
            image_width=1920,
            image_height=1080,
            fx=1000.0,
            fy=1000.0,
            cx=960.0,
            cy=540.0,
            distortion_params=distortion,
        )
        self.assertEqual(intrinsics.image_width, 1920)
        self.assertEqual(intrinsics.image_height, 1080)
        self.assertAlmostEqual(intrinsics.fx, 1000.0)
        self.assertAlmostEqual(intrinsics.fy, 1000.0)

    def test_camera_creation(self) -> None:
        """Test Camera creation with intrinsics."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
        )
        camera = pym_camera.Camera(intrinsics)

        self.assertEqual(camera.image_width, 640)
        self.assertEqual(camera.image_height, 480)

        # Check that the default transform is identity
        center = camera.center_of_projection
        np.testing.assert_allclose(center, [0, 0, 0], atol=1e-6)

    def test_camera_with_transform(self) -> None:
        """Test Camera creation with a custom transform."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
        )

        # Create a translation matrix
        eye_from_world = np.eye(4, dtype=np.float32)
        eye_from_world[0:3, 3] = [1, 2, 3]  # Translate camera

        camera = pym_camera.Camera(intrinsics, eye_from_world)

        # The center of projection should be the inverse translation
        center = camera.center_of_projection
        np.testing.assert_allclose(center, [-1, -2, -3], atol=1e-6)

    def test_camera_look_at(self) -> None:
        """Test Camera.look_at method."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
        )
        camera = pym_camera.Camera(intrinsics)

        # Position camera at (0, 0, -5) looking at origin
        new_camera = camera.look_at(
            position=np.asarray([0, 0, -5], dtype=np.float32),
            target=np.asarray([0, 0, 0], dtype=np.float32),
        )

        # Check camera position
        center = new_camera.center_of_projection
        np.testing.assert_allclose(center, [0, 0, -5], atol=1e-6)

    def test_camera_project_unproject(self) -> None:
        """Test Camera.project and Camera.unproject are inverses."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )
        camera = pym_camera.Camera(intrinsics)

        # Create some 3D points in front of the camera
        world_points = np.asarray(
            [[0, 0, 5], [1, 0, 5], [0, 1, 5], [-1, -1, 10]], dtype=np.float32
        )

        # Project to image space
        image_points = camera.project(world_points)

        # Unproject back to world space
        recovered_points = camera.unproject(image_points)

        np.testing.assert_allclose(recovered_points, world_points, atol=1e-5)

    def test_intrinsics_upsample_downsample(self) -> None:
        """Test that upsample and downsample are inverses."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
        )

        upsampled = intrinsics.upsample(2.0)
        self.assertEqual(upsampled.image_width, 1280)
        self.assertEqual(upsampled.image_height, 960)
        self.assertAlmostEqual(upsampled.fx, 1000.0)

        downsampled = upsampled.downsample(2.0)
        self.assertEqual(downsampled.image_width, 640)
        self.assertEqual(downsampled.image_height, 480)
        self.assertAlmostEqual(downsampled.fx, 500.0)

    def test_intrinsics_crop(self) -> None:
        """Test intrinsics crop operation."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )

        # Crop to top-left 320x240 region
        cropped = intrinsics.crop(top=0, left=0, width=320, height=240)
        self.assertEqual(cropped.image_width, 320)
        self.assertEqual(cropped.image_height, 240)
        # Focal length should be unchanged
        self.assertAlmostEqual(cropped.fx, 500.0)
        # Principal point should be shifted
        self.assertAlmostEqual(cropped.cx, 320.0)
        self.assertAlmostEqual(cropped.cy, 240.0)

    def test_camera_crop(self) -> None:
        """Test camera crop operation."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
        )
        camera = pym_camera.Camera(intrinsics)

        cropped_camera = camera.crop(top=0, left=0, width=320, height=240)
        self.assertEqual(cropped_camera.image_width, 320)
        self.assertEqual(cropped_camera.image_height, 240)

    def test_camera_upsample(self) -> None:
        """Test camera upsample operation."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=320,
            image_height=240,
        )
        camera = pym_camera.Camera(intrinsics)

        upsampled = camera.upsample(2.0)
        self.assertEqual(upsampled.image_width, 640)
        self.assertEqual(upsampled.image_height, 480)


if __name__ == "__main__":
    unittest.main()
