# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import pymomentum.camera as pym_camera


class TestCamera(unittest.TestCase):
    def test_project_unproject_roundtrip_identity_camera(self) -> None:
        """Project world points through an identity-pose camera and unproject back."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )
        camera = pym_camera.Camera(intrinsics)

        world_points = np.asarray(
            [[0, 0, 5], [1, 0, 5], [0, 1, 5], [-1, -1, 10]], dtype=np.float32
        )

        image_points = camera.project(world_points)
        recovered = camera.unproject(image_points)
        np.testing.assert_allclose(recovered, world_points, atol=1e-5)

    def test_project_unproject_roundtrip_transformed_camera(self) -> None:
        """Project/unproject roundtrip with a non-identity camera pose."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )

        # Place camera at (1, 2, -5) looking roughly toward origin via a
        # pure-translation transform (eye_from_world translates world points).
        eye_from_world = np.eye(4, dtype=np.float32)
        eye_from_world[0:3, 3] = [-1, -2, 5]

        camera = pym_camera.Camera(intrinsics, eye_from_world)

        world_points = np.asarray([[0, 0, 0], [1, 1, 0], [-1, 2, -1]], dtype=np.float32)

        image_points = camera.project(world_points)
        recovered = camera.unproject(image_points)
        np.testing.assert_allclose(recovered, world_points, atol=1e-5)

    def test_center_of_projection_with_transform(self) -> None:
        """center_of_projection should equal the inverse of the translation."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
        )

        eye_from_world = np.eye(4, dtype=np.float32)
        eye_from_world[0:3, 3] = [1, 2, 3]

        camera = pym_camera.Camera(intrinsics, eye_from_world)
        np.testing.assert_allclose(camera.center_of_projection, [-1, -2, -3], atol=1e-6)

    def test_look_at(self) -> None:
        """look_at should place the camera at the requested position and
        orient it toward the target."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
        )
        camera = pym_camera.Camera(intrinsics)

        position = np.asarray([0, 0, -5], dtype=np.float32)
        target = np.asarray([0, 0, 0], dtype=np.float32)
        new_camera = camera.look_at(position=position, target=target)

        # Camera position should match
        np.testing.assert_allclose(
            new_camera.center_of_projection, [0, 0, -5], atol=1e-6
        )

        # Principal axis should point from camera toward target (positive z)
        axis = new_camera.world_space_principle_axis
        expected_dir = target - position
        expected_dir = expected_dir / np.linalg.norm(expected_dir)
        np.testing.assert_allclose(axis, expected_dir, atol=1e-6)

    def test_crop_preserves_projection(self) -> None:
        """Cropping a camera should shift pixel coordinates by the crop offset
        but produce the same depth values."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )
        camera = pym_camera.Camera(intrinsics)

        world_points = np.asarray([[0, 0, 5], [1, -1, 8]], dtype=np.float32)

        top, left = 50, 100
        cropped_camera = camera.crop(top=top, left=left, width=320, height=240)

        orig_proj = camera.project(world_points)
        crop_proj = cropped_camera.project(world_points)

        # Pixel coordinates should be offset by (left, top)
        np.testing.assert_allclose(crop_proj[:, 0], orig_proj[:, 0] - left, atol=1e-5)
        np.testing.assert_allclose(crop_proj[:, 1], orig_proj[:, 1] - top, atol=1e-5)
        # Depth should be unchanged
        np.testing.assert_allclose(crop_proj[:, 2], orig_proj[:, 2], atol=1e-5)

    def test_upsample_project_unproject_roundtrip(self) -> None:
        """Upsampled camera should still produce valid project/unproject
        roundtrips, and upsample+downsample should recover the original
        projection."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=320,
            image_height=240,
            fx=250.0,
            fy=250.0,
            cx=160.0,
            cy=120.0,
        )
        camera = pym_camera.Camera(intrinsics)

        world_points = np.asarray([[0, 0, 5], [1, -1, 8]], dtype=np.float32)

        factor = 2.0
        upsampled = camera.upsample(factor)

        # Image dimensions should scale
        self.assertEqual(upsampled.image_width, 640)
        self.assertEqual(upsampled.image_height, 480)

        # Project/unproject roundtrip through the upsampled camera
        up_proj = upsampled.project(world_points)
        recovered = upsampled.unproject(up_proj)
        np.testing.assert_allclose(recovered, world_points, atol=1e-5)

    def test_intrinsics_upsample_downsample_roundtrip(self) -> None:
        """Upsampling then downsampling intrinsics should recover the original
        projection."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )

        points = np.asarray([[0.5, -0.3, 4.0]], dtype=np.float32)
        orig_proj = intrinsics.project(points)

        roundtripped = intrinsics.upsample(2.0).downsample(2.0)
        rt_proj = roundtripped.project(points)

        np.testing.assert_allclose(rt_proj, orig_proj, atol=1e-5)

    def test_frame_keeps_points_visible(self) -> None:
        """frame() should reposition the camera so that all points project
        within the image bounds."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )
        camera = pym_camera.Camera(intrinsics)
        camera = camera.look_at(
            position=np.asarray([0, 0, -10], dtype=np.float32),
            target=np.asarray([0, 0, 0], dtype=np.float32),
        )

        # Points spread widely; they may not all be visible from the
        # initial camera position.
        points = np.asarray(
            [[-3, -3, 0], [3, 3, 0], [-3, 3, 0], [3, -3, 0]],
            dtype=np.float32,
        )

        framed = camera.frame(points)

        projected = framed.project(points)
        # All projected x-coords should be within [0, image_width]
        self.assertTrue(
            np.all(projected[:, 0] >= 0),
            f"Some x-coords are negative: {projected[:, 0]}",
        )
        self.assertTrue(
            np.all(projected[:, 0] <= framed.image_width),
            f"Some x-coords exceed image width: {projected[:, 0]}",
        )
        # All projected y-coords should be within [0, image_height]
        self.assertTrue(
            np.all(projected[:, 1] >= 0),
            f"Some y-coords are negative: {projected[:, 1]}",
        )
        self.assertTrue(
            np.all(projected[:, 1] <= framed.image_height),
            f"Some y-coords exceed image height: {projected[:, 1]}",
        )
        # All points should be in front of the camera (positive depth)
        self.assertTrue(
            np.all(projected[:, 2] > 0),
            f"Some points are behind the camera: {projected[:, 2]}",
        )

    def test_pinhole_num_intrinsic_parameters(self) -> None:
        """Test that pinhole model returns correct number of parameters."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )
        self.assertEqual(intrinsics.num_intrinsic_parameters, 4)

    def test_opencv_num_intrinsic_parameters(self) -> None:
        """Test that OpenCV model returns correct number of parameters."""
        intrinsics = pym_camera.OpenCVIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )
        self.assertEqual(intrinsics.num_intrinsic_parameters, 14)

    def test_pinhole_get_intrinsic_parameters(self) -> None:
        """Test getting intrinsic parameters from pinhole model."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=550.0,
            cx=320.0,
            cy=240.0,
        )
        params = intrinsics.get_intrinsic_parameters()
        self.assertEqual(len(params), 4)
        np.testing.assert_allclose(params, [500.0, 550.0, 320.0, 240.0], atol=1e-5)

    def test_opencv_get_intrinsic_parameters(self) -> None:
        """Test getting intrinsic parameters from OpenCV model."""
        distortion = pym_camera.OpenCVDistortionParameters()
        distortion.k1 = 0.1
        distortion.k2 = -0.05
        distortion.p1 = 0.01
        distortion.p2 = 0.02

        intrinsics = pym_camera.OpenCVIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion_params=distortion,
        )
        params = intrinsics.get_intrinsic_parameters()
        self.assertEqual(len(params), 14)
        # Check fx, fy, cx, cy
        np.testing.assert_allclose(params[:4], [500.0, 500.0, 320.0, 240.0], atol=1e-5)
        # Check k1, k2
        np.testing.assert_allclose(params[4:6], [0.1, -0.05], atol=1e-5)
        # Check p1, p2
        np.testing.assert_allclose(params[10:12], [0.01, 0.02], atol=1e-5)

    def test_pinhole_clone_and_set_parameters(self) -> None:
        """Test cloning and modifying pinhole intrinsics."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )

        # Clone and modify
        cloned = intrinsics.clone()
        new_params = np.asarray([600.0, 650.0, 350.0, 260.0], dtype=np.float32)
        cloned.set_intrinsic_parameters(new_params)

        # Verify clone was modified
        cloned_params = cloned.get_intrinsic_parameters()
        np.testing.assert_allclose(cloned_params, new_params, atol=1e-5)

        # Verify original is unchanged
        original_params = intrinsics.get_intrinsic_parameters()
        np.testing.assert_allclose(
            original_params, [500.0, 500.0, 320.0, 240.0], atol=1e-5
        )

    def test_opencv_clone_and_set_parameters(self) -> None:
        """Test cloning and modifying OpenCV intrinsics."""
        distortion = pym_camera.OpenCVDistortionParameters()
        distortion.k1 = 0.1

        intrinsics = pym_camera.OpenCVIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion_params=distortion,
        )

        # Clone and modify
        cloned = intrinsics.clone()
        new_params = np.zeros(14, dtype=np.float32)
        new_params[:4] = [600.0, 650.0, 350.0, 260.0]
        new_params[4] = 0.2  # k1
        cloned.set_intrinsic_parameters(new_params)

        # Verify clone was modified
        cloned_params = cloned.get_intrinsic_parameters()
        self.assertAlmostEqual(cloned_params[0], 600.0, places=5)
        self.assertAlmostEqual(cloned_params[4], 0.2, places=5)

        # Verify original is unchanged
        original_params = intrinsics.get_intrinsic_parameters()
        self.assertAlmostEqual(original_params[0], 500.0, places=5)
        self.assertAlmostEqual(original_params[4], 0.1, places=5)

    def test_pinhole_project_intrinsics_jacobian(self) -> None:
        """Test project_intrinsics_jacobian for pinhole model."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )

        point = np.asarray([1.0, 0.5, 2.0], dtype=np.float32)
        projected, jacobian, valid = intrinsics.project_intrinsics_jacobian(point)

        self.assertTrue(valid)
        self.assertEqual(jacobian.shape, (3, 4))

        # Verify the projected point matches project()
        points_arr = np.asarray([[1.0, 0.5, 2.0]], dtype=np.float32)
        projected_arr = intrinsics.project(points_arr)
        np.testing.assert_allclose(projected[:2], projected_arr[0, :2], atol=1e-5)

        # Verify analytical Jacobian values for pinhole model:
        # u = fx * (x/z) + cx, v = fy * (y/z) + cy
        # du/dfx = x/z = 1.0/2.0 = 0.5
        # du/dfy = 0
        # du/dcx = 1
        # du/dcy = 0
        # dv/dfx = 0
        # dv/dfy = y/z = 0.5/2.0 = 0.25
        # dv/dcx = 0
        # dv/dcy = 1
        expected_jacobian = np.array(
            [[0.5, 0.0, 1.0, 0.0], [0.0, 0.25, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(jacobian, expected_jacobian, atol=1e-5)

    def test_project_intrinsics_jacobian_behind_camera(self) -> None:
        """Test that project_intrinsics_jacobian returns invalid for points behind camera."""
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
        )

        behind_point = np.asarray([1.0, 0.5, -2.0], dtype=np.float32)
        projected, jacobian, valid = intrinsics.project_intrinsics_jacobian(
            behind_point
        )

        self.assertFalse(valid)
        self.assertEqual(jacobian.shape, (3, 4))

    def test_fisheye_project_unproject_roundtrip(self) -> None:
        """Project 3D points through a fisheye camera and unproject back."""
        distortion = pym_camera.OpenCVFisheyeDistortionParameters()
        distortion.k1 = 0.02
        distortion.k2 = -0.01
        distortion.k3 = 0.005
        distortion.k4 = -0.002

        intrinsics = pym_camera.OpenCVFisheyeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion_params=distortion,
        )
        camera = pym_camera.Camera(intrinsics)

        world_points = np.asarray(
            [[0, 0, 5], [1, 0, 5], [0, 1, 5], [-1, -1, 10]], dtype=np.float32
        )

        image_points = camera.project(world_points)
        recovered = camera.unproject(image_points)
        np.testing.assert_allclose(recovered, world_points, atol=1e-4)

    def test_fisheye_matches_numpy_reference(self) -> None:
        """Verify fisheye projection matches a pure-numpy equidistant model."""
        k1, k2, k3, k4 = 0.02, -0.01, 0.005, -0.002
        fx_val, fy_val, cx_val, cy_val = 500.0, 500.0, 320.0, 240.0

        distortion = pym_camera.OpenCVFisheyeDistortionParameters()
        distortion.k1 = k1
        distortion.k2 = k2
        distortion.k3 = k3
        distortion.k4 = k4

        intrinsics = pym_camera.OpenCVFisheyeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=fx_val,
            fy=fy_val,
            cx=cx_val,
            cy=cy_val,
            distortion_params=distortion,
        )

        points_3d = np.asarray(
            [[0.5, 0.3, 4.0], [-1.0, 0.5, 6.0], [0.0, 0.0, 3.0], [2.0, -1.0, 8.0]],
            dtype=np.float32,
        )

        # Numpy reference implementation of equidistant fisheye projection
        a = points_3d[:, 0] / points_3d[:, 2]
        b = points_3d[:, 1] / points_3d[:, 2]
        r = np.sqrt(a * a + b * b)
        theta = np.arctan(r)
        theta2 = theta * theta
        theta_d = theta * (
            1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
        )
        scale = np.where(r > 1e-8, theta_d / r, 1.0)
        u_ref = fx_val * scale * a + cx_val
        v_ref = fy_val * scale * b + cy_val

        projected = intrinsics.project(points_3d)

        np.testing.assert_allclose(projected[:, 0], u_ref, atol=1e-4)
        np.testing.assert_allclose(projected[:, 1], v_ref, atol=1e-4)

    def test_fisheye_zero_distortion_matches_pinhole_small_angles(self) -> None:
        """With zero distortion and small angles, fisheye should match pinhole.

        At small angles atan(r) ≈ r, so the equidistant model converges to
        the pinhole model.  At large angles the models diverge, so we only
        test with points near the optical axis.
        """
        fx_val, fy_val, cx_val, cy_val = 500.0, 500.0, 320.0, 240.0

        fisheye = pym_camera.OpenCVFisheyeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=fx_val,
            fy=fy_val,
            cx=cx_val,
            cy=cy_val,
        )
        pinhole = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=fx_val,
            fy=fy_val,
            cx=cx_val,
            cy=cy_val,
        )

        # Small-angle points (x/z and y/z are small)
        points = np.asarray(
            [[0.1, 0.05, 5.0], [-0.05, 0.1, 8.0], [0.0, 0.0, 3.0]],
            dtype=np.float32,
        )

        fisheye_proj = fisheye.project(points)
        pinhole_proj = pinhole.project(points)
        np.testing.assert_allclose(fisheye_proj, pinhole_proj, atol=5e-3)

    def test_fisheye_crop_preserves_projection(self) -> None:
        """Cropping a fisheye camera should shift pixel coordinates by the crop
        offset but produce the same depth values."""
        distortion = pym_camera.OpenCVFisheyeDistortionParameters()
        distortion.k1 = 0.01
        distortion.k2 = -0.005

        intrinsics = pym_camera.OpenCVFisheyeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion_params=distortion,
        )
        camera = pym_camera.Camera(intrinsics)

        world_points = np.asarray([[0, 0, 5], [1, -1, 8]], dtype=np.float32)

        top, left = 50, 100
        cropped_camera = camera.crop(top=top, left=left, width=320, height=240)

        orig_proj = camera.project(world_points)
        crop_proj = cropped_camera.project(world_points)

        # Pixel coordinates should be offset by (left, top)
        np.testing.assert_allclose(crop_proj[:, 0], orig_proj[:, 0] - left, atol=1e-5)
        np.testing.assert_allclose(crop_proj[:, 1], orig_proj[:, 1] - top, atol=1e-5)
        # Depth should be unchanged
        np.testing.assert_allclose(crop_proj[:, 2], orig_proj[:, 2], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
