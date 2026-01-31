# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import numpy.typing as npt
import pymomentum.geometry as geometry


def _brute_force_closest_points(
    src_pts: npt.NDArray,
    tgt_pts: npt.NDArray,
    tgt_normals: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    n_batch = src_pts.shape[0]
    n_src_pts = src_pts.shape[1]

    closest_pts = np.zeros(src_pts.shape, dtype=np.float32)
    closest_normals = np.zeros(src_pts.shape, dtype=np.float32)
    closest_index = np.zeros((n_batch, n_src_pts), dtype=np.int32)

    for i_batch in range(n_batch):
        for j_pt in range(n_src_pts):
            gt_diff = src_pts[i_batch, j_pt, :] - tgt_pts[i_batch, ...]
            gt_dist = np.linalg.norm(gt_diff, axis=-1)
            gt_closest_pt_idx = np.argmin(gt_dist)
            closest_index[i_batch, j_pt] = gt_closest_pt_idx
            closest_pts[i_batch, j_pt, :] = tgt_pts[i_batch, gt_closest_pt_idx, :]
            if tgt_normals is not None:
                closest_normals[i_batch, j_pt, :] = tgt_normals[
                    i_batch, gt_closest_pt_idx, :
                ]

    return closest_pts, closest_normals, closest_index, closest_index >= 0


class TestClosestPoints(unittest.TestCase):
    def test_closest_points(self) -> None:
        np.random.seed(0)  # ensure repeatability

        n_src_pts = 3
        n_tgt_pts = 20
        n_batch = 2
        dim = 3
        src_pts = np.random.rand(n_batch, n_src_pts, dim).astype(np.float32)
        tgt_pts = np.random.rand(n_batch, n_tgt_pts, dim).astype(np.float32)

        closest_pts, closest_idx, closest_valid = geometry.find_closest_points(
            src_pts, tgt_pts
        )
        closest_dist = np.linalg.norm(closest_pts - src_pts, axis=-1)
        self.assertTrue(np.all(closest_valid))

        gt_closest_pts, _, gt_closest_idx, _ = _brute_force_closest_points(
            src_pts, tgt_pts
        )
        self.assertTrue(np.allclose(gt_closest_pts, closest_pts))
        self.assertTrue(np.allclose(gt_closest_idx, closest_idx))

        # Verify that if we pass in a small enough max_dist we don't return any points:
        min_all_dist = np.min(closest_dist)
        closest_pts_2, closest_idx_2, closest_valid_2 = geometry.find_closest_points(
            src_pts, tgt_pts, float(0.5 * min_all_dist)
        )
        self.assertFalse(np.any(closest_valid_2))
        self.assertTrue(
            np.allclose(
                closest_idx_2, -1 * np.ones((n_batch, n_src_pts), dtype=np.int32)
            )
        )

    def test_closest_points_with_normal(self) -> None:
        np.random.seed(0)  # ensure repeatability

        n_src_pts = 6
        n_tgt_pts = 22
        n_batch = 3
        dim = 3
        src_pts = np.random.rand(n_batch, n_src_pts, dim).astype(np.float32)
        tgt_pts = np.random.rand(n_batch, n_tgt_pts, dim).astype(np.float32)

        def normalized(arr: npt.NDArray) -> npt.NDArray:
            norms = np.linalg.norm(arr, axis=-1, keepdims=True)
            return (arr / norms).astype(np.float32)

        src_normals = normalized(
            np.abs(np.random.rand(n_batch, n_src_pts, dim).astype(np.float32))
        )
        tgt_normals = normalized(
            np.abs(np.random.rand(n_batch, n_tgt_pts, dim).astype(np.float32))
        )

        (
            closest_pts,
            closest_normals,
            closest_idx,
            closest_valid,
        ) = geometry.find_closest_points(src_pts, src_normals, tgt_pts, tgt_normals)
        self.assertTrue(np.all(closest_valid))

        # Normals are all in the positive quadrant so no points should be rejected.
        (
            gt_closest_pts,
            gt_closest_normals,
            gt_closest_idx,
            _,
        ) = _brute_force_closest_points(src_pts, tgt_pts, tgt_normals)

        self.assertTrue(np.allclose(gt_closest_pts, closest_pts))
        self.assertTrue(np.allclose(gt_closest_normals, closest_normals))
        self.assertTrue(np.allclose(gt_closest_idx, closest_idx))

        # Now try with the opposite normals, all points should be rejected:
        _, _, closest_idx_2, closest_valid_2 = geometry.find_closest_points(
            src_pts, -src_normals, tgt_pts, tgt_normals
        )
        self.assertFalse(np.any(closest_valid_2))
        self.assertFalse(np.any(closest_idx_2 >= 0))
