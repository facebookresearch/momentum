# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import pymomentum.geometry as geometry
import torch


def _brute_force_closest_points(
    src_pts: torch.Tensor,
    tgt_pts: torch.Tensor,
    tgt_normals: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_batch = src_pts.size(0)
    n_src_pts = src_pts.size(1)

    closest_pts = torch.zeros(src_pts.shape)
    closest_normals = torch.zeros(src_pts.shape)
    closest_index = torch.zeros(n_batch, n_src_pts, dtype=torch.int)

    for i_batch in range(n_batch):
        for j_pt in range(n_src_pts):
            gt_diff = (
                src_pts[i_batch, j_pt, :].expand_as(tgt_pts[i_batch, ...])
                - tgt_pts[i_batch, ...]
            )
            gt_dist = torch.linalg.norm(gt_diff, dim=-1, keepdim=False)
            gt_closest_dist, gt_closest_pt_idx = torch.min(gt_dist, dim=0)
            closest_index[i_batch, j_pt] = gt_closest_pt_idx
            closest_pts[i_batch, j_pt, :] = tgt_pts[i_batch, gt_closest_pt_idx, :]
            if tgt_normals is not None:
                closest_normals[i_batch, j_pt, :] = tgt_normals[
                    i_batch, gt_closest_pt_idx, :
                ]

    return closest_pts, closest_normals, closest_index, closest_index >= 0


class TestClosestPoints(unittest.TestCase):
    def test_closest_points(self) -> None:
        torch.manual_seed(0)  # ensure repeatability

        n_src_pts = 3
        n_tgt_pts = 20
        n_batch = 2
        dim = 3
        src_pts = torch.rand(n_batch, n_src_pts, dim)
        tgt_pts = torch.rand(n_batch, n_tgt_pts, dim)

        closest_pts, closest_idx, closest_valid = geometry.find_closest_points(
            src_pts, tgt_pts
        )
        closest_dist = torch.norm(closest_pts - src_pts, dim=-1, keepdim=False)
        self.assertTrue(torch.all(closest_valid))

        gt_closest_pts, _, gt_closest_idx, _ = _brute_force_closest_points(
            src_pts, tgt_pts
        )
        self.assertTrue(torch.allclose(gt_closest_pts, closest_pts))
        self.assertTrue(torch.allclose(gt_closest_idx, closest_idx))

        # Verify that if we pass in a small enough min_dist we don't return any points:
        min_all_dist, _ = torch.min(torch.flatten(closest_dist), dim=0)
        closest_pts_2, closest_idx_2, closest_valid_2 = geometry.find_closest_points(
            src_pts, tgt_pts, 0.5 * min_all_dist.item()
        )
        self.assertFalse(torch.any(closest_valid_2))
        self.assertTrue(
            torch.allclose(
                closest_idx_2, -1 * torch.ones(n_batch, n_src_pts, dtype=torch.int)
            )
        )

    def test_closest_points_with_normal(self) -> None:
        torch.manual_seed(0)  # ensure repeatability

        n_src_pts = 6
        n_tgt_pts = 22
        n_batch = 3
        dim = 3
        src_pts = torch.rand(n_batch, n_src_pts, dim)
        tgt_pts = torch.rand(n_batch, n_tgt_pts, dim)

        def normalized(t: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.normalize(t, dim=-1)

        src_normals = normalized(torch.abs(torch.rand(n_batch, n_src_pts, dim)))
        tgt_normals = normalized(torch.abs(torch.rand(n_batch, n_tgt_pts, dim)))

        (
            closest_pts,
            closest_normals,
            closest_idx,
            closest_valid,
        ) = geometry.find_closest_points(src_pts, src_normals, tgt_pts, tgt_normals)
        self.assertTrue(torch.all(closest_valid))

        # Normals are all in the positive quadrant so no points should be rejected.
        (
            gt_closest_pts,
            gt_closest_normals,
            gt_closest_idx,
            _,
        ) = _brute_force_closest_points(src_pts, tgt_pts, tgt_normals)

        self.assertTrue(torch.allclose(gt_closest_pts, closest_pts))
        self.assertTrue(torch.allclose(gt_closest_normals, closest_normals))
        self.assertTrue(torch.allclose(gt_closest_idx, closest_idx))

        # Now try with the opposite normals, all points should be rejected:
        _, _, closest_idx_2, closest_valid_2 = geometry.find_closest_points(
            src_pts, -src_normals, tgt_pts, tgt_normals
        )
        self.assertFalse(torch.any(closest_valid_2))
        self.assertFalse(torch.any(closest_idx_2 >= 0))
