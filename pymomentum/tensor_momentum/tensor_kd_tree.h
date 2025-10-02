/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

#include <tuple>

namespace pymomentum {

// Returns [closest point, closest index, valid]
std::tuple<at::Tensor, at::Tensor, at::Tensor> findClosestPoints(
    at::Tensor points_source,
    at::Tensor points_target,
    float maxDist);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
findClosestPointsWithNormals(
    at::Tensor points_source,
    at::Tensor normals_source,
    at::Tensor points_target,
    at::Tensor normals_target,
    float maxDist,
    float maxNormalDot);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
findClosestPointsOnMesh(
    at::Tensor points_source,
    at::Tensor vertices_target,
    at::Tensor faces_target,
    float maxDist);

} // namespace pymomentum
