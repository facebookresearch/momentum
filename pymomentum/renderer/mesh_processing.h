// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <Eigen/Core>

#include <optional>

namespace pymomentum {

using RowMatrixf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::tuple<RowMatrixf, RowMatrixf, RowMatrixi, RowMatrixf, RowMatrixi> subdivideMesh(
    RowMatrixf vertices,
    RowMatrixf normals,
    RowMatrixi triangles,
    std::optional<RowMatrixf> textureCoords,
    std::optional<RowMatrixi> textureTriangles,
    int levels,
    float max_edge_length);

} // namespace pymomentum
