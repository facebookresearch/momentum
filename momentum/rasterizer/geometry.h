/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/mesh.h>
#include <momentum/rasterizer/camera.h>
#include <Eigen/Core>
#include <utility>

namespace momentum::rasterizer {

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// Use the momentum Mesh class as the rasterizer Mesh type
using Mesh = momentum::MeshT<float>;

// Utility functions for creating primitives.
Mesh makeSphere(int subdivisionLevel);
Mesh makeCylinder(int numCircleSubdivisions, int numLengthSubdivisions);
std::array<Mesh, 2> makeCheckerboard(float width, int numChecks, int subdivisions);

Mesh makeCapsule(
    int numCircleSubdivisions,
    int numLengthSubdivisions,
    float startRadius = 1.0,
    float endRadius = 1.0,
    float length = 1.0);

Mesh makeArrow(
    int numCircleSubdivisions,
    int numLengthSubdivisions,
    float innerRadius,
    float outerRadius,
    float tipLength,
    float cylinderLength);

Mesh makeArrowhead(
    int numCircleSubdivisions,
    float innerRadius,
    float outerRadius,
    float length,
    float translation);

// Creates an octahedron which is basically two pyramids stuck together.
// "radius" is the width around the middle (where the two pyramid bases meet).
// "midFraction" is the ratio of the two pyramid heights.
std::tuple<Mesh, std::vector<Eigen::Vector3f>> makeOctahedron(
    float radius = 0.5,
    float midFraction = 0.5);

Eigen::Matrix4f
makeCylinderTransform(const Eigen::Vector3f& startPos, const Eigen::Vector3f& endPos, float radius);

Eigen::Matrix4f makeSphereTransform(const Eigen::Vector3f& center, float radius);

// Subdivides the lines until the longest segment is no longer than maxLength, except that it won't
// generate more than maxSubdivisions segments for each line (to prevent infinite recursion).
std::vector<Eigen::Vector3f> subdivideLines(
    const std::vector<Eigen::Vector3f>& lines,
    float maxLength,
    size_t maxSubdivisions = 100);

std::tuple<RowMatrixXf, RowMatrixXf, RowMatrixXi, RowMatrixXf, RowMatrixXi>
subdivideMeshNoSmoothing(
    Eigen::Ref<const RowMatrixXf> vertices_orig,
    Eigen::Ref<const RowMatrixXf> normals_orig,
    Eigen::Ref<const RowMatrixXi> triangles_orig,
    Eigen::Ref<const RowMatrixXf> textureCoords_orig,
    Eigen::Ref<const RowMatrixXi> textureTriangles_orig,
    float max_edge_length);

Mesh subdivideMeshNoSmoothing(const Mesh& mesh, float max_edge_length, size_t max_depth = 3);

std::vector<Eigen::Vector3f>
makeCameraFrustumLines(const Camera& camera, float distance, size_t nSamples = 20);

} // namespace momentum::rasterizer
