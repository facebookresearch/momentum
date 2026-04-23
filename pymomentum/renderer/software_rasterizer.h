/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "pymomentum/renderer/rasterizer_primitives.h"

#include <momentum/camera/camera.h>
#include <momentum/character/character.h>
#include <momentum/rasterizer/fwd.h>
#include <momentum/rasterizer/rasterizer.h>

#include <ATen/ATen.h>
#include <pybind11/numpy.h>
#include <Eigen/Geometry>

#include <optional>

namespace pymomentum {

// A SIMD-based software rasterizer that supports the Nimble camera model
// and per-pixel Phong shading/lighting.
void rasterizeMesh(
    at::Tensor positions,
    std::optional<at::Tensor> normals,
    at::Tensor triangles,
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    const std::optional<pybind11::buffer>& vertexIndexBuffer,
    const std::optional<pybind11::buffer>& triangleIndexBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<at::Tensor> textureCoordinates,
    std::optional<at::Tensor> textureTriangles,
    std::optional<at::Tensor> perVertexDiffuseColor,
    std::optional<std::vector<momentum::rasterizer::Light>> lights,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset);

void rasterizeWireframe(
    at::Tensor positions,
    at::Tensor triangles,
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    float width,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset);

void rasterizeSpheres(
    at::Tensor center,
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    std::optional<at::Tensor> radius,
    std::optional<at::Tensor> color,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<std::vector<momentum::rasterizer::Light>> lights,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    int subdivisionLevel);

void rasterizeCylinders(
    at::Tensor start_position,
    at::Tensor end_position,
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    std::optional<at::Tensor> radius,
    std::optional<at::Tensor> color,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<std::vector<momentum::rasterizer::Light>> lights,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    int lengthSubdivisions,
    int radiusSubdivisions);

void rasterizeCapsules(
    at::Tensor transformation,
    at::Tensor radius,
    at::Tensor length,
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<std::vector<momentum::rasterizer::Light>> lights,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    int cylinderLengthSubdivisions,
    int cylinderRadiusSubdivisions);

enum class SkeletonStyle {
  Octahedrons,
  Pipes,
  Lines,
};

void rasterizeSkeleton(
    const momentum::Character& character,
    at::Tensor skeletonState,
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& jointMaterial,
    const std::optional<momentum::rasterizer::PhongMaterial>& cylinderMaterial,
    std::optional<std::vector<momentum::rasterizer::Light>> lights,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    std::optional<at::Tensor> activeJoints,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    float sphereRadius,
    float cylinderRadius,
    int sphereSubdivisionLevel,
    int cylinderLengthSubdivisions,
    int cylinderRadiusSubdivisions,
    SkeletonStyle style);

void rasterizeCharacter(
    const momentum::Character& character,
    at::Tensor skeletonState,
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    const std::optional<pybind11::buffer>& vertexIndexBuffer,
    const std::optional<pybind11::buffer>& triangleIndexBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<at::Tensor> perVertexDiffuseColor,
    std::optional<std::vector<momentum::rasterizer::Light>> lights,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    std::optional<Eigen::Vector3f> wireframeColor);

void rasterizeCheckerboard(
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& material1,
    const std::optional<momentum::rasterizer::PhongMaterial>& material2,
    std::optional<std::vector<momentum::rasterizer::Light>> lights,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    float width,
    int numChecks,
    int subdivisions);

} // namespace pymomentum
