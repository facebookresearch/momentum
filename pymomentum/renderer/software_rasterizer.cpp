/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/renderer/software_rasterizer.h"

#include "pymomentum/renderer/rasterizer_utility.h"
#include "pymomentum/tensor_momentum/tensor_momentum_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/character/character.h>
#include <momentum/character/linear_skinning.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/common/exception.h>
#include <momentum/math/fwd.h>
#include <momentum/math/mesh.h>

#include <momentum/rasterizer/geometry.h>
#include <momentum/rasterizer/image.h>
#include <momentum/rasterizer/rasterizer.h>

#include <drjit/packet.h>
#include <fmt/format.h>
#include <pybind11/pybind11.h>

#include <optional>
#include <stdexcept>
#include <utility>

namespace py = pybind11;

namespace pymomentum {

using rasterizer_detail::convertLightsToEyeSpace;
using rasterizer_detail::make_mdspan;
using rasterizer_detail::validateRasterizerBuffers;

std::optional<at::Tensor> maybeSelect(std::optional<at::Tensor> tensor, int64_t index) {
  if (!tensor) {
    return {};
  }

  return tensor->select(0, index);
}

// rasterize with CameraData3d which considers distortion
void rasterizeMesh(
    at::Tensor positions,
    std::optional<at::Tensor> normals,
    at::Tensor triangles,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    std::optional<at::Tensor> surfaceNormalsBuffer,
    std::optional<at::Tensor> vertexIndexBuffer,
    std::optional<at::Tensor> triangleIndexBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<at::Tensor> textureCoordinates,
    std::optional<at::Tensor> textureTriangles,
    std::optional<at::Tensor> perVertexDiffuseColor,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const int nVertsBindingID = -1;
  const int nTrisBindingID = -2;

  TensorChecker checker("rasterize");
  positions = checker.validateAndFixTensor(
      positions, "positions", {nVertsBindingID, 3}, {"nVertices", "xyz"}, at::kFloat, true, false);

  if (normals.has_value()) {
    normals = checker.validateAndFixTensor(
        normals.value(),
        "normals",
        {nVertsBindingID, 3},
        {"nVertices", "xyz"},
        at::kFloat,
        true,
        true);
  }

  triangles = checker.validateAndFixTensor(
      triangles, "triangles", {nTrisBindingID, 3}, {"nTriangles", "xyz"}, at::kInt, true, false);

  if (textureCoordinates.has_value()) {
    textureCoordinates = checker.validateAndFixTensor(
        textureCoordinates.value(),
        "texture_coordinates",
        {nVertsBindingID, 2},
        {"nVerts", "uv"},
        at::kFloat,
        true,
        false);
  }

  if (textureTriangles.has_value()) {
    textureTriangles = checker.validateAndFixTensor(
        textureTriangles.value(),
        "triangles",
        {nTrisBindingID, 3},
        {"nTriangles", "xyz"},
        at::kInt,
        true,
        false);
  }

  if (perVertexDiffuseColor.has_value()) {
    perVertexDiffuseColor = checker.validateAndFixTensor(
        perVertexDiffuseColor.value(),
        "perVertexDiffuseColor",
        {nVertsBindingID, 3},
        {"nVerts", "rgb"},
        at::kFloat,
        true,
        false);
  }

  std::tie(zBuffer, rgbBuffer, surfaceNormalsBuffer, vertexIndexBuffer, triangleIndexBuffer) =
      validateRasterizerBuffers(
          checker,
          camera,
          zBuffer,
          rgbBuffer,
          surfaceNormalsBuffer,
          vertexIndexBuffer,
          triangleIndexBuffer);

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  Eigen::VectorXf emptyTextureCoords;
  Eigen::VectorXf emptyNormals;
  Eigen::VectorXi emptyTextureTriangles;
  Eigen::VectorXf emptyPerVertexDiffuseColor;
  momentum::rasterizer::rasterizeMesh(
      toEigenMap<float>(positions),
      normals.has_value() ? toEigenMap<float>(*normals) : emptyNormals,
      toEigenMap<int>(triangles),
      textureCoordinates.has_value() ? toEigenMap<float>(*textureCoordinates) : emptyTextureCoords,
      textureTriangles.has_value() ? toEigenMap<int32_t>(*textureTriangles) : emptyTextureTriangles,
      perVertexDiffuseColor.has_value() ? toEigenMap<float>(*perVertexDiffuseColor)
                                        : emptyPerVertexDiffuseColor,
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      material.value_or(momentum::rasterizer::PhongMaterial{}),
      make_mdspan<float, 2>(zBuffer),
      make_mdspan<float, 3>(rgbBuffer),
      make_mdspan<float, 3>(surfaceNormalsBuffer),
      make_mdspan<int, 2>(vertexIndexBuffer),
      make_mdspan<int, 2>(triangleIndexBuffer),
      lights_eye,
      backfaceCulling,
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeWireframe(
    at::Tensor positions,
    at::Tensor triangles,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    float thickness,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const int nVertsBindingID = -1;
  const int nTrisBindingID = -2;

  TensorChecker checker("rasterize");
  positions = checker.validateAndFixTensor(
      positions, "positions", {nVertsBindingID, 3}, {"nVertices", "xyz"}, at::kFloat, true, false);

  triangles = checker.validateAndFixTensor(
      triangles, "triangles", {nTrisBindingID, 3}, {"nTriangles", "xyz"}, at::kInt, true, false);

  std::tie(zBuffer, rgbBuffer, std::ignore, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, {}, {}, {});

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel.
  Eigen::VectorXf emptyTextureCoords;
  Eigen::VectorXi emptyTextureTriangles;
  momentum::rasterizer::rasterizeWireframe(
      toEigenMap<float>(positions),
      toEigenMap<int>(triangles),
      camera,
      modelMatrix.value_or(Eigen::Matrix4f::Identity()),
      nearClip,
      color.value_or(Eigen::Vector3f{1, 1, 1}),
      thickness,
      make_mdspan<float, 2>(zBuffer),
      make_mdspan<float, 3>(rgbBuffer),
      backfaceCulling,
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));
}

void rasterizeSpheres(
    at::Tensor center,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    std::optional<at::Tensor> surfaceNormalsBuffer,
    std::optional<at::Tensor> radius,
    std::optional<at::Tensor> color,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix_in,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    int subdivisionLevel) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  const int nSpheresBindingID = -1;

  TensorChecker checker("rasterize");
  center = checker.validateAndFixTensor(
      center, "center", {nSpheresBindingID, 3}, {"nSpheres", "xyz"}, at::kFloat, true, false);

  if (radius) {
    *radius = checker.validateAndFixTensor(
        *radius, "radius", {nSpheresBindingID}, {"nSpheres"}, at::kFloat, true, false);
  }

  if (color) {
    *color = checker.validateAndFixTensor(
        *color, "color", {nSpheresBindingID, 3}, {"nSpheres", "rgb"}, at::kFloat, true, false);
  }

  std::tie(zBuffer, rgbBuffer, surfaceNormalsBuffer, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, surfaceNormalsBuffer, {}, {});

  const auto nSpheres = checker.getBoundValue(nSpheresBindingID);

  const auto sphereMesh = momentum::rasterizer::makeSphere(subdivisionLevel);

  const Eigen::Matrix4f modelMatrix = modelMatrix_in.value_or(Eigen::Matrix4f::Identity());

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel

  for (int i = 0; i < nSpheres; ++i) {
    const float radiusVal = radius ? toEigenMap<float>(*radius)(i) : 1.0f;
    auto materialCur = material.value_or(momentum::rasterizer::PhongMaterial());
    if (color) {
      materialCur.diffuseColor = toEigenMap<float>(*color).segment<3>(3 * i);
    }

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translate(toEigenMap<float>(center).segment<3>(3 * i));
    transform.scale(radiusVal);

    momentum::rasterizer::rasterizeMesh(
        sphereMesh,
        camera,
        modelMatrix * transform.matrix(),
        nearClip,
        materialCur,
        make_mdspan<float, 2>(zBuffer),
        make_mdspan<float, 3>(rgbBuffer),
        make_mdspan<float, 3>(surfaceNormalsBuffer),
        {},
        {},
        lights_eye,
        backfaceCulling,
        depthOffset,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

void rasterizeCylinders(
    at::Tensor start_position,
    at::Tensor end_position,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    std::optional<at::Tensor> surfaceNormalsBuffer,
    std::optional<at::Tensor> radius,
    std::optional<at::Tensor> color,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix_in,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    int lengthSubdivisions,
    int radiusSubdivisions) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  const int nCylindersBindingID = -1;

  TensorChecker checker("rasterize");
  start_position = checker.validateAndFixTensor(
      start_position,
      "start_position",
      {nCylindersBindingID, 3},
      {"nCylinders", "xyz"},
      at::kFloat,
      true,
      false);

  end_position = checker.validateAndFixTensor(
      end_position,
      "end_position",
      {nCylindersBindingID, 3},
      {"nCylinders", "xyz"},
      at::kFloat,
      true,
      false);

  if (radius) {
    *radius = checker.validateAndFixTensor(
        *radius, "radius", {nCylindersBindingID}, {"nSpheres"}, at::kFloat, true, false);
  }

  if (color) {
    *color = checker.validateAndFixTensor(
        *color, "color", {nCylindersBindingID, 3}, {"nCylinders", "rgb"}, at::kFloat, true, false);
  }

  std::tie(zBuffer, rgbBuffer, surfaceNormalsBuffer, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, surfaceNormalsBuffer, {}, {});

  const auto nCylinders = checker.getBoundValue(nCylindersBindingID);

  const auto cylinderMesh =
      momentum::rasterizer::makeCylinder(radiusSubdivisions, lengthSubdivisions);

  const Eigen::Matrix4f modelMatrix = modelMatrix_in.value_or(Eigen::Matrix4f::Identity());

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel

  for (int i = 0; i < nCylinders; ++i) {
    const Eigen::Vector3f startPos = toEigenMap<float>(start_position).segment<3>(3 * i);
    const Eigen::Vector3f endPos = toEigenMap<float>(end_position).segment<3>(3 * i);
    const float radiusVal = radius ? toEigenMap<float>(*radius)(i) : 1.0f;
    auto materialCur = material.value_or(momentum::rasterizer::PhongMaterial());
    if (color) {
      materialCur.diffuseColor = toEigenMap<float>(*color).segment<3>(3 * i);
    }

    const float length = (endPos - startPos).norm();
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translate(startPos);
    transform.rotate(
        Eigen::Quaternionf::FromTwoVectors(
            Eigen::Vector3f::UnitX(), (endPos - startPos).normalized()));
    transform.scale(Eigen::Vector3f(length, radiusVal, radiusVal));

    momentum::rasterizer::rasterizeMesh(
        cylinderMesh,
        camera,
        modelMatrix * transform.matrix(),
        nearClip,
        materialCur,
        make_mdspan<float, 2>(zBuffer),
        make_mdspan<float, 3>(rgbBuffer),
        make_mdspan<float, 3>(surfaceNormalsBuffer),
        {},
        {},
        lights_eye,
        backfaceCulling,
        depthOffset,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

void rasterizeCapsules(
    at::Tensor transformation,
    at::Tensor radius,
    at::Tensor length,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    std::optional<at::Tensor> surfaceNormalsBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix_in,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    int cylinderLengthSubdivisions,
    int cylinderRadiusSubdivisions) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  const int nCapsulesBindingId = -1;

  TensorChecker checker("rasterize");
  transformation = checker.validateAndFixTensor(
      transformation,
      "transformation",
      {nCapsulesBindingId, 4, 4},
      {"nCapsules", "rows", "cols"},
      at::kFloat,
      true,
      false);

  radius = checker.validateAndFixTensor(
      radius,
      "radius",
      {nCapsulesBindingId, 2},
      {"nCapsules", "startEnd"},
      at::kFloat,
      true,
      false);

  length = checker.validateAndFixTensor(
      length, "length", {nCapsulesBindingId}, {"nCapsules"}, at::kFloat, true, false);

  std::tie(zBuffer, rgbBuffer, surfaceNormalsBuffer, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, surfaceNormalsBuffer, {}, {});

  const auto nCapsules = checker.getBoundValue(nCapsulesBindingId);

  const Eigen::Matrix4f modelMatrix = modelMatrix_in.value_or(Eigen::Matrix4f::Identity());

  // Capsules are closed surfaces so we might as well always cull:
  const bool backfaceCulling = true;

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel

  for (int i = 0; i < nCapsules; ++i) {
    const Eigen::Matrix4f transform =
        toEigenMap<float>(transformation).segment<16>(16 * i).reshaped(4, 4).transpose();

    const Eigen::Vector2f radiusVal = toEigenMap<float>(radius).segment<2>(2 * i);
    const float lengthVal = toEigenMap<float>(length)(i);

    auto materialCur = material.value_or(momentum::rasterizer::PhongMaterial());

    momentum::rasterizer::rasterizeMesh(
        momentum::rasterizer::makeCapsule(
            cylinderRadiusSubdivisions,
            cylinderLengthSubdivisions,
            radiusVal[0],
            radiusVal[1],
            lengthVal),
        camera,
        modelMatrix * transform.matrix(),
        nearClip,
        materialCur,
        make_mdspan<float, 2>(zBuffer),
        make_mdspan<float, 3>(rgbBuffer),
        make_mdspan<float, 3>(surfaceNormalsBuffer),
        {},
        {},
        lights_eye,
        backfaceCulling,
        depthOffset,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

namespace {

struct SkeletonRenderingAssets {
  momentum::rasterizer::Mesh jointMesh;
  std::vector<Eigen::Vector3f> jointLines;
  std::vector<Eigen::Vector3f> jointCircles;
  momentum::rasterizer::Mesh sphereMesh;
  float lineThickness;
  float circleRadius;
  Eigen::Vector3f lineColor;
  Eigen::Vector3f circleColor;
};

SkeletonRenderingAssets initSkeletonAssets(
    SkeletonStyle style,
    const std::optional<momentum::rasterizer::PhongMaterial>& jointMaterial,
    const std::optional<momentum::rasterizer::PhongMaterial>& cylinderMaterial,
    float sphereRadius,
    float cylinderRadius,
    int sphereSubdivisionLevel,
    int cylinderLengthSubdivisions,
    int cylinderRadiusSubdivisions) {
  SkeletonRenderingAssets assets;

  if (style == SkeletonStyle::Octahedrons) {
    std::tie(assets.jointMesh, assets.jointLines) = momentum::rasterizer::makeOctahedron(0.15, 0.1);
    assets.sphereMesh = momentum::rasterizer::makeSphere(sphereSubdivisionLevel);

  } else if (style == SkeletonStyle::Lines) {
    for (size_t i = 0; i < static_cast<size_t>(cylinderLengthSubdivisions); ++i) {
      assets.jointLines.emplace_back(i / float(cylinderLengthSubdivisions), 0, 0);
      assets.jointLines.emplace_back((i + 1) / float(cylinderLengthSubdivisions), 0, 0);
    }
    assets.jointCircles.emplace_back(0, 0, 0);
  } else {
    assets.sphereMesh = momentum::rasterizer::makeSphere(sphereSubdivisionLevel);
    assets.jointMesh =
        momentum::rasterizer::makeCylinder(cylinderRadiusSubdivisions, cylinderLengthSubdivisions);
  }

  assets.lineThickness =
      (style == SkeletonStyle::Lines) ? std::max(1.0f, 2.0f * cylinderRadius) : 1.0f;
  assets.circleRadius = std::max(1.0f, sphereRadius);

  assets.lineColor = (style == SkeletonStyle::Lines)
      ? cylinderMaterial.value_or(momentum::rasterizer::PhongMaterial{}).diffuseColor
      : Eigen::Vector3f::Constant(0.3f);
  assets.circleColor = jointMaterial.value_or(momentum::rasterizer::PhongMaterial{}).diffuseColor;

  return assets;
}

void rasterizeSkeletonJoints(
    const SkeletonRenderingAssets& assets,
    const std::vector<bool>& activeJoints,
    const Eigen::Ref<const Eigen::VectorXf>& skelStateMap,
    const momentum::rasterizer::Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const std::optional<momentum::rasterizer::PhongMaterial>& jointMaterial,
    rasterizer_detail::Span<float, 2> zBufferSpan,
    rasterizer_detail::Span<float, 3> rgbBufferSpan,
    rasterizer_detail::Span<float, 3> surfaceNormalsSpan,
    const std::vector<momentum::rasterizer::Light>& lights_eye,
    bool backfaceCulling,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    float sphereRadius) {
  const size_t nJoints = activeJoints.size();

  for (size_t jJoint = 0; jJoint < nJoints; ++jJoint) {
    if (!activeJoints[jJoint]) {
      continue;
    }

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translate(skelStateMap.segment<3>(8 * jJoint));
    transform.scale(sphereRadius);

    if (!assets.jointCircles.empty()) {
      momentum::rasterizer::rasterizeCircles(
          assets.jointCircles,
          camera,
          modelMatrix * transform.matrix(),
          nearClip,
          assets.lineColor,
          assets.circleColor,
          0.5f * assets.lineThickness,
          assets.circleRadius,
          zBufferSpan,
          rgbBufferSpan,
          depthOffset,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }

    if (!assets.sphereMesh.vertices.empty()) {
      momentum::rasterizer::rasterizeMesh(
          assets.sphereMesh,
          camera,
          modelMatrix * transform.matrix(),
          nearClip,
          jointMaterial.value_or(momentum::rasterizer::PhongMaterial{}),
          zBufferSpan,
          rgbBufferSpan,
          surfaceNormalsSpan,
          {},
          {},
          lights_eye,
          backfaceCulling,
          depthOffset,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }
  }
}

void rasterizeSkeletonBones(
    const SkeletonRenderingAssets& assets,
    SkeletonStyle style,
    const std::vector<std::pair<size_t, size_t>>& jointConnections,
    const Eigen::Ref<const Eigen::VectorXf>& skelStateMap,
    const momentum::rasterizer::Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const std::optional<momentum::rasterizer::PhongMaterial>& cylinderMaterial,
    rasterizer_detail::Span<float, 2> zBufferSpan,
    rasterizer_detail::Span<float, 3> rgbBufferSpan,
    rasterizer_detail::Span<float, 3> surfaceNormalsSpan,
    const std::vector<momentum::rasterizer::Light>& lights_eye,
    bool backfaceCulling,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    float cylinderRadius) {
  for (const auto& [childJoint, parentJoint] : jointConnections) {
    const auto parentXF = momentum::Transform(
        skelStateMap.segment<3>(8 * parentJoint),
        Eigen::Quaternionf(skelStateMap.segment<4>(8 * parentJoint + 3)),
        skelStateMap(8 * parentJoint + 7));

    const auto childXF = momentum::Transform(
        skelStateMap.segment<3>(8 * childJoint),
        Eigen::Quaternionf(skelStateMap.segment<4>(8 * childJoint + 3)),
        skelStateMap(8 * childJoint + 7));

    const Eigen::Vector3f diff = childXF.translation - parentXF.translation;
    const float length = diff.norm();
    if (length < 1e-6) {
      continue;
    }

    // x always points to the next joint, y and z axes are aligned
    // to the local y and z axes of the parent joint where possible:
    const Eigen::Vector3f xVec = diff.normalized();
    Eigen::Vector3f yVec = (parentXF.toLinear() * Eigen::Vector3f::UnitY()).normalized();
    Eigen::Vector3f zVec = (parentXF.toLinear() * Eigen::Vector3f::UnitZ()).normalized();
    yVec = xVec.cross(zVec).normalized();
    zVec = xVec.cross(yVec).normalized();

    Eigen::Matrix3f rotMat = Eigen::Matrix3f::Identity();
    rotMat.col(0) = xVec;
    rotMat.col(1) = yVec;
    rotMat.col(2) = zVec;

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translate(parentXF.translation);
    transform.linear() = rotMat;

    if (style == SkeletonStyle::Octahedrons) {
      transform.scale(Eigen::Vector3f::Constant(length));
    } else {
      transform.scale(Eigen::Vector3f(length, cylinderRadius, cylinderRadius));
    }

    if (!assets.jointMesh.vertices.empty()) {
      momentum::rasterizer::rasterizeMesh(
          assets.jointMesh,
          camera,
          modelMatrix * transform.matrix(),
          nearClip,
          cylinderMaterial.value_or(momentum::rasterizer::PhongMaterial{}),
          zBufferSpan,
          rgbBufferSpan,
          surfaceNormalsSpan,
          {},
          {},
          lights_eye,
          backfaceCulling,
          depthOffset,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }

    if (!assets.jointLines.empty()) {
      momentum::rasterizer::rasterizeLines(
          assets.jointLines,
          camera,
          modelMatrix * transform.matrix(),
          nearClip,
          assets.lineColor,
          assets.lineThickness,
          zBufferSpan,
          rgbBufferSpan,
          depthOffset - 0.001f,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }
  }
}

} // namespace

void rasterizeSkeleton(
    const momentum::Character& character,
    at::Tensor skeletonState,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    std::optional<at::Tensor> surfaceNormalsBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& jointMaterial,
    const std::optional<momentum::rasterizer::PhongMaterial>& cylinderMaterial,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix_in,
    bool backfaceCulling,
    std::optional<at::Tensor> activeJoints_in,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    float sphereRadius,
    float cylinderRadius,
    int sphereSubdivisionLevel,
    int cylinderLengthSubdivisions,
    int cylinderRadiusSubdivisions,
    SkeletonStyle style) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  TensorChecker checker("rasterize_skeleton");

  skeletonState = checker.validateAndFixTensor(
      skeletonState,
      "skeletonState",
      {(int)character.skeleton.joints.size(), 8},
      {"nJoints", "pos+rot+scale"},
      at::kFloat,
      true,
      false);

  std::tie(zBuffer, rgbBuffer, surfaceNormalsBuffer, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, surfaceNormalsBuffer, {}, {});

  const auto nJoints = character.skeleton.joints.size();

  const Eigen::Matrix4f modelMatrix = modelMatrix_in.value_or(Eigen::Matrix4f::Identity());

  std::vector<bool> activeJoints(nJoints);
  if (activeJoints_in.has_value()) {
    activeJoints =
        tensorToJointSet(character.skeleton, *activeJoints_in, DefaultJointSet::ALL_ONES);
  } else {
    // Make all joints active except b_world, which we usually don't want to
    // render because it's confusing:
    for (size_t i = 0; i < character.skeleton.joints.size(); ++i) {
      activeJoints[i] = (character.skeleton.joints[i].name != "body_world");
    }
  }

  // Now construct the parent relationships:
  std::vector<std::pair<size_t, size_t>> jointConnections;
  for (size_t iJoint = 0; iJoint < character.skeleton.joints.size(); ++iJoint) {
    if (!activeJoints[iJoint]) {
      continue;
    }

    size_t grandparent = character.skeleton.joints[iJoint].parent;
    while (grandparent != momentum::kInvalidIndex && !activeJoints[grandparent]) {
      MT_THROW_IF(
          grandparent >= character.skeleton.joints.size(),
          "Grandparent joint index {} exceeds skeleton size {}",
          grandparent,
          character.skeleton.joints.size());
      grandparent = character.skeleton.joints[grandparent].parent;
    }

    if (grandparent != momentum::kInvalidIndex) {
      jointConnections.emplace_back(iJoint, grandparent);
    }
  }

  const auto assets = initSkeletonAssets(
      style,
      jointMaterial,
      cylinderMaterial,
      sphereRadius,
      cylinderRadius,
      sphereSubdivisionLevel,
      cylinderLengthSubdivisions,
      cylinderRadiusSubdivisions);

  const auto skelStateMap = toEigenMap<float>(skeletonState);

  rasterizeSkeletonJoints(
      assets,
      activeJoints,
      skelStateMap,
      camera,
      modelMatrix,
      nearClip,
      jointMaterial,
      make_mdspan<float, 2>(zBuffer),
      make_mdspan<float, 3>(rgbBuffer),
      make_mdspan<float, 3>(surfaceNormalsBuffer),
      lights_eye,
      backfaceCulling,
      depthOffset,
      imageOffset,
      sphereRadius);

  rasterizeSkeletonBones(
      assets,
      style,
      jointConnections,
      skelStateMap,
      camera,
      modelMatrix,
      nearClip,
      cylinderMaterial,
      make_mdspan<float, 2>(zBuffer),
      make_mdspan<float, 3>(rgbBuffer),
      make_mdspan<float, 3>(surfaceNormalsBuffer),
      lights_eye,
      backfaceCulling,
      depthOffset,
      imageOffset,
      cylinderRadius);
}

void rasterizeCharacter(
    const momentum::Character& character,
    at::Tensor skeletonState,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    std::optional<at::Tensor> surfaceNormalsBuffer,
    std::optional<at::Tensor> vertexIndexBuffer,
    std::optional<at::Tensor> triangleIndexBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<at::Tensor> perVertexDiffuseColor,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix_in,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    std::optional<Eigen::Vector3f> wireframeColor) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (!character.skinWeights || !character.mesh) {
    throw std::runtime_error("Mesh is missing from character.");
  }

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  TensorChecker checker("rasterize_character");

  skeletonState = checker.validateAndFixTensor(
      skeletonState,
      "skeletonState",
      {(int)character.skeleton.joints.size(), 8},
      {"nJoints", "pos+rot+scale"},
      at::kFloat,
      true,
      false);

  if (perVertexDiffuseColor) {
    perVertexDiffuseColor = checker.validateAndFixTensor(
        *perVertexDiffuseColor,
        "perVertexDiffuseColor",
        {(int)character.mesh->vertices.size(), 3},
        {"nVertices", "rgb"},
        at::kFloat,
        true,
        false);
  }

  std::tie(zBuffer, rgbBuffer, surfaceNormalsBuffer, vertexIndexBuffer, triangleIndexBuffer) =
      validateRasterizerBuffers(
          checker,
          camera,
          zBuffer,
          rgbBuffer,
          surfaceNormalsBuffer,
          vertexIndexBuffer,
          triangleIndexBuffer);

  const Eigen::Matrix4f modelMatrix = modelMatrix_in.value_or(Eigen::Matrix4f::Identity());

  const auto nJoints = character.skeleton.joints.size();

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel

  std::vector<momentum::JointState> jointStates(character.skeleton.joints.size());
  const auto skelStateMap = toEigenMap<float>(skeletonState);
  for (size_t jJoint = 0; jJoint < nJoints; ++jJoint) {
    const auto tf = momentum::Transform(
        skelStateMap.segment<3>(8 * jJoint),
        Eigen::Quaternionf(skelStateMap.segment<4>(8 * jJoint + 3)),
        skelStateMap(8 * jJoint + 7));

    auto& js = jointStates.at(jJoint);
    js.transform = tf;
  }

  momentum::Mesh posedMesh = *character.mesh;
  momentum::applySSD(
      character.inverseBindPose, *character.skinWeights, *character.mesh, jointStates, posedMesh);

  // Don't trust the skinned normals:
  posedMesh.updateNormals();

  momentum::rasterizer::rasterizeMesh(
      posedMesh.vertices,
      posedMesh.normals,
      posedMesh.faces,
      posedMesh.texcoords,
      posedMesh.texcoord_faces,
      perVertexDiffuseColor.has_value() ? toEigenMap<float>(*perVertexDiffuseColor)
                                        : Eigen::VectorXf{},
      camera,
      modelMatrix,
      nearClip,
      material.value_or(momentum::rasterizer::PhongMaterial{}),
      make_mdspan<float, 2>(zBuffer),
      make_mdspan<float, 3>(rgbBuffer),
      make_mdspan<float, 3>(surfaceNormalsBuffer),
      make_mdspan<int, 2>(vertexIndexBuffer),
      make_mdspan<int, 2>(triangleIndexBuffer),
      lights_eye,
      backfaceCulling,
      depthOffset,
      imageOffset.value_or(Eigen::Vector2f::Zero()));

  if (wireframeColor.has_value()) {
    momentum::rasterizer::rasterizeWireframe(
        posedMesh.vertices,
        posedMesh.faces,
        camera,
        modelMatrix,
        nearClip,
        *wireframeColor,
        1.0f,
        make_mdspan<float, 2>(zBuffer),
        make_mdspan<float, 3>(rgbBuffer),
        backfaceCulling,
        depthOffset - 0.001,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

void rasterizeCheckerboard(
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    std::optional<at::Tensor> surfaceNormalsBuffer,
    const std::optional<momentum::rasterizer::PhongMaterial>& material1_in,
    const std::optional<momentum::rasterizer::PhongMaterial>& material2_in,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    bool backfaceCulling,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    float width,
    int numChecks,
    int subdivisions) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  if (numChecks <= 0) {
    throw std::runtime_error("num_checks should be positive.");
  }

  if (width <= 0) {
    throw std::runtime_error("width should be positive.");
  }

  if (subdivisions < 1) {
    throw std::runtime_error("subdivisions should be positive.");
  }

  const auto lights_eye = convertLightsToEyeSpace(std::move(lights_world), camera);

  TensorChecker checker("rasterize_checkerboard");

  std::tie(zBuffer, rgbBuffer, surfaceNormalsBuffer, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, surfaceNormalsBuffer, {}, {});

  const auto& meshes = momentum::rasterizer::makeCheckerboard(width, numChecks, subdivisions);

  // Simply white/gray scheme.  All-white or all-black tends to look bad in the
  // render:
  std::array<momentum::rasterizer::PhongMaterial, 2> materials = {
      material1_in.value_or(momentum::rasterizer::PhongMaterial{Eigen::Vector3f(0.8f, 0.8f, 0.8f)}),
      material2_in.value_or(
          momentum::rasterizer::PhongMaterial{Eigen::Vector3f(0.5f, 0.5f, 0.5f)})};

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel
  for (int k = 0; k < 2; ++k) {
    momentum::rasterizer::rasterizeMesh(
        meshes[k],
        camera,
        modelMatrix.value_or(Eigen::Matrix4f::Identity()),
        nearClip,
        materials[k],
        make_mdspan<float, 2>(zBuffer),
        make_mdspan<float, 3>(rgbBuffer),
        make_mdspan<float, 3>(surfaceNormalsBuffer),
        {},
        {},
        lights_eye,
        backfaceCulling,
        depthOffset,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

} // namespace pymomentum
