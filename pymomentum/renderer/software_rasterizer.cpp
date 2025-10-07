// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "pymomentum/renderer/software_rasterizer.h"

#include "pymomentum/tensor_momentum/tensor_momentum_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/character/character.h>
#include <momentum/character/linear_skinning.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
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

namespace {

template <typename T, size_t R>
Kokkos::mdspan<T, Kokkos::dextents<std::ptrdiff_t, R>> make_mdspan(const at::Tensor& t) {
  if (t.scalar_type() != toScalarType<T>()) {
    throw std::runtime_error("Scalar type incorrect in mdspan conversion.");
  }

  if (t.dim() != R) {
    throw std::runtime_error(fmt::format(
        "Incorrect dimension in mdspan conversion; expected ndim={} but got {}",
        R,
        formatTensorSizes(t)));
  }

  if (!t.device().is_cpu()) {
    throw std::runtime_error("Expected CPU tensor in mdspan conversion");
  }

  std::array<std::ptrdiff_t, R> extents{};
  for (int64_t i = 0; i < t.dim(); ++i) {
    extents[i] = t.size(i);
  }

  return Kokkos::mdspan<T, Kokkos::dextents<std::ptrdiff_t, R>>(t.data_ptr<T>(), extents);
}

template <typename T, size_t R>
Kokkos::mdspan<T, Kokkos::dextents<std::ptrdiff_t, R>> make_mdspan(std::optional<at::Tensor> t) {
  if (!t.has_value()) {
    return Kokkos::mdspan<T, Kokkos::dextents<std::ptrdiff_t, R>>();
  }

  return make_mdspan<T, R>(t.value());
}

template <typename T, size_t R>
Kokkos::mdspan<T, Kokkos::dextents<std::ptrdiff_t, R>> make_mdspan(const py::buffer_info& info) {
  if (info.format != py::format_descriptor<T>::format()) {
    throw std::runtime_error("Incorrect scalar format in Numpy array.");
  }

  if (info.ndim != R) {
    throw std::runtime_error(fmt::format(
        "Incorrect dimension in mdspan conversion; expected ndim={} but got {}",
        R,
        formatTensorSizes(info.shape)));
  }

  std::array<std::ptrdiff_t, R> extents{};
  for (int64_t i = 0; i < R; ++i) {
    extents.at(i) = info.shape.at(i);
  }

  return Kokkos::mdspan<T, Kokkos::dextents<std::ptrdiff_t, R>>(static_cast<T*>(info.ptr), extents);
}

std::tuple<
    at::Tensor,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>>
validateRasterizerBuffers(
    TensorChecker& checker,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    std::optional<at::Tensor> surfaceNormalsBuffer = {},
    std::optional<at::Tensor> vertexIndexBuffer = {},
    std::optional<at::Tensor> triangleIndexBuffer = {}) {
  // Values chosen to not conflict with any other bound values:
  const int widthBindingID = -2003;
  const int heightBindingID = -2004;

  const int32_t imageWidth = camera.imageWidth();
  const int32_t imageHeight = camera.imageHeight();
  const int32_t paddedWidth = momentum::rasterizer::padImageWidthForRasterizer(imageWidth);

  zBuffer = checker.validateAndFixTensor(
      zBuffer,
      "zBuffer",
      {heightBindingID, widthBindingID},
      {"imageHeight", "imageWidth"},
      at::kFloat,
      true,
      false);

  if (rgbBuffer) {
    *rgbBuffer = checker.validateAndFixTensor(
        *rgbBuffer,
        "rgbBuffer",
        {heightBindingID, widthBindingID, 3},
        {"imageHeight", "imageWidth", "rgb"},
        at::kFloat,
        true,
        false);
  }

  if (surfaceNormalsBuffer) {
    *surfaceNormalsBuffer = checker.validateAndFixTensor(
        *surfaceNormalsBuffer,
        "surfaceNormalsBuffer",
        {heightBindingID, widthBindingID, 3},
        {"imageHeight", "imageWidth", "rgb"},
        at::kFloat,
        true,
        false);
  }

  if (vertexIndexBuffer) {
    *vertexIndexBuffer = checker.validateAndFixTensor(
        *vertexIndexBuffer,
        "vertexIndexBuffer",
        {heightBindingID, widthBindingID},
        {"imageHeight", "imageWidth"},
        at::kInt,
        true,
        false);
  }

  if (triangleIndexBuffer) {
    *triangleIndexBuffer = checker.validateAndFixTensor(
        *triangleIndexBuffer,
        "triangleIndexBuffer",
        {heightBindingID, widthBindingID},
        {"imageHeight", "imageWidth"},
        at::kInt,
        true,
        false);
  }

  const auto zBufferWidth = checker.getBoundValue(widthBindingID);
  const auto zBufferHeight = checker.getBoundValue(heightBindingID);

  // Write a custom error message rather than relying on the tensor checker so
  // we can warn people about the padding requirement.
  if (zBufferHeight != imageHeight || zBufferWidth != paddedWidth) {
    throw std::runtime_error(fmt::format(
        "Expected z buffer of size {} x {} but got {} x {}.  "
        "Note that the width must be padded out to the nearest {} to ensure fast SIMD code.  ",
        imageHeight,
        paddedWidth,
        zBufferHeight,
        zBufferWidth,
        momentum::rasterizer::kSimdPacketSize));
  }

  return {zBuffer, rgbBuffer, surfaceNormalsBuffer, vertexIndexBuffer, triangleIndexBuffer};
}

std::vector<momentum::rasterizer::Light> convertLightsToEyeSpace(
    std::optional<std::vector<momentum::rasterizer::Light>> lights,
    const momentum::rasterizer::Camera& camera) {
  if (!lights.has_value()) {
    return {};
  }

  const Eigen::Affine3f worldToEyeXF = camera.worldFromEye();
  std::vector<momentum::rasterizer::Light> result;
  result.reserve(lights->size());
  for (const auto& l : *lights) {
    result.push_back(momentum::rasterizer::transformLight(l, worldToEyeXF));
  }

  return result;
}

} // namespace

at::Tensor createZBuffer(const momentum::rasterizer::Camera& camera, float farClip) {
  return at::full(
      {camera.imageHeight(), momentum::rasterizer::padImageWidthForRasterizer(camera.imageWidth())},
      farClip,
      at::kFloat);
}

at::Tensor createRGBBuffer(
    const momentum::rasterizer::Camera& camera,
    std::optional<Eigen::Vector3f> bgColor) {
  auto result = at::zeros(
      {camera.imageHeight(),
       momentum::rasterizer::padImageWidthForRasterizer(camera.imageWidth()),
       3},
      at::kFloat);

  if (bgColor.has_value()) {
    result.select(-1, 0) = bgColor->x();
    result.select(-1, 1) = bgColor->y();
    result.select(-1, 2) = bgColor->z();
  }

  return result;
}

at::Tensor createIndexBuffer(const momentum::rasterizer::Camera& camera) {
  return at::full(
      {camera.imageHeight(), momentum::rasterizer::padImageWidthForRasterizer(camera.imageWidth())},
      -1,
      at::kInt);
}

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

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel.
  for (size_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    Eigen::VectorXf emptyTextureCoords;
    Eigen::VectorXf emptyNormals;
    Eigen::VectorXi emptyTextureTriangles;
    Eigen::VectorXf emptyPerVertexDiffuseColor;
    momentum::rasterizer::rasterizeMesh(
        toEigenMap<float>(positions.select(0, iBatch)),
        normals.has_value() ? toEigenMap<float>(normals->select(0, iBatch)) : emptyNormals,
        toEigenMap<int>(triangles.select(0, iBatch)),
        textureCoordinates.has_value() ? toEigenMap<float>(textureCoordinates->select(0, iBatch))
                                       : emptyTextureCoords,
        textureTriangles.has_value() ? toEigenMap<int32_t>(textureTriangles->select(0, iBatch))
                                     : emptyTextureTriangles,
        perVertexDiffuseColor.has_value()
            ? toEigenMap<float>(perVertexDiffuseColor->select(0, iBatch))
            : emptyPerVertexDiffuseColor,
        camera,
        modelMatrix.value_or(Eigen::Matrix4f::Identity()),
        nearClip,
        material.value_or(momentum::rasterizer::PhongMaterial{}),
        make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
        make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
        make_mdspan<float, 3>(maybeSelect(surfaceNormalsBuffer, iBatch)),
        make_mdspan<int, 2>(maybeSelect(vertexIndexBuffer, iBatch)),
        make_mdspan<int, 2>(maybeSelect(triangleIndexBuffer, iBatch)),
        lights_eye,
        backfaceCulling,
        depthOffset,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
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
  for (size_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    Eigen::VectorXf emptyTextureCoords;
    Eigen::VectorXi emptyTextureTriangles;
    momentum::rasterizer::rasterizeWireframe(
        toEigenMap<float>(positions.select(0, iBatch)),
        toEigenMap<int>(triangles.select(0, iBatch)),
        camera,
        modelMatrix.value_or(Eigen::Matrix4f::Identity()),
        nearClip,
        color.value_or(Eigen::Vector3f{1, 1, 1}),
        thickness,
        make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
        make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
        backfaceCulling,
        depthOffset,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
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
  for (size_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    auto zBufferCur = zBuffer.select(0, iBatch);
    auto rgbBufferCur = maybeSelect(rgbBuffer, iBatch);
    auto centerCur = center.select(0, iBatch);
    auto radiiCur = maybeSelect(radius, iBatch);
    auto colorsCur = maybeSelect(color, iBatch);

    for (int i = 0; i < nSpheres; ++i) {
      const float radiusVal = radiiCur ? toEigenMap<float>(*radiiCur)(i) : 1.0f;
      auto materialCur = material.value_or(momentum::rasterizer::PhongMaterial());
      if (colorsCur) {
        materialCur.diffuseColor = toEigenMap<float>(*colorsCur).segment<3>(3 * i);
      }

      Eigen::Affine3f transform = Eigen::Affine3f::Identity();
      transform.translate(toEigenMap<float>(centerCur).segment<3>(3 * i));
      transform.scale(radiusVal);

      momentum::rasterizer::rasterizeMesh(
          sphereMesh,
          camera,
          modelMatrix * transform.matrix(),
          nearClip,
          materialCur,
          make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
          make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
          make_mdspan<float, 3>(maybeSelect(surfaceNormalsBuffer, iBatch)),
          {},
          {},
          lights_eye,
          backfaceCulling,
          depthOffset,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }
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
  for (size_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    auto zBufferCur = zBuffer.select(0, iBatch);
    auto rgbBufferCur = maybeSelect(rgbBuffer, iBatch);
    auto startPosCur = start_position.select(0, iBatch);
    auto endPosCur = end_position.select(0, iBatch);
    auto radiusCur = maybeSelect(radius, iBatch);
    auto colorCur = maybeSelect(color, iBatch);

    for (int i = 0; i < nCylinders; ++i) {
      const Eigen::Vector3f startPos = toEigenMap<float>(startPosCur).segment<3>(3 * i);
      const Eigen::Vector3f endPos = toEigenMap<float>(endPosCur).segment<3>(3 * i);
      const float radiusVal = radiusCur ? toEigenMap<float>(*radiusCur)(i) : 1.0f;
      auto materialCur = material.value_or(momentum::rasterizer::PhongMaterial());
      if (colorCur) {
        materialCur.diffuseColor = toEigenMap<float>(*colorCur).segment<3>(3 * i);
      }

      const float length = (endPos - startPos).norm();
      Eigen::Affine3f transform = Eigen::Affine3f::Identity();
      transform.translate(startPos);
      transform.rotate(Eigen::Quaternionf::FromTwoVectors(
          Eigen::Vector3f::UnitX(), (endPos - startPos).normalized()));
      transform.scale(Eigen::Vector3f(length, radiusVal, radiusVal));

      momentum::rasterizer::rasterizeMesh(
          cylinderMesh,
          camera,
          modelMatrix * transform.matrix(),
          nearClip,
          materialCur,
          make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
          make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
          make_mdspan<float, 3>(maybeSelect(surfaceNormalsBuffer, iBatch)),
          {},
          {},
          lights_eye,
          backfaceCulling,
          depthOffset,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }
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
  for (size_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    auto zBufferCur = zBuffer.select(0, iBatch);
    auto rgbBufferCur = maybeSelect(rgbBuffer, iBatch);
    auto transformationCur = transformation.select(0, iBatch);
    auto radiusCur = radius.select(0, iBatch);
    auto lengthCur = length.select(0, iBatch);

    for (int i = 0; i < nCapsules; ++i) {
      const Eigen::Matrix4f transform =
          toEigenMap<float>(transformationCur).segment<16>(16 * i).reshaped(4, 4).transpose();

      const Eigen::Vector2f radiusVal = toEigenMap<float>(radiusCur).segment<2>(2 * i);
      const float lengthVal = toEigenMap<float>(lengthCur)(i);

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
          make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
          make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
          make_mdspan<float, 3>(maybeSelect(surfaceNormalsBuffer, iBatch)),
          {},
          {},
          lights_eye,
          backfaceCulling,
          depthOffset,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }
  }
}

void rasterizeLines(
    at::Tensor positions,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    float width,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const int nLinesBindingId = -1;

  TensorChecker checker("rasterize");
  positions = checker.validateAndFixTensor(
      positions,
      "positions",
      {nLinesBindingId, 2, 3},
      {"nLines", "start_end", "xyz"},
      at::kFloat,
      true,
      false);

  std::tie(zBuffer, rgbBuffer, std::ignore, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, {}, {}, {});

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel
  for (size_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    auto zBufferCur = zBuffer.select(0, iBatch);
    auto rgbBufferCur = maybeSelect(rgbBuffer, iBatch);
    auto positionsCur = positions.select(0, iBatch);

    momentum::rasterizer::rasterizeLines(
        toEigenMap<float>(positionsCur),
        camera,
        modelMatrix.value_or(Eigen::Matrix4f::Identity()),
        nearClip,
        color.value_or(Eigen::Vector3f::Ones()),
        width,
        make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
        make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
        depthOffset,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

void rasterizeCircles(
    at::Tensor positions,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    float lineThickness,
    float radius,
    const std::optional<Eigen::Vector3f>& lineColor,
    std::optional<Eigen::Vector3f> fillColor,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  const int nCirclesBindingId = -1;

  TensorChecker checker("rasterize");
  positions = checker.validateAndFixTensor(
      positions, "positions", {nCirclesBindingId, 3}, {"nCircles", "xyz"}, at::kFloat, true, false);

  // If no color is provided, use fill=white as default.
  if (!lineColor.has_value() && !fillColor.has_value()) {
    fillColor = Eigen::Vector3f::Ones();
  }

  std::tie(zBuffer, rgbBuffer, std::ignore, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, {}, {}, {});

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel
  for (size_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    auto positionsCur = positions.select(0, iBatch);

    momentum::rasterizer::rasterizeCircles(
        toEigenMap<float>(positionsCur),
        camera,
        modelMatrix.value_or(Eigen::Matrix4f::Identity()),
        nearClip,
        lineColor,
        fillColor,
        lineThickness,
        radius,
        make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
        make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
        depthOffset,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

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
      grandparent = character.skeleton.joints[grandparent].parent;
    }

    if (grandparent != momentum::kInvalidIndex) {
      jointConnections.emplace_back(iJoint, grandparent);
    }
  }

  momentum::rasterizer::Mesh jointMesh;
  std::vector<Eigen::Vector3f> jointLines;
  std::vector<Eigen::Vector3f> jointCircles;
  momentum::rasterizer::Mesh sphereMesh;

  if (style == SkeletonStyle::Octahedrons) {
    std::tie(jointMesh, jointLines) = momentum::rasterizer::makeOctahedron(0.15, 0.1);
    sphereMesh = momentum::rasterizer::makeSphere(sphereSubdivisionLevel);

  } else if (style == SkeletonStyle::Lines) {
    for (size_t i = 0; i < cylinderLengthSubdivisions; ++i) {
      jointLines.emplace_back(i / float(cylinderLengthSubdivisions), 0, 0);
      jointLines.emplace_back((i + 1) / float(cylinderLengthSubdivisions), 0, 0);
    }
    jointCircles.emplace_back(0, 0, 0);
  } else {
    sphereMesh = momentum::rasterizer::makeSphere(sphereSubdivisionLevel);
    jointMesh =
        momentum::rasterizer::makeCylinder(cylinderRadiusSubdivisions, cylinderLengthSubdivisions);
  }

  const float lineThickness =
      (style == SkeletonStyle::Lines) ? std::max(1.0f, 2.0f * cylinderRadius) : 1.0f;
  const float circleRadius = std::max(1.0f, sphereRadius);

  const Eigen::Vector3f lineColor = (style == SkeletonStyle::Lines)
      ? cylinderMaterial.value_or(momentum::rasterizer::PhongMaterial{}).diffuseColor
      : Eigen::Vector3f::Constant(0.3f);
  const Eigen::Vector3f circleColor =
      jointMaterial.value_or(momentum::rasterizer::PhongMaterial{}).diffuseColor;

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel
  for (size_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    auto zBufferCur = zBuffer.select(0, iBatch);
    auto rgbBufferCur = maybeSelect(rgbBuffer, iBatch);
    auto skelStateCur = skeletonState.select(0, iBatch);

    for (size_t jJoint = 0; jJoint < nJoints; ++jJoint) {
      if (!activeJoints[jJoint]) {
        continue;
      }

      Eigen::Affine3f transform = Eigen::Affine3f::Identity();
      transform.translate(toEigenMap<float>(skelStateCur).segment<3>(8 * jJoint));
      transform.scale(sphereRadius);

      if (!jointCircles.empty()) {
        momentum::rasterizer::rasterizeCircles(
            jointCircles,
            camera,
            modelMatrix * transform.matrix(),
            nearClip,
            lineColor,
            circleColor,
            0.5f * lineThickness,
            circleRadius,
            make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
            make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
            depthOffset,
            imageOffset.value_or(Eigen::Vector2f::Zero()));
      }

      if (!sphereMesh.vertices.empty()) {
        momentum::rasterizer::rasterizeMesh(
            sphereMesh,
            camera,
            modelMatrix * transform.matrix(),
            nearClip,
            jointMaterial.value_or(momentum::rasterizer::PhongMaterial{}),
            make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
            make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
            make_mdspan<float, 3>(maybeSelect(surfaceNormalsBuffer, iBatch)),
            {},
            {},
            lights_eye,
            backfaceCulling,
            depthOffset,
            imageOffset.value_or(Eigen::Vector2f::Zero()));
      }
    }

    const auto skelStateMap = toEigenMap<float>(skelStateCur);

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

      if (!jointMesh.vertices.empty()) {
        momentum::rasterizer::rasterizeMesh(
            jointMesh,
            camera,
            modelMatrix * transform.matrix(),
            nearClip,
            cylinderMaterial.value_or(momentum::rasterizer::PhongMaterial{}),
            make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
            make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
            make_mdspan<float, 3>(maybeSelect(surfaceNormalsBuffer, iBatch)),
            {},
            {},
            lights_eye,
            backfaceCulling,
            depthOffset,
            imageOffset.value_or(Eigen::Vector2f::Zero()));
      }

      if (!jointLines.empty()) {
        momentum::rasterizer::rasterizeLines(
            jointLines,
            camera,
            modelMatrix * transform.matrix(),
            nearClip,
            lineColor,
            lineThickness,
            make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
            make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
            depthOffset - 0.001f,
            imageOffset.value_or(Eigen::Vector2f::Zero()));
      }
    }
  }
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
  for (size_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    auto zBufferCur = zBuffer.select(0, iBatch);
    auto rgbBufferCur = maybeSelect(rgbBuffer, iBatch);
    auto skelStateCur = skeletonState.select(0, iBatch);

    std::vector<momentum::JointState> jointStates(character.skeleton.joints.size());
    const auto skelStateMap = toEigenMap<float>(skelStateCur);
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
        perVertexDiffuseColor.has_value()
            ? toEigenMap<float>(perVertexDiffuseColor->select(0, iBatch))
            : Eigen::VectorXf{},
        camera,
        modelMatrix,
        nearClip,
        material.value_or(momentum::rasterizer::PhongMaterial{}),
        make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
        make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
        make_mdspan<float, 3>(maybeSelect(surfaceNormalsBuffer, iBatch)),
        make_mdspan<int, 2>(maybeSelect(vertexIndexBuffer, iBatch)),
        make_mdspan<int, 2>(maybeSelect(triangleIndexBuffer, iBatch)),
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
          make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
          make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
          backfaceCulling,
          depthOffset - 0.001,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }
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
  for (size_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    for (int k = 0; k < 2; ++k) {
      momentum::rasterizer::rasterizeMesh(
          meshes[k],
          camera,
          modelMatrix.value_or(Eigen::Matrix4f::Identity()),
          nearClip,
          materials[k],
          make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
          make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
          make_mdspan<float, 3>(maybeSelect(surfaceNormalsBuffer, iBatch)),
          {},
          {},
          lights_eye,
          backfaceCulling,
          depthOffset,
          imageOffset.value_or(Eigen::Vector2f::Zero()));
    }
  }
}

momentum::rasterizer::Tensor<float, 3> toFloatImage(const pybind11::array_t<const float>& img) {
  if (img.ndim() != 3 || img.shape(2) != 3) {
    throw std::runtime_error("Expected tensor shape height x width x 3 for RGB image.");
  }

  momentum::rasterizer::Tensor<float, 3> result(
      {static_cast<std::ptrdiff_t>(img.shape(0)),
       static_cast<std::ptrdiff_t>(img.shape(1)),
       static_cast<std::ptrdiff_t>(img.shape(2))});
  auto r = img.unchecked<3>(); // x must have ndim = 3; can be non-writable
  for (py::ssize_t i = 0; i < r.shape(0); i++) {
    for (py::ssize_t j = 0; j < r.shape(1); j++) {
      for (py::ssize_t k = 0; k < r.shape(2); k++) {
        result(i, j, k) = r(i, j, k);
      }
    }
  }
  return result;
}

momentum::rasterizer::PhongMaterial createPhongMaterial(
    const std::optional<Eigen::Vector3f>& diffuseColor,
    const std::optional<Eigen::Vector3f>& specularColor,
    const std::optional<float>& specularExponent,
    const std::optional<Eigen::Vector3f>& emissiveColor,
    const std::optional<pybind11::array_t<const float>>& diffuseTexture,
    const std::optional<pybind11::array_t<const float>>& emissiveTexture) {
  momentum::rasterizer::PhongMaterial result;
  result.diffuseColor = diffuseColor.value_or(Eigen::Vector3f::Ones());
  result.specularColor = specularColor.value_or(Eigen::Vector3f::Zero());
  result.specularExponent = specularExponent.value_or(10.0f);
  result.emissiveColor = emissiveColor.value_or(Eigen::Vector3f::Zero());

  if (diffuseTexture) {
    result.diffuseTextureMap = toFloatImage(diffuseTexture.value());
  }
  if (emissiveTexture) {
    result.emissiveTextureMap = toFloatImage(emissiveTexture.value());
  }

  return result;
}

void alphaMatte(
    at::Tensor zBuffer,
    at::Tensor rgbBuffer,
    const pybind11::array& tgtRgbImage,
    float alpha) {
  TensorChecker checker("alpha_matte");

  const int widthBindingID = -2003;
  const int heightBindingID = -2004;

  zBuffer = checker.validateAndFixTensor(
      zBuffer,
      "zBuffer",
      {heightBindingID, widthBindingID},
      {"imageHeight", "imageWidth"},
      at::kFloat,
      true,
      false);

  rgbBuffer = checker.validateAndFixTensor(
      rgbBuffer,
      "rgbBuffer",
      {heightBindingID, widthBindingID, 3},
      {"imageHeight", "imageWidth", "rgb"},
      at::kFloat,
      true,
      false);
  for (size_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    const auto tgtRgbImage_info = tgtRgbImage.request();
    if (tgtRgbImage_info.format == py::format_descriptor<float>::format()) {
      momentum::rasterizer::alphaMatte(
          make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
          make_mdspan<float, 3>(rgbBuffer.select(0, iBatch)),
          make_mdspan<float, 3>(tgtRgbImage_info),
          alpha);
    } else if (tgtRgbImage_info.format == py::format_descriptor<uint8_t>::format()) {
      momentum::rasterizer::alphaMatte(
          make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
          make_mdspan<float, 3>(rgbBuffer.select(0, iBatch)),
          make_mdspan<uint8_t, 3>(tgtRgbImage_info),
          alpha);
    } else {
      throw std::runtime_error("tgt_rgb_image must be a uint8 or float32 RGB array.");
    }
  }
}

void rasterizeCameraFrustum(
    const momentum::rasterizer::Camera& frustumCamera,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    float lineWidth,
    float distance,
    size_t numSamples,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  if (nearClip <= 0) {
    throw std::runtime_error("near_clip should be positive.");
  }

  if (lineWidth <= 0) {
    throw std::runtime_error("line_width should be positive.");
  }

  if (numSamples <= 2) {
    throw std::runtime_error("num_samples should be at least 3.");
  }

  if (distance <= 0) {
    throw std::runtime_error("distance should be positive.");
  }

  TensorChecker checker("rasterize_camera_frustum");

  std::tie(zBuffer, rgbBuffer, std::ignore, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, {}, {}, {});

  const std::vector<Eigen::Vector3f> frustumLinesEye =
      momentum::rasterizer::makeCameraFrustumLines(frustumCamera, distance, numSamples);

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel
  for (int64_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    momentum::rasterizer::rasterizeLines(
        frustumLinesEye,
        camera,
        modelMatrix.value_or(Eigen::Matrix4f::Identity()) * frustumCamera.worldFromEye().matrix(),
        nearClip,
        color.value_or(Eigen::Vector3f::Ones()),
        lineWidth,
        make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
        make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
        depthOffset,
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

void rasterizeLines2D(
    at::Tensor positions,
    at::Tensor rgbBuffer,
    float thickness,
    const std::optional<Eigen::Vector3f>& color,
    std::optional<at::Tensor> zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  const int nLinesBindingId = -1;
  const int heightBindingId = -2003;
  const int widthBindingId = -2004;

  TensorChecker checker("rasterize_lines_2d");
  positions = checker.validateAndFixTensor(
      positions,
      "positions",
      {nLinesBindingId, 2, 2},
      {"nLines", "start_end", "xy"},
      at::kFloat,
      true,
      false);

  rgbBuffer = checker.validateAndFixTensor(
      rgbBuffer,
      "rgb_buffer",
      {heightBindingId, widthBindingId, 3},
      {"height", "width", "rgb"},
      at::kFloat,
      true,
      false);

  if (zBuffer.has_value()) {
    zBuffer = checker.validateAndFixTensor(
        *zBuffer,
        "z_buffer",
        {heightBindingId, widthBindingId},
        {"height", "width"},
        at::kFloat,
        true,
        false);
  }

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel
  for (size_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    auto rgbBufferCur = rgbBuffer.select(0, iBatch);
    auto positionsCur = positions.select(0, iBatch);
    auto zBufferCur =
        zBuffer.has_value() ? maybeSelect(zBuffer, iBatch) : std::optional<at::Tensor>{};

    momentum::rasterizer::rasterizeLines2D(
        toEigenMap<float>(positionsCur),
        color.value_or(Eigen::Vector3f::Ones()),
        thickness,
        make_mdspan<float, 3>(rgbBufferCur),
        make_mdspan<float, 2>(zBufferCur),
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

void rasterizeCircles2D(
    at::Tensor positions,
    at::Tensor rgbBuffer,
    float lineThickness,
    float radius,
    const std::optional<Eigen::Vector3f>& lineColor,
    std::optional<Eigen::Vector3f> fillColor,
    std::optional<at::Tensor> zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset) {
  drjit::scoped_flush_denormals flushDenorm(true);
  pybind11::gil_scoped_release release;

  const int nCirclesBindingId = -1;
  const int heightBindingId = -2003;
  const int widthBindingId = -2004;

  TensorChecker checker("rasterize_circles_2d");
  positions = checker.validateAndFixTensor(
      positions, "positions", {nCirclesBindingId, 2}, {"nCircles", "xy"}, at::kFloat, true, false);

  rgbBuffer = checker.validateAndFixTensor(
      rgbBuffer,
      "rgb_buffer",
      {heightBindingId, widthBindingId, 3},
      {"height", "width", "rgb"},
      at::kFloat,
      true,
      false);

  // If no color is provided, use fill=white as default.
  if (!lineColor.has_value() && !fillColor.has_value()) {
    fillColor = Eigen::Vector3f::Ones();
  }

  if (zBuffer.has_value()) {
    zBuffer = checker.validateAndFixTensor(
        *zBuffer,
        "z_buffer",
        {heightBindingId, widthBindingId},
        {"height", "width"},
        at::kFloat,
        true,
        false);
  }

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel
  for (size_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    auto rgbBufferCur = rgbBuffer.select(0, iBatch);
    auto positionsCur = positions.select(0, iBatch);
    auto zBufferCur =
        zBuffer.has_value() ? maybeSelect(zBuffer, iBatch) : std::optional<at::Tensor>{};

    momentum::rasterizer::rasterizeCircles2D(
        toEigenMap<float>(positionsCur),
        lineColor,
        fillColor,
        lineThickness,
        radius,
        make_mdspan<float, 3>(rgbBufferCur),
        make_mdspan<float, 2>(zBufferCur),
        imageOffset.value_or(Eigen::Vector2f::Zero()));
  }
}

void rasterizeTransforms(
    at::Tensor transforms,
    const momentum::rasterizer::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    std::optional<at::Tensor> surfaceNormalsBuffer,
    float scale,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<std::vector<momentum::rasterizer::Light>> lights_world,
    const std::optional<Eigen::Matrix4f>& modelMatrix_in,
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

  const int nTransformsBindingId = -1;

  TensorChecker checker("rasterize");

  transforms = checker.validateAndFixTensor(
      transforms,
      "transforms",
      {nTransformsBindingId, 4, 4},
      {"nTransforms", "rows", "cols"},
      at::kFloat,
      true,
      false);

  const float c1 = 1.0f;
  const float c2 = 0.3f;
  const bool backfaceCulling = true;

  momentum::rasterizer::PhongMaterial materials[3] = {
      material.value_or(momentum::rasterizer::PhongMaterial(Eigen::Vector3f(c1, c2, c2))),
      material.value_or(momentum::rasterizer::PhongMaterial(Eigen::Vector3f(c2, c1, c2))),
      material.value_or(momentum::rasterizer::PhongMaterial(Eigen::Vector3f(c2, c2, c1)))};

  std::tie(zBuffer, rgbBuffer, surfaceNormalsBuffer, std::ignore, std::ignore) =
      validateRasterizerBuffers(checker, camera, zBuffer, rgbBuffer, surfaceNormalsBuffer, {}, {});

  const auto nTransforms = checker.getBoundValue(nTransformsBindingId);

  const auto arrowMesh = momentum::rasterizer::makeArrow(
      radiusSubdivisions, lengthSubdivisions, 0.05f, 0.2f, 0.4f, 0.6f);

  const Eigen::Matrix4f modelMatrix = modelMatrix_in.value_or(Eigen::Matrix4f::Identity());

  // We don't expect batched to be the common case, so don't try to process
  // the batches in parallel
  for (int64_t iBatch = 0; iBatch < checker.getBatchSize(); ++iBatch) {
    auto zBufferCur = zBuffer.select(0, iBatch);
    auto rgbBufferCur = maybeSelect(rgbBuffer, iBatch);

    auto transformsCur = transforms.select(0, iBatch);
    auto a = transformsCur.accessor<float, 3>();

    for (int i = 0; i < nTransforms; ++i) {
      const Eigen::Vector3f origin(a[i][0][3], a[i][1][3], a[i][2][3]);
      for (int j = 0; j < 3; ++j) {
        const Eigen::Vector3f col_j(a[i][0][j], a[i][1][j], a[i][2][j]);
        const float length = col_j.norm();
        if (length == 0) {
          continue;
        }

        const auto& materialCur = materials[j];

        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.translate(origin);
        transform.rotate(
            Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), col_j.normalized()));
        transform.scale(scale * Eigen::Vector3f(length, length, length));

        momentum::rasterizer::rasterizeMesh(
            arrowMesh,
            camera,
            modelMatrix * transform.matrix(),
            nearClip,
            materialCur,
            make_mdspan<float, 2>(zBuffer.select(0, iBatch)),
            make_mdspan<float, 3>(maybeSelect(rgbBuffer, iBatch)),
            make_mdspan<float, 3>(maybeSelect(surfaceNormalsBuffer, iBatch)),
            {},
            {},
            lights_eye,
            backfaceCulling,
            depthOffset,
            imageOffset.value_or(Eigen::Vector2f::Zero()));
      }
    }
  }
}

} // namespace pymomentum
