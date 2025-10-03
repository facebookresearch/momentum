/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <drjit/fwd.h>
#include <mdspan/mdspan.hpp>
#include <momentum/common/aligned.h>
#include <momentum/rasterizer/camera.h>
#include <momentum/rasterizer/fwd.h>
#include <momentum/rasterizer/tensor.h>
#include <Eigen/Geometry>
#include <gsl/span>
#include <optional>

namespace momentum::rasterizer {

using index_t = std::ptrdiff_t;

// mdspan type aliases for cleaner signatures
template <typename T, size_t Rank>
using Span = Kokkos::mdspan<T, Kokkos::dextents<index_t, Rank>>;

template <typename T, size_t Rank>
using ConstSpan = Span<const T, Rank>;

template <size_t Rank>
using Extents = Kokkos::dextents<index_t, Rank>;

using Span2f = Span<float, 2>;
using Span3f = Span<float, 3>;
using Span2i = Span<int32_t, 2>;

struct Mesh;

struct PhongMaterial {
  Eigen::Vector3f diffuseColor;
  Eigen::Vector3f specularColor;
  float specularExponent;
  Eigen::Vector3f emissiveColor;

  Tensor<float, 3> diffuseTextureMap;
  Tensor<float, 3> emissiveTextureMap;

  [[nodiscard]] bool hasTextureMap() const {
    return !diffuseTextureMap.empty() || !emissiveTextureMap.empty();
  }

  PhongMaterial(
      const Eigen::Vector3f& diffuseColor = Eigen::Vector3f::Ones(),
      const Eigen::Vector3f& specularColor = Eigen::Vector3f::Zero(),
      float specularExponent = 10.0f,
      const Eigen::Vector3f& emissiveColor = Eigen::Vector3f::Zero())
      : diffuseColor(diffuseColor),
        specularColor(specularColor),
        specularExponent(specularExponent),
        emissiveColor(emissiveColor) {}
};

enum class LightType { Point, Directional, Ambient };

// Basic light for use in rendering.
struct Light {
  Light() = default;
  Light(const Eigen::Vector3f& position, const Eigen::Vector3f& color, LightType type)
      : position(position), color(color), type(type) {}

  Eigen::Vector3f position{0, 0, 0};
  Eigen::Vector3f color{1, 1, 1};
  LightType type = LightType::Point;
};

Light createAmbientLight(const Eigen::Vector3f& color = Eigen::Vector3f::Ones());
Light createDirectionalLight(
    const Eigen::Vector3f& dir,
    const Eigen::Vector3f& color = Eigen::Vector3f::Ones());
Light createPointLight(
    const Eigen::Vector3f& pos,
    const Eigen::Vector3f& color = Eigen::Vector3f::Ones());

Light transformLight(const Light& light, const Eigen::Affine3f& xf);

index_t padImageWidthForRasterizer(index_t width);

// Rasterize the mesh to the depth/RGB buffer using a Phong lighting model.
//   positions_world: Vertex positions in world space
//   normals_world: Vertex normals in world space
//   triangles: triangles
//   textureCoords: texture coordinates
//   textureTriangles: array of triangles in texture space.  Should have the same size as the
//        triangles array but contain indices into the textureCoords array.  Supports texture
//        vertices being different from mesh vertices so you can have discontinuities in the texture
//        map.  If textureTriangles is not provided, the regular triangles array will be used in its
//        place.
//   camera: Camera to render from
//   modelMatrix: Additional transform to apply to the model.  Unlike the camera extrinsics it is
//        allowed to use non-uniform scale and shear.
//   nearClip: Near clipping value: triangles closer than this are not rendered.
//   material: Phong material to use when rendering.
//   zBuffer: input/output depth buffer.  If you want to render multiple objects in a scene, you
//        can reuse the same depth buffer.  Must be padded out to a multiple of 8 for proper SIMD
//        support (makeRasterizerZBuffer does this automatically).
//   rgbBuffer: input/output RGB buffer.  Has the same requirements as the depth buffer.
//   surfaceNormalsBuffer: input/output surface normal buffer.  Writes the eye-space surface normal
//        as (x,y,z) triplet for each pixel.
//   vertexIndexBuffer: input/output buffer of vertex indices; writes the index of the closest
//        vertex in the triangle for every rendered pixel (values where the depth buffer is set).
//   triangleIndexBuffer: writes the index of the closest triangle for every rendered pixel.
//   lights_eye: Lights in eye coordinates.  If not provided, uses a default lighting setup
//        with a single light colocated with the camera.
//   backfaceCulling: Enable back-face culling; speeds up the render but means back-facing surfaces
//        will not appear.
//   depthOffset: Offset the depth; useful for e.g. rendering the skeleton slightly in front of the
//        mesh.
//   imageOffset: Offset within the image by (delta_x, delta_y) pixels.  Useful for rendering
//        slightly off to the side of the background or another mesh so you can compare the two
//        without needing to construct a special camera.
//
// For use with Torch tensors, takes the positions/normals/triangles as a flat array of floats:
void rasterizeMesh(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Eigen::Ref<const Eigen::VectorXf>& normals_world,
    const Eigen::Ref<const Eigen::VectorXi>& triangles,
    const Eigen::Ref<const Eigen::VectorXf>& textureCoords,
    const Eigen::Ref<const Eigen::VectorXi>& textureTriangles,
    const Eigen::Ref<const Eigen::VectorXf>& perVertexDiffuseColor,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const PhongMaterial& material,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    Span3f surfaceNormalsBuffer = {},
    Span2i vertexIndexBuffer = {},
    Span2i triangleIndexBuffer = {},
    const std::vector<Light>& lights_eye = {},
    bool backfaceCulling = true,
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0});

void rasterizeLines(
    gsl::span<const Eigen::Vector3f>& positions_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    float thickness,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0});

void rasterizeLines(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    float thickness,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0});

void rasterizeCircles(
    gsl::span<const Eigen::Vector3f> positions_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const std::optional<Eigen::Vector3f>& lineColor,
    const std::optional<Eigen::Vector3f>& fillColor,
    float lineThickness,
    float radius,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0});

void rasterizeCircles(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const std::optional<Eigen::Vector3f>& lineColor,
    const std::optional<Eigen::Vector3f>& fillColor,
    float lineThickness,
    float radius,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0});

// Takes the positions/normals/triangles as vectors of Eigen::Vector3f.
void rasterizeMesh(
    gsl::span<const Eigen::Vector3f> positions_world,
    gsl::span<const Eigen::Vector3f> normals_world,
    gsl::span<const Eigen::Matrix<uint32_t, 3, 1>> triangles,
    gsl::span<const Eigen::Vector2f> textureCoords,
    gsl::span<const Eigen::Matrix<uint32_t, 3, 1>> textureTriangles,
    const Eigen::Ref<const Eigen::VectorXf>& perVertexDiffuseColor,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const PhongMaterial& material,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    Span3f surfaceNormalsBuffer = {},
    Span2i vertexIndexBuffer = {},
    Span2i triangleIndexBuffer = {},
    const std::vector<Light>& lights_eye = {},
    bool backfaceCulling = true,
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0});

void rasterizeMesh(
    const Mesh& mesh,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const PhongMaterial& material,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    Span3f surfaceNormalsBuffer = {},
    Span2i vertexIndexBuffer = {},
    Span2i triangleIndexBuffer = {},
    const std::vector<Light>& lights_eye = {},
    bool backfaceCulling = true,
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0});

// Takes the positions/normals/triangles as vectors of Eigen::Vector3f.
void rasterizeMesh(
    gsl::span<const Eigen::Vector3f> positions_world,
    gsl::span<const Eigen::Vector3f> normals_world,
    gsl::span<const Eigen::Vector3i> triangles,
    gsl::span<const Eigen::Vector2f> textureCoords,
    gsl::span<const Eigen::Vector3i> textureTriangles,
    const Eigen::Ref<const Eigen::VectorXf>& perVertexDiffuseColor,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const PhongMaterial& material,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    Span3f surfaceNormalsBuffer = {},
    Span2i vertexIndexBuffer = {},
    Span2i triangleIndexBuffer = {},
    const std::vector<Light>& lights_eye = {},
    bool backfaceCulling = true,
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0});

void rasterizeWireframe(
    gsl::span<const Eigen::Vector3f> positions_world,
    gsl::span<const Eigen::Vector3i> triangles,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    float thickness,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    bool backfaceCulling = true,
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0});

void rasterizeWireframe(
    const Eigen::Ref<const Eigen::VectorXf>& positions_world,
    const Eigen::Ref<const Eigen::VectorXi>& triangles,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    float thickness,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    bool backfaceCulling = true,
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0});

void rasterizeLines(
    gsl::span<const Eigen::Vector3f> positions_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    float thickness,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0});

// A "splat" is an oriented circle centered at the provided
// position and oriented orthogonal to the normal.  It's a
// nice way to rasterize point clouds like those constructed
// from depth maps.
void rasterizeSplats(
    gsl::span<const Eigen::Vector3f> positions_world,
    gsl::span<const Eigen::Vector3f> normals_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const PhongMaterial& frontMaterial,
    const PhongMaterial& backMaterial,
    float radius,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    const std::vector<Light>& lights_eye = {},
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0});

// 2D rasterization functions that operate directly in image space
// without camera projection or z-buffer

// Rasterize lines in 2D image space
void rasterizeLines2D(
    gsl::span<const Eigen::Vector2f> positions_image,
    const Eigen::Vector3f& color,
    float thickness,
    Span3f rgbBuffer,
    Span2f zBuffer = {},
    const Eigen::Vector2f& imageOffset = {0, 0});

void rasterizeLines2D(
    const Eigen::Ref<const Eigen::VectorXf>& positions_image,
    const Eigen::Vector3f& color,
    float thickness,
    Span3f rgbBuffer,
    Span2f zBuffer = {},
    const Eigen::Vector2f& imageOffset = {0, 0});

// Rasterize circles in 2D image space
void rasterizeCircles2D(
    gsl::span<const Eigen::Vector2f> positions_image,
    const std::optional<Eigen::Vector3f>& lineColor,
    const std::optional<Eigen::Vector3f>& fillColor,
    float lineThickness,
    float radius,
    Span3f rgbBuffer,
    Span2f zBuffer = {},
    const Eigen::Vector2f& imageOffset = {0, 0});

void rasterizeCircles2D(
    const Eigen::Ref<const Eigen::VectorXf>& positions_image,
    const std::optional<Eigen::Vector3f>& lineColor,
    const std::optional<Eigen::Vector3f>& fillColor,
    float lineThickness,
    float radius,
    Span3f rgbBuffer,
    Span2f zBuffer = {},
    const Eigen::Vector2f& imageOffset = {0, 0});

Tensor2f makeRasterizerZBuffer(const Camera& camera);
Tensor3f makeRasterizerRGBBuffer(const Camera& camera);
Tensor2i makeRasterizerIndexBuffer(const Camera& camera);

} // namespace momentum::rasterizer
