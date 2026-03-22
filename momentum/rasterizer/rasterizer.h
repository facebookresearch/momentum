/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <mdspan/mdspan.hpp>
#include <momentum/camera/camera.h>
#include <momentum/common/aligned.h>
#include <momentum/rasterizer/fwd.h>
#include <momentum/rasterizer/geometry.h>
#include <momentum/rasterizer/tensor.h>
#include <Eigen/Geometry>
#include <optional>
#include <span>

namespace momentum::rasterizer {

using index_t = std::ptrdiff_t;

/// mdspan type aliases for cleaner signatures
/// Using layout_stride to support strided buffers (e.g., with SIMD padding)
template <typename T, size_t Rank>
using Span = Kokkos::mdspan<T, Kokkos::dextents<index_t, Rank>, Kokkos::layout_stride>;

/// Constant variant of Span for read-only access
template <typename T, size_t Rank>
using ConstSpan = Span<const T, Rank>;

/// Dynamic extent type for mdspan
template <size_t Rank>
using Extents = Kokkos::dextents<index_t, Rank>;

/// 2D span of floats
using Span2f = Span<float, 2>;
/// 3D span of floats
using Span3f = Span<float, 3>;
/// 2D span of 32-bit integers
using Span2i = Span<int32_t, 2>;

/// Phong material definition for realistic lighting calculations
///
/// This structure defines the material properties used in Phong shading,
/// including diffuse, specular, and emissive components, as well as optional
/// texture maps for enhanced visual fidelity.
struct PhongMaterial {
  /// Diffuse color component (base color under diffuse lighting)
  Eigen::Vector3f diffuseColor;
  /// Specular color component (color of specular highlights)
  Eigen::Vector3f specularColor;
  /// Specular exponent controlling the sharpness of specular highlights
  float specularExponent;
  /// Emissive color component (self-illumination)
  Eigen::Vector3f emissiveColor;

  /// Optional diffuse texture map (RGB channels)
  Tensor<float, 3> diffuseTextureMap;
  /// Optional emissive texture map (RGB channels)
  Tensor<float, 3> emissiveTextureMap;

  /// Check if material has any texture maps
  ///
  /// @return true if either diffuse or emissive texture maps are present
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

/// Types of lights supported by the rasterizer
enum class LightType { Point, Directional, Ambient };

/// Basic light structure for rendering calculations
///
/// Represents a light source with position, color, and type information
/// for use in Phong shading calculations.
struct Light {
  Light() = default;
  Light(const Eigen::Vector3f& position, const Eigen::Vector3f& color, LightType type)
      : position(position), color(color), type(type) {}

  /// Light position (for Point lights) or direction (for Directional lights)
  Eigen::Vector3f position{0, 0, 0};
  /// RGB color intensity of the light
  Eigen::Vector3f color{1, 1, 1};
  /// Type of light source
  LightType type = LightType::Point;
};

/// Create an ambient light source
///
/// @param color RGB color intensity (default: white)
/// @return Configured ambient light
Light createAmbientLight(const Eigen::Vector3f& color = Eigen::Vector3f::Ones());

/// Create a directional light source (like sunlight)
///
/// @param dir Direction vector of the light
/// @param color RGB color intensity (default: white)
/// @return Configured directional light
Light createDirectionalLight(
    const Eigen::Vector3f& dir,
    const Eigen::Vector3f& color = Eigen::Vector3f::Ones());

/// Create a point light source
///
/// @param pos Position of the light in world space
/// @param color RGB color intensity (default: white)
/// @return Configured point light
Light createPointLight(
    const Eigen::Vector3f& pos,
    const Eigen::Vector3f& color = Eigen::Vector3f::Ones());

/// Transform a light by the given transformation matrix
///
/// @param light Light to transform
/// @param xf Affine transformation matrix
/// @return Transformed light
Light transformLight(const Light& light, const Eigen::Affine3f& xf);

/// Pad image width to ensure SIMD alignment
/// @param width Original image width in pixels
/// @return Padded width (multiple of kSimdPacketSize = 8)
/// @note Required for proper SIMD vectorization in rasterizer operations
index_t padImageWidthForRasterizer(index_t width);

/// Rasterize a mesh with Phong lighting and optional texture mapping
/// @param positions_world Vertex positions in world space (flat: x1,y1,z1,x2,y2,z2,...)
/// @param normals_world Vertex normals in world space (flat array)
/// @param triangles Triangle vertex indices (flat: v1,v2,v3,v4,v5,v6,...)
/// @param textureCoords Texture UV coordinates (flat: u1,v1,u2,v2,...)
/// @param textureTriangles Texture triangle indices (can differ from mesh triangles for
/// discontinuities)
/// @param perVertexDiffuseColor Per-vertex color modulation (empty or 3*numVertices)
/// @param modelMatrix Model transform (supports non-uniform scale and shear unlike camera
/// extrinsics)
/// @param nearClip Near clipping distance (triangles closer are culled)
/// @param zBuffer Input/output depth buffer (must be SIMD-aligned, use makeRasterizerZBuffer)
/// @param rgbBuffer Optional RGB output buffer
/// @param surfaceNormalsBuffer Optional eye-space surface normals output (x,y,z per pixel)
/// @param vertexIndexBuffer Optional closest vertex index per pixel
/// @param triangleIndexBuffer Optional triangle index per pixel
/// @param lights_eye Lights in eye space (default: single light at camera)
/// @param depthOffset Depth bias for layered rendering
/// @param imageOffset Pixel offset for comparative rendering
/// @post All output buffers are updated where depth test passes
/// @note Multi-pass rendering: reuse zBuffer/rgbBuffer to composite multiple objects
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

/// Rasterize 3D line segments with depth testing
/// @param positions_world Line endpoints in world space (flat: x1,y1,z1,x2,y2,z2,...)
/// @param thickness Line thickness in pixels
/// @param zBuffer **Required** depth buffer (SIMD-aligned)
/// @note Lines use actual computed depth for depth testing (not screen-space depth)
/// @note Not anti-aliased; consider supersampling for smoother rendering
void rasterizeLines(
    std::span<const Eigen::Vector3f> positions_world,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    float thickness,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0});

/// Rasterize 3D line segments (flat array version)
///
/// @see rasterizeLines(std::span<const Eigen::Vector3f>&, const Camera&, const Eigen::Matrix4f&,
/// float, const Eigen::Vector3f&, float, Span2f, Span3f, float, const Eigen::Vector2f&)
///
/// @param positions_world Flat array of 3D positions (x1,y1,z1,x2,y2,z2,...)
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

/// Rasterize 3D circles projected to screen space
/// @param lineColor Optional outline color (nullopt = no outline)
/// @param fillColor Optional fill color (nullopt = no fill)
/// @param radius Circle radius in world units
/// @note Uses actual circle depth for depth testing
/// @note Not anti-aliased; consider supersampling
void rasterizeCircles(
    std::span<const Eigen::Vector3f> positions_world,
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

/// Rasterize 3D circles (flat array version)
///
/// @see rasterizeCircles(std::span<const Eigen::Vector3f>, const Camera&, const Eigen::Matrix4f&,
/// float, const std::optional<Eigen::Vector3f>&, const std::optional<Eigen::Vector3f>&, float,
/// float, Span2f, Span3f, float, const Eigen::Vector2f&)
///
/// @param positions_world Flat array of 3D positions (x1,y1,z1,x2,y2,z2,...)
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

/// Rasterize a mesh using Eigen vector containers (uint32 triangles)
///
/// @see rasterizeMesh(const Eigen::Ref<const Eigen::VectorXf>&, const Eigen::Ref<const
/// Eigen::VectorXf>&, const Eigen::Ref<const Eigen::VectorXi>&, const Eigen::Ref<const
/// Eigen::VectorXf>&, const Eigen::Ref<const Eigen::VectorXi>&, const Eigen::Ref<const
/// Eigen::VectorXf>&, const Camera&, const Eigen::Matrix4f&, float, const PhongMaterial&, Span2f,
/// Span3f, Span3f, Span2i, Span2i, const std::vector<Light>&, bool, float, const Eigen::Vector2f&)
///
/// @param positions_world Vector of 3D vertex positions in world space
/// @param normals_world Vector of 3D vertex normals in world space
/// @param triangles Vector of triangles with uint32 indices
void rasterizeMesh(
    std::span<const Eigen::Vector3f> positions_world,
    std::span<const Eigen::Vector3f> normals_world,
    std::span<const Eigen::Matrix<uint32_t, 3, 1>> triangles,
    std::span<const Eigen::Vector2f> textureCoords,
    std::span<const Eigen::Matrix<uint32_t, 3, 1>> textureTriangles,
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

/// Rasterize a mesh using a Mesh object
///
/// @see rasterizeMesh(const Eigen::Ref<const Eigen::VectorXf>&, const Eigen::Ref<const
/// Eigen::VectorXf>&, const Eigen::Ref<const Eigen::VectorXi>&, const Eigen::Ref<const
/// Eigen::VectorXf>&, const Eigen::Ref<const Eigen::VectorXi>&, const Eigen::Ref<const
/// Eigen::VectorXf>&, const Camera&, const Eigen::Matrix4f&, float, const PhongMaterial&, Span2f,
/// Span3f, Span3f, Span2i, Span2i, const std::vector<Light>&, bool, float, const Eigen::Vector2f&)
///
/// @param mesh Mesh object containing all geometry data
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

/// Rasterize a mesh using Eigen vector containers (int32 triangles)
///
/// @see rasterizeMesh(const Eigen::Ref<const Eigen::VectorXf>&, const Eigen::Ref<const
/// Eigen::VectorXf>&, const Eigen::Ref<const Eigen::VectorXi>&, const Eigen::Ref<const
/// Eigen::VectorXf>&, const Eigen::Ref<const Eigen::VectorXi>&, const Eigen::Ref<const
/// Eigen::VectorXf>&, const Camera&, const Eigen::Matrix4f&, float, const PhongMaterial&, Span2f,
/// Span3f, Span3f, Span2i, Span2i, const std::vector<Light>&, bool, float, const Eigen::Vector2f&)
///
/// @param positions_world Vector of 3D vertex positions in world space
/// @param normals_world Vector of 3D vertex normals in world space
/// @param triangles Vector of triangles with int32 indices
void rasterizeMesh(
    std::span<const Eigen::Vector3f> positions_world,
    std::span<const Eigen::Vector3f> normals_world,
    std::span<const Eigen::Vector3i> triangles,
    std::span<const Eigen::Vector2f> textureCoords,
    std::span<const Eigen::Vector3i> textureTriangles,
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

/// Rasterize mesh wireframe with outlined edges
///
/// This function renders only the edges of triangles, useful for debugging geometry
/// or creating wireframe visualizations. Lines are not anti-aliased; for smoother
/// wireframes consider using supersampling.
///
/// @param positions_world Vector of 3D vertex positions in world space
/// @param triangles Vector of triangles defining mesh connectivity
/// @param camera Camera for 3D projection
/// @param modelMatrix Additional model transformation matrix
/// @param nearClip Near clipping distance
/// @param color RGB color for all wireframe edges
/// @param thickness Line thickness in pixels
/// @param zBuffer Input/output depth buffer (SIMD-aligned)
/// @param rgbBuffer Optional input/output RGB color buffer
/// @param backfaceCulling Enable back-face culling
/// @param depthOffset Depth offset for layered rendering
/// @param imageOffset Pixel offset for comparative rendering
void rasterizeWireframe(
    std::span<const Eigen::Vector3f> positions_world,
    std::span<const Eigen::Vector3i> triangles,
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

/// Rasterize mesh wireframe (flat array version)
///
/// @see rasterizeWireframe(std::span<const Eigen::Vector3f>, std::span<const Eigen::Vector3i>,
/// const Camera&, const Eigen::Matrix4f&, float, const Eigen::Vector3f&, float, Span2f, Span3f,
/// bool, float, const Eigen::Vector2f&)
///
/// @param positions_world Flat array of 3D positions (x1,y1,z1,x2,y2,z2,...)
/// @param triangles Flat array of triangle indices
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

/// Rasterize oriented circular splats (surface-oriented circles)
/// @param normals_world Normal vectors defining splat orientation (orthogonal to circle plane)
/// @param frontMaterial Material for front-facing splats
/// @param backMaterial Material for back-facing splats
/// @param radius Splat radius in world units
/// @note Useful for rendering point clouds with surface-like appearance
void rasterizeSplats(
    std::span<const Eigen::Vector3f> positions_world,
    std::span<const Eigen::Vector3f> normals_world,
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

/// Rasterize 2D line segments directly in image space (no 3D projection)
/// @param positions_image Line endpoints in image coordinates (x1,y1,x2,y2,...)
/// @param zBuffer Optional depth buffer (fills with zeros when provided)
/// @note When zBuffer provided, lines are placed at depth=0 (in image plane)
void rasterizeLines2D(
    std::span<const Eigen::Vector2f> positions_image,
    const Eigen::Vector3f& color,
    float thickness,
    Span3f rgbBuffer,
    Span2f zBuffer = {},
    const Eigen::Vector2f& imageOffset = {0, 0});

/// Rasterize 2D line segments (flat array version)
///
/// @see rasterizeLines2D(std::span<const Eigen::Vector2f>, const Eigen::Vector3f&, float, Span3f,
/// Span2f, const Eigen::Vector2f&)
///
/// @param positions_image Flat array of 2D positions (x1,y1,x2,y2,...)
void rasterizeLines2D(
    const Eigen::Ref<const Eigen::VectorXf>& positions_image,
    const Eigen::Vector3f& color,
    float thickness,
    Span3f rgbBuffer,
    Span2f zBuffer = {},
    const Eigen::Vector2f& imageOffset = {0, 0});

/// Rasterize 2D circles directly in image space (no 3D projection)
/// @param radius Circle radius in pixels (not world units)
/// @param zBuffer Optional depth buffer (fills with zeros when provided)
void rasterizeCircles2D(
    std::span<const Eigen::Vector2f> positions_image,
    const std::optional<Eigen::Vector3f>& lineColor,
    const std::optional<Eigen::Vector3f>& fillColor,
    float lineThickness,
    float radius,
    Span3f rgbBuffer,
    Span2f zBuffer = {},
    const Eigen::Vector2f& imageOffset = {0, 0});

/// Rasterize 2D circles (flat array version)
///
/// @see rasterizeCircles2D(std::span<const Eigen::Vector2f>, const std::optional<Eigen::Vector3f>&,
/// const std::optional<Eigen::Vector3f>&, float, float, Span3f, Span2f, const Eigen::Vector2f&)
///
/// @param positions_image Flat array of 2D positions (x1,y1,x2,y2,...)
void rasterizeCircles2D(
    const Eigen::Ref<const Eigen::VectorXf>& positions_image,
    const std::optional<Eigen::Vector3f>& lineColor,
    const std::optional<Eigen::Vector3f>& fillColor,
    float lineThickness,
    float radius,
    Span3f rgbBuffer,
    Span2f zBuffer = {},
    const Eigen::Vector2f& imageOffset = {0, 0});

/// Create a properly sized and aligned depth buffer for rasterization
///
/// @param camera Camera configuration defining image dimensions
/// @return SIMD-aligned depth buffer tensor initialized to infinity
Tensor2f makeRasterizerZBuffer(const Camera& camera);

/// Create a properly sized RGB color buffer for rasterization
///
/// @param camera Camera configuration defining image dimensions
/// @return RGB color buffer tensor initialized to black
Tensor3f makeRasterizerRGBBuffer(const Camera& camera);

/// Create a properly sized index buffer for rasterization
///
/// @param camera Camera configuration defining image dimensions
/// @return Index buffer tensor initialized to -1 (invalid indices)
Tensor2i makeRasterizerIndexBuffer(const Camera& camera);

} // namespace momentum::rasterizer
