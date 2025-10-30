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
#include <momentum/rasterizer/geometry.h>
#include <momentum/rasterizer/tensor.h>
#include <Eigen/Geometry>
#include <optional>
#include <span>

namespace momentum::rasterizer {

using index_t = std::ptrdiff_t;

/// mdspan type aliases for cleaner signatures
template <typename T, size_t Rank>
using Span = Kokkos::mdspan<T, Kokkos::dextents<index_t, Rank>>;

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

  /// Constructor with default Phong material values
  ///
  /// @param diffuseColor Base diffuse color (default: white)
  /// @param specularColor Specular highlight color (default: black)
  /// @param specularExponent Specular sharpness (default: 10.0)
  /// @param emissiveColor Self-illumination color (default: black)
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
  /// Constructor for creating a light with specified properties
  ///
  /// @param position Light position in world/eye space
  /// @param color RGB color intensity of the light
  /// @param type Type of light (Point, Directional, or Ambient)
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

/// Pad image width to ensure proper SIMD alignment
///
/// @param width Original image width
/// @return Padded width (multiple of 8 for SIMD support)
index_t padImageWidthForRasterizer(index_t width);

/// Rasterize a mesh to depth/RGB buffer using Phong lighting model
///
/// This function renders a 3D mesh with realistic lighting using the Phong shading model.
/// It supports texture mapping, multi-pass rendering, and various output buffers for
/// advanced rendering techniques.
///
/// @param positions_world Vertex positions in world space (flat array of floats)
/// @param normals_world Vertex normals in world space (flat array of floats)
/// @param triangles Triangle indices into vertex arrays
/// @param textureCoords Texture coordinates for vertices
/// @param textureTriangles Array of triangles in texture space. Should have the same size as the
///        triangles array but contain indices into the textureCoords array. Supports texture
///        vertices being different from mesh vertices so you can have discontinuities in the
///        texture map. If textureTriangles is not provided, the regular triangles array will be
///        used in its place.
/// @param perVertexDiffuseColor Per-vertex diffuse color modulation
/// @param camera Camera to render from
/// @param modelMatrix Additional transform to apply to the model. Unlike the camera extrinsics it
/// is
///        allowed to use non-uniform scale and shear.
/// @param nearClip Near clipping value: triangles closer than this are not rendered.
/// @param material Phong material to use when rendering.
/// @param zBuffer Input/output depth buffer. If you want to render multiple objects in a scene, you
///        can reuse the same depth buffer. Must be padded out to a multiple of 8 for proper SIMD
///        support (makeRasterizerZBuffer does this automatically).
/// @param rgbBuffer Input/output RGB buffer. Has the same requirements as the depth buffer.
/// @param surfaceNormalsBuffer Input/output surface normal buffer. Writes the eye-space surface
/// normal
///        as (x,y,z) triplet for each pixel.
/// @param vertexIndexBuffer Input/output buffer of vertex indices; writes the index of the closest
///        vertex in the triangle for every rendered pixel (values where the depth buffer is set).
/// @param triangleIndexBuffer Writes the index of the closest triangle for every rendered pixel.
/// @param lights_eye Lights in eye coordinates. If not provided, uses a default lighting setup
///        with a single light colocated with the camera.
/// @param backfaceCulling Enable back-face culling; speeds up the render but means back-facing
/// surfaces
///        will not appear.
/// @param depthOffset Offset the depth; useful for e.g. rendering the skeleton slightly in front of
/// the
///        mesh.
/// @param imageOffset Offset within the image by (delta_x, delta_y) pixels. Useful for rendering
///        slightly off to the side of the background or another mesh so you can compare the two
///        without needing to construct a special camera.
///
/// For use with Torch tensors, takes the positions/normals/triangles as a flat array of floats
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

/// Rasterize 3D line segments with specified thickness and color
///
/// This function projects 3D line segments to image space using camera transformation
/// and rasterizes them using the actual computed line depth for depth testing.
/// Lines are not anti-aliased; for smoother rendering consider using supersampling.
///
/// @param positions_world 3D line vertex positions in world space (consecutive pairs form line
/// segments) @param camera Camera for 3D projection @param modelMatrix Additional model
/// transformation matrix @param nearClip Near clipping distance @param color RGB color for all line
/// segments @param thickness Line thickness in pixels @param zBuffer **Required** input/output
/// depth buffer (SIMD-aligned) @param rgbBuffer Optional input/output RGB color buffer @param
/// depthOffset Depth offset for layered rendering @param imageOffset Pixel offset for comparative
/// rendering
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

/// Rasterize 3D circles (possibly filled) projected to screen space
///
/// This function projects 3D circle centers to image space using camera transformation
/// and rasterizes them using the actual computed circle depth for depth testing.
/// Circles can be rendered as outlines only, filled only, or both. Circles are not
/// anti-aliased; for smoother rendering consider using supersampling.
///
/// @param positions_world 3D circle center positions in world space
/// @param camera Camera for 3D projection
/// @param modelMatrix Additional model transformation matrix
/// @param nearClip Near clipping distance
/// @param lineColor Optional outline color (if not set, no outline is drawn)
/// @param fillColor Optional fill color (if not set, circles are not filled)
/// @param lineThickness Outline thickness in pixels
/// @param radius Circle radius in world units
/// @param zBuffer **Required** input/output depth buffer (SIMD-aligned)
/// @param rgbBuffer Optional input/output RGB color buffer
/// @param depthOffset Depth offset for layered rendering
/// @param imageOffset Pixel offset for comparative rendering
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

/// Rasterize oriented circular splats with normal-based orientation
///
/// A "splat" is an oriented circle centered at the provided position and oriented
/// orthogonal to the normal. This is particularly useful for rasterizing point clouds
/// like those constructed from depth maps, providing a surface-like appearance.
///
/// @param positions_world 3D splat center positions in world space
/// @param normals_world 3D normal vectors defining splat orientation
/// @param camera Camera for 3D projection
/// @param modelMatrix Additional model transformation matrix
/// @param nearClip Near clipping distance
/// @param frontMaterial Phong material for front-facing splats
/// @param backMaterial Phong material for back-facing splats
/// @param radius Splat radius in world units
/// @param zBuffer Input/output depth buffer (SIMD-aligned)
/// @param rgbBuffer Optional input/output RGB color buffer
/// @param lights_eye Lights in eye coordinates
/// @param depthOffset Depth offset for layered rendering
/// @param imageOffset Pixel offset for comparative rendering
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

/// Rasterize 2D line segments directly in image space
///
/// This function renders line segments using 2D image coordinates directly, without 3D
/// projection. When a depth buffer is provided, it fills the buffer with zeros, effectively
/// placing the lines within the image plane. Useful for UI overlays or 2D graphics.
/// Lines are not anti-aliased; for smoother rendering consider using supersampling.
///
/// @param positions_image 2D line vertex positions in image coordinates (consecutive pairs form
/// line segments) @param color RGB color for all line segments @param thickness Line thickness in
/// pixels @param rgbBuffer Input/output RGB color buffer @param zBuffer **Optional** depth buffer
/// (fills with zeros when provided) @param imageOffset Pixel offset for positioning
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

/// Rasterize 2D circles directly in image space
///
/// This function renders circles using 2D image coordinates directly, without 3D
/// projection. When a depth buffer is provided, it fills the buffer with zeros, effectively
/// placing the circles within the image plane. Circles can be rendered as outlines only,
/// filled only, or both. Circles are not anti-aliased; for smoother rendering consider
/// using supersampling.
///
/// @param positions_image 2D circle center positions in image coordinates
/// @param lineColor Optional outline color (if not set, no outline is drawn)
/// @param fillColor Optional fill color (if not set, circles are not filled)
/// @param lineThickness Outline thickness in pixels
/// @param radius Circle radius in pixels
/// @param rgbBuffer Input/output RGB color buffer
/// @param zBuffer **Optional** depth buffer (fills with zeros when provided)
/// @param imageOffset Pixel offset for positioning
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
