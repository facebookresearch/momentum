/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/rasterizer/camera.h>
#include <momentum/rasterizer/fwd.h>
#include <momentum/rasterizer/rasterizer.h>
#include <Eigen/Core>
#include <gsl/span>
#include <string>

namespace momentum::rasterizer {

/// Horizontal alignment options for text rendering
enum class HorizontalAlignment {
  Left,
  Center,
  Right,
};

/// Vertical alignment options for text rendering
enum class VerticalAlignment {
  Top,
  Center,
  Bottom,
};

/// Rasterize text at 3D world positions
///
/// Projects 3D positions to image space using the camera and renders text strings at those
/// locations. Uses an embedded bitmap font for rendering.
///
/// @param positionsWorld 3D positions in world coordinates where text should be rendered
/// @param texts Text strings to render at each position
/// @param camera Camera to render from
/// @param modelMatrix Model transformation matrix
/// @param nearClip Near clipping distance
/// @param color RGB color for the text
/// @param textScale Integer scaling factor for text size (1 = 1 pixel per font pixel)
/// @param zBuffer Input/output depth buffer (SIMD-aligned)
/// @param rgbBuffer Optional input/output RGB color buffer
/// @param imageOffset Pixel offset for positioning
/// @param horizontalAlignment Horizontal text alignment relative to position
/// @param verticalAlignment Vertical text alignment relative to position
void rasterizeText(
    gsl::span<const Eigen::Vector3f> positionsWorld,
    gsl::span<const std::string> texts,
    const Camera& camera,
    const Eigen::Matrix4f& modelMatrix,
    float nearClip,
    const Eigen::Vector3f& color,
    int textScale,
    Span2f zBuffer,
    Span3f rgbBuffer = {},
    float depthOffset = 0,
    const Eigen::Vector2f& imageOffset = {0, 0},
    HorizontalAlignment horizontalAlignment = HorizontalAlignment::Left,
    VerticalAlignment verticalAlignment = VerticalAlignment::Top);

/// Rasterize text directly in 2D image space
///
/// Renders text at 2D image positions without camera projection or depth testing.
///
/// @param positionsImage 2D positions in image coordinates where text should be rendered
/// @param texts Text strings to render at each position
/// @param color RGB color for the text
/// @param textScale Integer scaling factor for text size (1 = 1 pixel per font pixel)
/// @param rgbBuffer Input/output RGB color buffer
/// @param zBuffer Optional depth buffer (fills with zeros when provided)
/// @param imageOffset Pixel offset for positioning
/// @param horizontalAlignment Horizontal text alignment relative to position
/// @param verticalAlignment Vertical text alignment relative to position
void rasterizeText2D(
    gsl::span<const Eigen::Vector2f> positionsImage,
    gsl::span<const std::string> texts,
    const Eigen::Vector3f& color,
    int textScale,
    Span3f rgbBuffer,
    Span2f zBuffer = {},
    const Eigen::Vector2f& imageOffset = {0, 0},
    HorizontalAlignment horizontalAlignment = HorizontalAlignment::Left,
    VerticalAlignment verticalAlignment = VerticalAlignment::Top);

} // namespace momentum::rasterizer
