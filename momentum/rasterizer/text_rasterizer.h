/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/camera/camera.h>
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

/// Rasterize text at 3D world positions with depth testing
/// @param texts Text strings to render at each position
/// @param textScale Integer scaling factor (1 = 1 pixel per font pixel)
/// @note Uses embedded bitmap font for rendering
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

/// Rasterize text at 2D image positions (no camera projection or depth testing)
/// @param textScale Integer scaling factor (1 = 1 pixel per font pixel)
/// @param zBuffer Optional depth buffer (fills with zeros when provided)
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
