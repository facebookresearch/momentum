/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/rasterizer/fwd.h>
#include <momentum/rasterizer/rasterizer.h>

namespace momentum::rasterizer {

/// Blend rendered image onto target with alpha matting and downsampling
/// @param zBuffer Depth buffer from rendering (supersampled)
/// @param rgbBuffer RGB buffer from rendering (supersampled)
/// @param tgtImage Target image to blend into (downsampled)
/// @param alpha Global alpha multiplier for blending (0-1)
/// @pre zBuffer and rgbBuffer must have dimensions that are integer multiples of tgtImage
template <typename T>
void alphaMatte(Span2f zBuffer, Span3f rgbBuffer, Span<T, 3> tgtImage, float alpha = 1.0f);

} // namespace momentum::rasterizer
