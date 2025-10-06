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

template <typename T>
void alphaMatte(Span2f zBuffer, Span3f rgbBuffer, const Span<T, 3>& tgtImage, float alpha = 1.0f);

} // namespace momentum::rasterizer
