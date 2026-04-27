/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/texture_classification.h"

#include <momentum/character/texture_classification.h>
#include <momentum/common/exception.h>

namespace pymomentum {

namespace {

/// Validate a texture numpy array and return a TextureView.
momentum::TextureView validateTexture(const py::array_t<uint8_t>& texture) {
  py::buffer_info texInfo = texture.request();
  MT_THROW_IF(texInfo.ndim != 3, "texture must be a 3D array [height, width, 3]");
  MT_THROW_IF(texInfo.shape[2] != 3, "texture must have 3 channels (RGB)");
  MT_THROW_IF(
      texInfo.shape[0] <= 0 || texInfo.shape[1] <= 0, "texture dimensions must be positive");

  return momentum::TextureView{
      static_cast<const uint8_t*>(texInfo.ptr),
      static_cast<int64_t>(texInfo.shape[0]),
      static_cast<int64_t>(texInfo.shape[1]),
      static_cast<int64_t>(texInfo.strides[0]),
      static_cast<int64_t>(texInfo.strides[1]),
      static_cast<int64_t>(texInfo.strides[2])};
}

/// Validate a region colors numpy array and return a RegionColorsView.
momentum::RegionColorsView validateRegionColors(const py::array_t<uint8_t>& regionColors) {
  py::buffer_info colorsInfo = regionColors.request();
  MT_THROW_IF(colorsInfo.ndim != 2, "region_colors must be a 2D array [n_regions, 3]");
  MT_THROW_IF(colorsInfo.shape[1] != 3, "region_colors must have 3 columns (RGB)");

  return momentum::RegionColorsView{
      static_cast<const uint8_t*>(colorsInfo.ptr),
      static_cast<int32_t>(colorsInfo.shape[0]),
      static_cast<int64_t>(colorsInfo.strides[0]),
      static_cast<int64_t>(colorsInfo.strides[1])};
}

} // namespace

std::vector<std::vector<int32_t>> classifyTrianglesByTexture(
    const momentum::Mesh& mesh,
    const py::array_t<uint8_t>& texture,
    const py::array_t<uint8_t>& regionColors,
    float threshold,
    int32_t numSamples) {
  const auto texView = validateTexture(texture);
  const auto colorsView = validateRegionColors(regionColors);

  std::vector<std::vector<int32_t>> result;
  {
    py::gil_scoped_release release;
    result = momentum::classifyTrianglesByTexture(mesh, texView, colorsView, threshold, numSamples);
  }
  return result;
}

momentum::Mesh splitMeshByTextureRegion(
    const momentum::Mesh& mesh,
    const py::array_t<uint8_t>& texture,
    const py::array_t<uint8_t>& regionColors,
    int32_t numBinarySearchSteps) {
  const auto texView = validateTexture(texture);
  const auto colorsView = validateRegionColors(regionColors);

  return momentum::splitMeshByTextureRegion(mesh, texView, colorsView, numBinarySearchSteps);
}

} // namespace pymomentum
