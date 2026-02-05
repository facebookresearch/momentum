/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <drjit/array.h>
#include <drjit/util.h>
#include <momentum/common/aligned.h>
#include <momentum/rasterizer/image.h>
#include <momentum/rasterizer/rasterizer.h>
#include <momentum/rasterizer/utility.h>
#include <cfloat>

#include <momentum/common/exception.h>

namespace momentum::rasterizer {

template <typename TTgt, typename TSrc>
TTgt toRGBValue(const TSrc& rgb);

template <>
float toRGBValue<float, uint8_t>(const uint8_t& rgb) {
  return rgb / 255.0f;
}

template <>
float toRGBValue<float, float>(const float& rgb) {
  return rgb;
}

template <>
uint8_t toRGBValue<uint8_t, float>(const float& rgb) {
  return static_cast<uint8_t>(std::clamp(rgb * 255.0f, 0.0f, 255.0f));
}

template <>
uint8_t toRGBValue<uint8_t, uint8_t>(const uint8_t& rgb) {
  return rgb;
}

template <typename TTgt, typename TSrc>
drjit::Array<drjit::Packet<TTgt, kSimdPacketSize>, 3> toRGBPacket(const TSrc& rgb);

template <typename TTgt, typename TSrc>
drjit::Array<drjit::Packet<TTgt, kSimdPacketSize>, 3> toRGBPacket(
    const drjit::Array<drjit::Packet<TSrc, kSimdPacketSize>, 3>& rgb);

template <>
Vector3fP toRGBPacket<float, uint8_t>(const Vector3bP& rgb) {
  return Vector3fP(rgb) / 255.0f;
}

template <>
Vector3fP toRGBPacket<float, float>(const Vector3fP& rgb) {
  return rgb;
}

template <>
Vector3bP toRGBPacket<uint8_t, uint8_t>(const Vector3bP& rgb) {
  return rgb;
}

template <>
Vector3bP toRGBPacket<uint8_t, float>(const Vector3fP& rgb) {
  return {drjit::clip(rgb * 255.0f, 0.0f, 255.0f)};
}

template <typename T>
void alphaMatte(Span2f zBuffer, Span3f rgbBuffer, Span<T, 3> tgtImage, float alpha) {
  using Vector3TP = drjit::Array<drjit::Packet<T, kSimdPacketSize>, 3>;

  // Validate buffer layout and alignment for rasterization buffers
  validateRasterizerBuffer(zBuffer, "Z buffer");
  validateRasterizerBuffer(rgbBuffer, "RGB buffer");

  const auto imageHeightSupersampled = static_cast<int32_t>(zBuffer.extent(0));
  const auto paddedWidthSupersampled = static_cast<int32_t>(zBuffer.extent(1));
  if (zBuffer.extent(0) != imageHeightSupersampled || zBuffer.extent(1) > paddedWidthSupersampled) {
    throw std::runtime_error("Z buffer too large to support alphaMatte.");
  }

  const auto imageHeightDownsampled = static_cast<int32_t>(tgtImage.extent(0));
  const auto imageWidthDownsampled = static_cast<int32_t>(tgtImage.extent(1));
  if (tgtImage.extent(0) != imageHeightDownsampled || tgtImage.extent(1) != imageWidthDownsampled) {
    throw std::runtime_error("Target image too large to support alphaMatte.");
  }

  if (imageHeightDownsampled == 0 || imageWidthDownsampled == 0) {
    throw std::runtime_error("Can't alphaMatte with empty image.");
  }

  if (rgbBuffer.extent(0) != imageHeightSupersampled ||
      rgbBuffer.extent(1) != paddedWidthSupersampled || rgbBuffer.extent(2) != 3) {
    throw std::runtime_error(
        "Invalid RGB buffer " + formatTensorSizes(rgbBuffer.extents()) + "; expected " +
        formatTensorSizes(
            Kokkos::dextents<index_t, 3>{imageHeightSupersampled, paddedWidthSupersampled, 3}));
  }

  const int32_t downsampleAmount = imageHeightSupersampled / imageHeightDownsampled;
  if (imageHeightDownsampled * downsampleAmount != imageHeightSupersampled) {
    std::ostringstream oss;
    oss << "Expected rendered image height to be an integer multiple of the target image height; "
        << " rendered image has size " << formatTensorSizes(rgbBuffer.extents())
        << " but downsampled image has dimensions " << formatTensorSizes(tgtImage.extents());
    throw std::runtime_error(oss.str());
  }

  if (imageWidthDownsampled * downsampleAmount > paddedWidthSupersampled) {
    std::ostringstream oss;
    oss << "Rendered image width is too small based on detected downsample amount of "
        << downsampleAmount << "x; " << " rendered image has size "
        << formatTensorSizes(rgbBuffer.extents()) << " but downsampled image has dimensions "
        << formatTensorSizes(tgtImage.extents());
    throw std::runtime_error(oss.str());
  }

  if (tgtImage.extent(2) != 3) {
    throw std::runtime_error(
        "Invalid RGB image " + formatTensorSizes(tgtImage.extents()) +
        "; expected last dimension to be 3.");
  }

  const float* zBufferPtr = zBuffer.data_handle();
  const float* rgbBufferPtr = rgbBuffer.data_handle();

  if (uintptr_t(zBufferPtr) % kSimdAlignment != 0 ||
      uintptr_t(rgbBufferPtr) % kSimdAlignment != 0) {
    std::ostringstream oss;
    oss << "All buffers must be aligned on " << kSimdAlignment << " boundaries.";
    throw std::runtime_error(oss.str());
  }

  const float VERY_FAR = FLT_MAX / 2.0f;

  // Since we're accumulating downsample*downsample entries, need to scale by
  // 1/downsample*downsample:
  const float denom = 1.0f / (downsampleAmount * downsampleAmount);

  // Temporary buffer for accumulating one row in:
  std::vector<float, momentum::AlignedAllocator<float, kSimdAlignment>> tempBuffer(
      4 * paddedWidthSupersampled);

  for (index_t iTgtRow = 0; iTgtRow < imageHeightDownsampled; ++iTgtRow) {
    // First walk through the downsample rows vertically and sum into the temp buffer:
    for (uint32_t jCol = 0; jCol < paddedWidthSupersampled; jCol += kSimdPacketSize) {
      const auto colIndices = jCol + drjit::arange<IntP>();
      auto rgba = drjit::zeros<Vector4fP>();
      for (index_t jSrcOffset = 0; jSrcOffset < downsampleAmount; ++jSrcOffset) {
        const index_t jSrcRow = iTgtRow * downsampleAmount + jSrcOffset;
        const auto depthCur =
            drjit::gather<FloatP>(zBuffer.data_handle() + jSrcRow * zBuffer.stride(0), colIndices);
        const auto rgb = drjit::gather<Vector3fP>(
            rgbBuffer.data_handle() + jSrcRow * rgbBuffer.stride(0), colIndices);

        const FloatP alphaCur = drjit::select(depthCur > VERY_FAR, FloatP(0), FloatP(1));

        for (int i = 0; i < 3; ++i) {
          rgba[i] += rgb[i];
        }
        rgba.w() += alphaCur;
      }

      drjit::scatter(tempBuffer.data(), rgba, colIndices);
    }

    for (const auto [colIndices, colMask] : drjit::range<IntP>(imageWidthDownsampled)) {
      auto rgba = drjit::zeros<Vector4fP>();
      // Walk through the columns and sum horizontally
      for (int32_t offset = 0; offset < downsampleAmount; ++offset) {
        rgba += drjit::gather<Vector4fP>(
            tempBuffer.data(), (colIndices * downsampleAmount) + offset, colMask);
      }

      rgba = drjit::clip(rgba * denom, 0.0f, 1.0f);
      // apply global alpha:
      rgba.w() *= alpha;

      // Check if target image is contiguous for optimization
      const bool tgtContiguous = isContiguous(tgtImage);

      if (tgtContiguous) {
        // Store the blended result in the image:
        auto* tgtImagePtr = tgtImage.data_handle() + iTgtRow * tgtImage.stride(0);
        const Vector3fP tgtPixel =
            toRGBPacket<float, T>(drjit::gather<Vector3TP>(tgtImagePtr, colIndices, colMask));
        const Vector3fP blended = drjit::head<3>(rgba) + (1.0f - rgba.w()) * tgtPixel;
        drjit::scatter(tgtImagePtr, toRGBPacket<T, float>(blended), colIndices, colMask);
      } else {
        for (int k = 0; k < kSimdPacketSize; ++k) {
          const auto jTgtCol = colIndices[k];
          if (jTgtCol >= imageWidthDownsampled) {
            continue;
          }
          const float alphaCur = rgba.w()[k];
          const float invAlphaCur = 1.0f - alphaCur;
          for (int64_t l = 0; l < 3; ++l) {
            float blended =
                invAlphaCur * toRGBValue<float, T>(tgtImage(iTgtRow, jTgtCol, l)) + rgba[l][k];
            tgtImage(iTgtRow, jTgtCol, l) = toRGBValue<T, float>(blended);
          }
        }
      }
    }
  }
}

// Explicit template instantiations
template void
alphaMatte<float>(Span2f zBuffer, Span3f rgbBuffer, Span<float, 3> tgtImage, float alpha);

template void
alphaMatte<uint8_t>(Span2f zBuffer, Span3f rgbBuffer, Span<uint8_t, 3> tgtImage, float alpha);

} // namespace momentum::rasterizer
