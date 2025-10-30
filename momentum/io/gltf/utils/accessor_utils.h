/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "momentum/common/exception.h"

#include <fx/gltf.h>
#include <gsl/span_ext>

namespace momentum {

static inline int32_t getComponentSize(const fx::gltf::Accessor::ComponentType& componentType) {
  switch (componentType) {
    case fx::gltf::Accessor::ComponentType::Byte:
    case fx::gltf::Accessor::ComponentType::UnsignedByte:
      return 1;
    case fx::gltf::Accessor::ComponentType::Short:
    case fx::gltf::Accessor::ComponentType::UnsignedShort:
      return 2;
    // case fx::gltf::Accessor::ComponentType::Int:
    case fx::gltf::Accessor::ComponentType::UnsignedInt:
    case fx::gltf::Accessor::ComponentType::Float:
      return 4;
    default:
      return -1;
  }
}

static inline int32_t getTypeSize(const fx::gltf::Accessor::Type& ty) {
  switch (ty) {
    case fx::gltf::Accessor::Type::Scalar:
      return 1;
    case fx::gltf::Accessor::Type::Vec2:
      return 2;
    case fx::gltf::Accessor::Type::Vec3:
      return 3;
    case fx::gltf::Accessor::Type::Vec4:
    case fx::gltf::Accessor::Type::Mat2:
      return 4;
    case fx::gltf::Accessor::Type::Mat3:
      return 9;
    case fx::gltf::Accessor::Type::Mat4:
      return 16;
    default:
      return -1;
  }
}

template <typename T>
std::vector<T> copyAccessorBuffer(const fx::gltf::Document& model, int32_t id) {
  const auto& accessor = model.accessors[id];

  const auto tsize = getTypeSize(accessor.type);
  const auto csize = getComponentSize(accessor.componentType);
  const auto elsize = tsize * csize;

  if (elsize != sizeof(T)) {
    MT_LOGW(
        "[{}] Accessor type mismatch. Expect to read {} bytes, but got type of {} bytes with {} components",
        __func__,
        sizeof(T),
        csize,
        tsize);
    return {};
  }

  const auto& view = model.bufferViews[accessor.bufferView];
  const auto stride = view.byteStride == 0 ? elsize : view.byteStride;

  const auto& buf = model.buffers[view.buffer];
  auto bytes = std::as_bytes(std::span<const uint8_t>(buf.data));
  bytes = bytes.subspan(view.byteOffset, view.byteLength);
  bytes = bytes.subspan(accessor.byteOffset);

  std::vector<T> r(accessor.count);
  for (size_t i = 0; i < accessor.count; i++) {
    std::memcpy(&r[i], &bytes[i * stride], elsize);
  }

  return r;
}

template <typename T>
std::vector<T> copyAlignedAccessorBuffer(const fx::gltf::Document& model, int32_t id) {
  const auto& accessor = model.accessors[id];
  if (accessor.count == 0) {
    MT_LOGW("Accessor count zero.");
    return {};
  }

  const auto tsize = getTypeSize(accessor.type);
  const auto csize = getComponentSize(accessor.componentType);
  const auto elsize = tsize * csize;

  if (elsize != sizeof(T)) {
    MT_LOGW(
        "[{}] Accessor type mismatch. Expect to read {} bytes, but got type of {} bytes with {} components",
        __func__,
        sizeof(T),
        csize,
        tsize);
    return {};
  }

  const auto& view = model.bufferViews[accessor.bufferView];
  const auto stride = view.byteStride == 0 ? elsize : view.byteStride;

  const auto& buf = model.buffers[view.buffer];
  auto bytes = std::as_bytes(std::span<const uint8_t>(buf.data));
  bytes = bytes.subspan(view.byteOffset, view.byteLength);
  bytes = bytes.subspan(accessor.byteOffset);

  std::vector<T> r(accessor.count);
  for (size_t i = 0; i < accessor.count; i++) {
    std::memcpy(&r[i], &bytes[i * stride], elsize);
  }

  return r;
}

// template to set accessor
template <typename T>
void setAccessorType(fx::gltf::Accessor& /* accessor */) {
  MT_THROW("Unsupported data type {}", typeid(T).name());
}

// create accessor buffer
template <typename T>
int32_t createAccessorBuffer(
    fx::gltf::Document& model,
    std::span<T> data,
    const bool align = false,
    const bool normalized = false) {
  // gltf 2.0 allows multiple buffers, but only the first buffer will be stored in
  // the binary file, all other ones need to be external files. Because of this we will
  // only ever store data in the first buffer

  // check if we have a buffer
  if (model.buffers.empty()) {
    model.buffers.resize(1);
  }

  // create buffer for data
  const size_t bufferIdx = 0;
  auto& buffer = model.buffers.front();

  // copy data from input buffer to output buffer
  const size_t bufferDataStart = buffer.data.size();
  const auto elementSize = sizeof(T);
  auto alignedElementSize = elementSize;
  auto dataSize = data.size() * alignedElementSize;

  // check for necessary alignment
  if (align && alignedElementSize % 4 != 0) {
    // element is not aligned, need to get bigger element size
    alignedElementSize = ((elementSize / 4) + 1) * 4;
    dataSize = data.size() * alignedElementSize;
    buffer.data.resize(bufferDataStart + dataSize);
    // need to go over each element one by one to copy
    for (size_t e = 0; e < data.size(); e++) {
      std::memcpy(&buffer.data[bufferDataStart + alignedElementSize * e], &data[e], elementSize);
    }
    buffer.byteLength = buffer.data.size();
  } else {
    // no alignment needed or already aligned, just copy data in
    buffer.data.resize(bufferDataStart + dataSize);
    std::memcpy(&buffer.data[bufferDataStart], data.data(), dataSize);
    buffer.byteLength = buffer.data.size();
  }

  // create bufferview
  const size_t bufferViewIdx = model.bufferViews.size();
  model.bufferViews.emplace_back();
  auto& bufferView = model.bufferViews.back();

  bufferView.buffer = bufferIdx;
  bufferView.byteLength = dataSize;
  bufferView.byteOffset = bufferDataStart;
  if (align && elementSize % 4 != 0) {
    bufferView.byteStride = alignedElementSize;
  } else {
    bufferView.byteStride = 0;
  }

  // create accessor
  const size_t accessorIdx = model.accessors.size();
  model.accessors.emplace_back();
  auto& accessor = model.accessors.back();

  // set component type and accessor size depending on the type
  setAccessorType<T>(accessor);
  accessor.bufferView = bufferViewIdx;
  accessor.byteOffset = 0;
  accessor.count = data.size();
  accessor.normalized = normalized;

  return accessorIdx;
}

template <>
inline void setAccessorType<const int32_t>(fx::gltf::Accessor& accessor) {
  accessor.componentType = fx::gltf::Accessor::ComponentType::UnsignedInt;
  accessor.type = fx::gltf::Accessor::Type::Scalar;
}

template <>
inline void setAccessorType<const float>(fx::gltf::Accessor& accessor) {
  accessor.componentType = fx::gltf::Accessor::ComponentType::Float;
  accessor.type = fx::gltf::Accessor::Type::Scalar;
}

template <>
inline void setAccessorType<const Vector2f>(fx::gltf::Accessor& accessor) {
  accessor.componentType = fx::gltf::Accessor::ComponentType::Float;
  accessor.type = fx::gltf::Accessor::Type::Vec2;
}

template <>
inline void setAccessorType<const Vector3f>(fx::gltf::Accessor& accessor) {
  accessor.componentType = fx::gltf::Accessor::ComponentType::Float;
  accessor.type = fx::gltf::Accessor::Type::Vec3;
}

template <>
inline void setAccessorType<const Vector4f>(fx::gltf::Accessor& accessor) {
  accessor.componentType = fx::gltf::Accessor::ComponentType::Float;
  accessor.type = fx::gltf::Accessor::Type::Vec4;
}

template <>
inline void setAccessorType<const Vector3b>(fx::gltf::Accessor& accessor) {
  accessor.componentType = fx::gltf::Accessor::ComponentType::UnsignedByte;
  accessor.type = fx::gltf::Accessor::Type::Vec3;
}

template <>
inline void setAccessorType<const Vector3i>(fx::gltf::Accessor& accessor) {
  accessor.componentType = fx::gltf::Accessor::ComponentType::UnsignedInt;
  accessor.type = fx::gltf::Accessor::Type::Vec3;
}

template <>
inline void setAccessorType<const Vector4s>(fx::gltf::Accessor& accessor) {
  accessor.componentType = fx::gltf::Accessor::ComponentType::UnsignedShort;
  accessor.type = fx::gltf::Accessor::Type::Vec4;
}

template <>
inline void setAccessorType<const Matrix4f>(fx::gltf::Accessor& accessor) {
  accessor.componentType = fx::gltf::Accessor::ComponentType::Float;
  accessor.type = fx::gltf::Accessor::Type::Mat4;
}

// add animation
template <typename T>
int32_t createSampler(
    fx::gltf::Document& model,
    fx::gltf::Animation& anim,
    std::span<T> data,
    const std::span<float>& timestamps) {
  MT_CHECK(
      data.size() == timestamps.size(), "data: {}, timestmaps: {}", data.size(), timestamps.size());

  // create new sampler
  const int32_t index = anim.samplers.size();
  anim.samplers.emplace_back();
  auto& sampler = anim.samplers.back();
  sampler.interpolation = fx::gltf::Animation::Sampler::Type::Step;
  sampler.input = createAccessorBuffer(model, timestamps);
  sampler.output = createAccessorBuffer(model, data);

  // return sampler index
  return index;
}

template <typename T>
int32_t createSampler(
    fx::gltf::Document& model,
    fx::gltf::Animation& anim,
    std::span<T> data,
    const int32_t timestampAccessor) {
  // create new sampler
  const int32_t index = anim.samplers.size();
  anim.samplers.emplace_back();
  auto& sampler = anim.samplers.back();
  sampler.interpolation = fx::gltf::Animation::Sampler::Type::Step;
  sampler.input = timestampAccessor;
  sampler.output = createAccessorBuffer(model, data);

  // return sampler index
  return index;
}

} // namespace momentum
