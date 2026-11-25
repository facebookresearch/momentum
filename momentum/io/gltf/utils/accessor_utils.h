/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "momentum/common/exception.h"
#include "momentum/common/log.h"
#include "momentum/io/file_save_options.h"

#include <fx/gltf.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gsl/span_ext>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace momentum {

inline int32_t getComponentSize(const fx::gltf::Accessor::ComponentType& componentType) {
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

inline int32_t getTypeSize(const fx::gltf::Accessor::Type& ty) {
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

namespace detail {

/// The component type and the number of components of an element that copyAccessorBuffer()
/// reads as `T`.
template <typename T>
struct AccessorElement;

template <>
struct AccessorElement<float> {
  using Component = float;
  static constexpr int32_t kCount = 1;
};

template <>
struct AccessorElement<uint16_t> {
  using Component = uint16_t;
  static constexpr int32_t kCount = 1;
};

template <>
struct AccessorElement<uint32_t> {
  using Component = uint32_t;
  static constexpr int32_t kCount = 1;
};

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct AccessorElement<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> {
  static_assert(Rows > 0 && Cols > 0, "Only fixed-size matrices can be read from an accessor");
  using Component = Scalar;
  static constexpr int32_t kCount = Rows * Cols;
};

template <typename Scalar, int Options>
struct AccessorElement<Eigen::Quaternion<Scalar, Options>> {
  using Component = Scalar;
  static constexpr int32_t kCount = 4;
};

/// The value of type `Value` stored at `data`, which need not be aligned.
template <typename Value>
[[nodiscard]] Value readValue(const std::byte* data) {
  Value value;
  std::memcpy(&value, data, sizeof(Value));
  return value;
}

/// The value of the component at `data`, stored with `componentType`.
[[nodiscard]] inline double readComponent(
    const std::byte* data,
    fx::gltf::Accessor::ComponentType componentType) {
  switch (componentType) {
    case fx::gltf::Accessor::ComponentType::Byte:
      return readValue<int8_t>(data);
    case fx::gltf::Accessor::ComponentType::UnsignedByte:
      return readValue<uint8_t>(data);
    case fx::gltf::Accessor::ComponentType::Short:
      return readValue<int16_t>(data);
    case fx::gltf::Accessor::ComponentType::UnsignedShort:
      return readValue<uint16_t>(data);
    case fx::gltf::Accessor::ComponentType::UnsignedInt:
      return readValue<uint32_t>(data);
    case fx::gltf::Accessor::ComponentType::Float:
      return readValue<float>(data);
    case fx::gltf::Accessor::ComponentType::None:
      break;
  }
  MT_THROW("Invalid accessor component type {}", static_cast<int>(componentType));
}

/// The value in [-1, 1] or [0, 1] that the glTF format gives a normalized integer component.
[[nodiscard]] inline float normalizedComponent(
    float value,
    fx::gltf::Accessor::ComponentType componentType) {
  switch (componentType) {
    case fx::gltf::Accessor::ComponentType::Byte:
      return std::max(value / 127.0f, -1.0f);
    case fx::gltf::Accessor::ComponentType::UnsignedByte:
      return value / 255.0f;
    case fx::gltf::Accessor::ComponentType::Short:
      return std::max(value / 32767.0f, -1.0f);
    case fx::gltf::Accessor::ComponentType::UnsignedShort:
      return value / 65535.0f;
    case fx::gltf::Accessor::ComponentType::UnsignedInt:
      return value / 4294967295.0f;
    case fx::gltf::Accessor::ComponentType::Float:
    case fx::gltf::Accessor::ComponentType::None:
      break;
  }
  return value;
}

/// Whether `componentType` is an unsigned integer type: unsigned byte, short or int.
[[nodiscard]] inline bool isUnsignedInteger(fx::gltf::Accessor::ComponentType componentType) {
  return componentType == fx::gltf::Accessor::ComponentType::UnsignedByte ||
      componentType == fx::gltf::Accessor::ComponentType::UnsignedShort ||
      componentType == fx::gltf::Accessor::ComponentType::UnsignedInt;
}

/// Buffer view `viewIndex` of the document. Throws when the document has no such buffer view.
[[nodiscard]] inline const fx::gltf::BufferView& bufferViewAt(
    const fx::gltf::Document& model,
    uint32_t viewIndex) {
  MT_THROW_IF(
      viewIndex >= model.bufferViews.size(),
      "Buffer view {} does not exist; the file has {} buffer views",
      viewIndex,
      model.bufferViews.size());
  return model.bufferViews[viewIndex];
}

/// The bytes of buffer view `viewIndex` from `byteOffset` on, which must hold `count` elements of
/// `elementSize` bytes, `stride` bytes apart. Throws when the buffer view or the elements are not
/// in the document's buffers.
[[nodiscard]] inline std::span<const std::byte> bufferViewBytes(
    const fx::gltf::Document& model,
    uint32_t viewIndex,
    size_t byteOffset,
    size_t count,
    size_t elementSize,
    size_t stride) {
  const auto& view = bufferViewAt(model, viewIndex);
  MT_THROW_IF(
      view.buffer < 0 || static_cast<size_t>(view.buffer) >= model.buffers.size(),
      "Buffer view {} refers to buffer {}, which does not exist",
      viewIndex,
      view.buffer);
  const auto& data = model.buffers[view.buffer].data;
  MT_THROW_IF(
      static_cast<size_t>(view.byteOffset) + view.byteLength > data.size(),
      "Buffer view {} ends after the end of buffer {}",
      viewIndex,
      view.buffer);
  const size_t end = count == 0 ? byteOffset : byteOffset + (count - 1) * stride + elementSize;
  MT_THROW_IF(
      end > view.byteLength,
      "Accessor data ends at byte {} of buffer view {}, which has {} bytes",
      end,
      viewIndex,
      view.byteLength);
  return std::as_bytes(std::span<const uint8_t>(data)).subspan(view.byteOffset + byteOffset);
}

/// How copyAccessorBuffer() stores a component of its result.
enum class StoredComponent : uint8_t { Float, UnsignedByte, UnsignedShort, UnsignedInt };

/// The StoredComponent of a component of type `Component`.
template <typename Component>
[[nodiscard]] constexpr StoredComponent storedComponent() {
  if constexpr (std::is_same_v<Component, float>) {
    return StoredComponent::Float;
  } else if constexpr (std::is_same_v<Component, uint8_t>) {
    return StoredComponent::UnsignedByte;
  } else if constexpr (std::is_same_v<Component, uint16_t>) {
    return StoredComponent::UnsignedShort;
  } else {
    static_assert(std::is_same_v<Component, uint32_t>, "Unsupported accessor component type");
    return StoredComponent::UnsignedInt;
  }
}

/// The number of elements of accessor `id`, which copyAccessorBuffer() allocates. Throws when the
/// accessor does not exist, or when its elements are not in its buffer view, so that no output is
/// allocated for elements that the file does not have.
[[nodiscard]] inline size_t accessorElementCount(const fx::gltf::Document& model, int32_t id) {
  MT_THROW_IF(
      id < 0 || static_cast<size_t>(id) >= model.accessors.size(),
      "Accessor {} does not exist; the file has {} accessors",
      id,
      model.accessors.size());
  const auto& accessor = model.accessors[id];
  const int32_t typeSize = getTypeSize(accessor.type);
  const int32_t componentSize = getComponentSize(accessor.componentType);
  if (accessor.bufferView >= 0 && typeSize > 0 && componentSize > 0) {
    const size_t elementSize = static_cast<size_t>(typeSize) * componentSize;
    const auto& view = bufferViewAt(model, accessor.bufferView);
    const size_t stride = view.byteStride == 0 ? elementSize : view.byteStride;
    static_cast<void>(bufferViewBytes(
        model, accessor.bufferView, accessor.byteOffset, accessor.count, elementSize, stride));
  }
  return accessor.count;
}

/// Writes the components of the elements of accessor `id`, which exists, to `out`, as
/// copyAccessorBuffer() converts them: `numComponents` components per element, each stored as
/// `stored`. Returns false, with a warning, when the elements cannot be stored so.
///
/// Not a template, so that the code exists once however many element types are read.
[[nodiscard]] inline bool readAccessorComponents(
    const fx::gltf::Document& model,
    int32_t id,
    int32_t numComponents,
    StoredComponent stored,
    std::span<std::byte> out) {
  const auto& accessor = model.accessors.at(id);
  const int32_t typeSize = getTypeSize(accessor.type);
  const int32_t componentSize = getComponentSize(accessor.componentType);
  if (typeSize != numComponents || componentSize <= 0) {
    MT_LOGW(
        "Accessor {} has elements of {} components of {} bytes, but elements of {} components are expected",
        id,
        typeSize,
        componentSize,
        numComponents);
    return false;
  }
  const bool storeFloats = stored == StoredComponent::Float;
  if (!storeFloats && !isUnsignedInteger(accessor.componentType)) {
    MT_LOGW(
        "Accessor {} has components of type {}, which are not unsigned integers",
        id,
        static_cast<int>(accessor.componentType));
    return false;
  }
  size_t storedSize = sizeof(float);
  double maxValue = 0.0;
  switch (stored) {
    case StoredComponent::Float:
      break;
    case StoredComponent::UnsignedByte:
      storedSize = sizeof(uint8_t);
      maxValue = std::numeric_limits<uint8_t>::max();
      break;
    case StoredComponent::UnsignedShort:
      storedSize = sizeof(uint16_t);
      maxValue = std::numeric_limits<uint16_t>::max();
      break;
    case StoredComponent::UnsignedInt:
      storedSize = sizeof(uint32_t);
      maxValue = std::numeric_limits<uint32_t>::max();
      break;
  }
  MT_THROW_IF(
      out.size() != static_cast<size_t>(accessor.count) * numComponents * storedSize,
      "Accessor {} needs {} bytes of output, but got {}",
      id,
      static_cast<size_t>(accessor.count) * numComponents * storedSize,
      out.size());

  // Stores component `index` of the output from the component at `data`.
  const auto store = [&](const std::byte* data, size_t index) {
    const double value = readComponent(data, accessor.componentType);
    std::byte* target = out.data() + index * storedSize;
    if (storeFloats) {
      auto component = static_cast<float>(value);
      if (accessor.normalized) {
        component = normalizedComponent(component, accessor.componentType);
      }
      std::memcpy(target, &component, sizeof(component));
      return;
    }
    MT_THROW_IF(
        value > maxValue,
        "Accessor {} has the value {}, which is too large for {} bytes",
        id,
        value,
        storedSize);
    const auto component = static_cast<uint32_t>(value);
    if (stored == StoredComponent::UnsignedByte) {
      const auto byte = static_cast<uint8_t>(component);
      std::memcpy(target, &byte, sizeof(byte));
    } else if (stored == StoredComponent::UnsignedShort) {
      const auto shortValue = static_cast<uint16_t>(component);
      std::memcpy(target, &shortValue, sizeof(shortValue));
    } else {
      std::memcpy(target, &component, sizeof(component));
    }
  };

  const auto numPerElement = static_cast<size_t>(numComponents);
  const size_t elementSize = numPerElement * componentSize;
  if (accessor.bufferView >= 0) {
    const auto& view = bufferViewAt(model, accessor.bufferView);
    const size_t stride = view.byteStride == 0 ? elementSize : view.byteStride;
    const auto bytes = bufferViewBytes(
        model, accessor.bufferView, accessor.byteOffset, accessor.count, elementSize, stride);
    for (size_t i = 0; i < accessor.count; ++i) {
      for (size_t c = 0; c < numPerElement; ++c) {
        store(&bytes[i * stride + c * componentSize], i * numPerElement + c);
      }
    }
  } else {
    // An accessor without a buffer view has zeros, but for its sparse elements.
    std::ranges::fill(out, std::byte{0});
  }
  if (!accessor.sparse.empty()) {
    const auto& sparse = accessor.sparse;
    MT_THROW_IF(
        !isUnsignedInteger(sparse.indices.componentType),
        "Accessor {} has sparse indices of type {}, which are not unsigned integers",
        id,
        static_cast<int>(sparse.indices.componentType));
    const auto count = static_cast<size_t>(sparse.count);
    const auto indexSize = static_cast<size_t>(getComponentSize(sparse.indices.componentType));
    const auto indexBytes = bufferViewBytes(
        model, sparse.indices.bufferView, sparse.indices.byteOffset, count, indexSize, indexSize);
    const auto valueBytes = bufferViewBytes(
        model, sparse.values.bufferView, sparse.values.byteOffset, count, elementSize, elementSize);
    for (size_t k = 0; k < count; ++k) {
      const auto index = static_cast<size_t>(
          readComponent(&indexBytes[k * indexSize], sparse.indices.componentType));
      MT_THROW_IF(
          index >= accessor.count,
          "Accessor {} has the sparse index {}, but only {} elements",
          id,
          index,
          accessor.count);
      for (size_t c = 0; c < numPerElement; ++c) {
        store(&valueBytes[k * elementSize + c * componentSize], index * numPerElement + c);
      }
    }
  }
  return true;
}

} // namespace detail

/// Read the elements of accessor `id` as values of type `T`.
///
/// The components are converted as the glTF format defines: a float `T` takes float components,
/// normalized integer components (mapped to [0, 1] or [-1, 1]) and other integer components; an
/// unsigned integer `T` takes unsigned integer components. Strided buffer views, sparse accessors
/// and accessors without a buffer view (whose elements are zero but for the sparse ones) are read.
///
/// @param[in] model The glTF document.
/// @param[in] id The accessor index; a negative index, which marks a missing accessor, gives no
///            elements.
/// @return The elements, or no elements with a warning when the accessor's elements cannot be
///         read as `T`: another number of components, or components that are not unsigned
///         integers for an integer `T`.
/// @throw std::runtime_error When the accessor or its data are not in the document.
template <typename T>
std::vector<T> copyAccessorBuffer(const fx::gltf::Document& model, int32_t id) {
  using Element = detail::AccessorElement<T>;
  using Component = typename Element::Component;
  static_assert(
      sizeof(T) == sizeof(Component) * Element::kCount, "T must hold only its components");
  if (id < 0) {
    return {};
  }
  std::vector<T> result(detail::accessorElementCount(model, id));
  if (!detail::readAccessorComponents(
          model,
          id,
          Element::kCount,
          detail::storedComponent<Component>(),
          std::as_writable_bytes(std::span<T>(result)))) {
    return {};
  }
  return result;
}

/// Read the elements of accessor `id` as values of type `T`, as copyAccessorBuffer() does, but
/// with a warning and no elements for an accessor without elements.
template <typename T>
std::vector<T> copyAlignedAccessorBuffer(const fx::gltf::Document& model, int32_t id) {
  if (id >= 0 && static_cast<size_t>(id) < model.accessors.size() &&
      model.accessors[id].count == 0) {
    MT_LOGW("Accessor count zero.");
    return {};
  }
  return copyAccessorBuffer<T>(model, id);
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
    buffer.byteLength = static_cast<uint32_t>(buffer.data.size());
  } else {
    // no alignment needed or already aligned, just copy data in
    buffer.data.resize(bufferDataStart + dataSize);
    std::memcpy(&buffer.data[bufferDataStart], data.data(), dataSize);
    buffer.byteLength = static_cast<uint32_t>(buffer.data.size());
  }

  // create bufferview
  const size_t bufferViewIdx = model.bufferViews.size();
  model.bufferViews.emplace_back();
  auto& bufferView = model.bufferViews.back();

  bufferView.buffer = static_cast<int32_t>(bufferIdx);
  bufferView.byteLength = static_cast<uint32_t>(dataSize);
  bufferView.byteOffset = static_cast<uint32_t>(bufferDataStart);
  if (align && elementSize % 4 != 0) {
    bufferView.byteStride = static_cast<uint32_t>(alignedElementSize);
  } else {
    bufferView.byteStride = 0;
  }

  // create accessor
  const size_t accessorIdx = model.accessors.size();
  model.accessors.emplace_back();
  auto& accessor = model.accessors.back();

  // set component type and accessor size depending on the type
  setAccessorType<T>(accessor);
  accessor.bufferView = static_cast<int32_t>(bufferViewIdx);
  accessor.byteOffset = 0;
  accessor.count = static_cast<uint32_t>(data.size());
  accessor.normalized = normalized;

  return static_cast<int32_t>(accessorIdx);
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
inline void setAccessorType<const Vector4b>(fx::gltf::Accessor& accessor) {
  accessor.componentType = fx::gltf::Accessor::ComponentType::UnsignedByte;
  accessor.type = fx::gltf::Accessor::Type::Vec4;
}

template <>
inline void setAccessorType<const Matrix4f>(fx::gltf::Accessor& accessor) {
  accessor.componentType = fx::gltf::Accessor::ComponentType::Float;
  accessor.type = fx::gltf::Accessor::Type::Mat4;
}

// add animation
/// Converts the save option to the GLTF sampler interpolation type.
inline fx::gltf::Animation::Sampler::Type toGltfSamplerType(
    GltfAnimationInterpolation interpolation) {
  switch (interpolation) {
    case GltfAnimationInterpolation::Step:
      return fx::gltf::Animation::Sampler::Type::Step;
    case GltfAnimationInterpolation::Linear:
      return fx::gltf::Animation::Sampler::Type::Linear;
  }
  return fx::gltf::Animation::Sampler::Type::Step;
}

template <typename T>
int32_t createSampler(
    fx::gltf::Document& model,
    fx::gltf::Animation& anim,
    std::span<T> data,
    const std::span<float>& timestamps,
    GltfAnimationInterpolation interpolation) {
  MT_CHECK(
      data.size() == timestamps.size(), "data: {}, timestmaps: {}", data.size(), timestamps.size());

  // create new sampler
  const auto index = static_cast<int32_t>(anim.samplers.size());
  anim.samplers.emplace_back();
  auto& sampler = anim.samplers.back();
  sampler.interpolation = toGltfSamplerType(interpolation);
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
    const int32_t timestampAccessor,
    GltfAnimationInterpolation interpolation) {
  // create new sampler
  const auto index = static_cast<int32_t>(anim.samplers.size());
  anim.samplers.emplace_back();
  auto& sampler = anim.samplers.back();
  sampler.interpolation = toGltfSamplerType(interpolation);
  sampler.input = timestampAccessor;
  sampler.output = createAccessorBuffer(model, data);

  // return sampler index
  return index;
}

/// Give @p document a buffer if it has none.
///
/// A document whose characters have no mesh and no motion has no buffer, but fx::gltf saves and
/// loads only documents with at least one buffer of at least one byte. The added buffer holds four
/// zero bytes that nothing references.
inline void ensureDocumentHasBuffer(fx::gltf::Document& document) {
  if (document.buffers.empty()) {
    auto& buffer = document.buffers.emplace_back();
    buffer.data.resize(4, 0);
    buffer.byteLength = static_cast<uint32_t>(buffer.data.size());
  }
}

} // namespace momentum
