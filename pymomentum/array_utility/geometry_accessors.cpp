/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/array_utility/geometry_accessors.h"

#include <momentum/common/exception.h>

#include <span>

namespace pymomentum {

// ============================================================================
// VectorAccessor implementation (generic for all Eigen vector strong types)
// ============================================================================

template <typename T, template <typename> class StrongType>
VectorAccessor<T, StrongType>::VectorAccessor(
    const py::buffer_info& bufferInfo,
    const LeadingDimensions& leadingDims,
    py::ssize_t vectorSize)
    : vectorSize_(vectorSize), leadingNDim_(leadingDims.ndim()) {
  data_ = static_cast<T*>(bufferInfo.ptr);

  // Extract strides (convert from bytes to elements)
  const auto totalNDim = bufferInfo.ndim;
  strides_.resize(totalNDim);
  for (int i = 0; i < totalNDim; ++i) {
    strides_[i] = static_cast<py::ssize_t>(bufferInfo.strides[i] / sizeof(T));
  }
}

template <typename T, template <typename> class StrongType>
py::ssize_t VectorAccessor<T, StrongType>::computeOffset(
    std::span<const py::ssize_t> batchIndices) const {
  py::ssize_t offset = 0;

  // Apply strides for each dimension
  // Broadcasting is automatically handled: if stride is 0, the index doesn't matter
  for (size_t i = 0; i < batchIndices.size(); ++i) {
    offset += batchIndices[i] * strides_[i];
  }

  return offset;
}

template <typename T, template <typename> class StrongType>
StrongType<T> VectorAccessor<T, StrongType>::get(std::span<const py::ssize_t> batchIndices) const {
  const auto offset = computeOffset(batchIndices);

  // Create a vector using strides
  Eigen::Matrix<T, Eigen::Dynamic, 1> vec(vectorSize_);
  const auto paramStride = strides_[leadingNDim_];

  for (py::ssize_t i = 0; i < vectorSize_; ++i) {
    const auto elementOffset = offset + i * paramStride;
    vec(i) = data_[elementOffset];
  }

  return StrongType<T>(vec);
}

template <typename T, template <typename> class StrongType>
void VectorAccessor<T, StrongType>::set(
    std::span<const py::ssize_t> batchIndices,
    const StrongType<T>& value) {
  const auto offset = computeOffset(batchIndices);
  const auto& vec = value.v;
  const auto paramStride = strides_[leadingNDim_];

  for (py::ssize_t i = 0; i < vectorSize_; ++i) {
    const auto elementOffset = offset + i * paramStride;
    data_[elementOffset] = vec(i);
  }
}

// Explicit template instantiations for VectorAccessor
template class VectorAccessor<float, momentum::ModelParametersT>;
template class VectorAccessor<double, momentum::ModelParametersT>;
template class VectorAccessor<float, momentum::BlendWeightsT>;
template class VectorAccessor<double, momentum::BlendWeightsT>;
template class VectorAccessor<float, momentum::JointParametersT>;
template class VectorAccessor<double, momentum::JointParametersT>;

// ============================================================================
// JointParametersAccessor implementation
// ============================================================================

template <typename T>
JointParametersAccessor<T>::JointParametersAccessor(
    const py::buffer_info& bufferInfo,
    const LeadingDimensions& leadingDims,
    py::ssize_t nJoints,
    JointParamsShape shape)
    : nJoints_(nJoints), shape_(shape), leadingNDim_(leadingDims.ndim()) {
  data_ = static_cast<T*>(bufferInfo.ptr);

  // Extract strides (convert from bytes to elements)
  const auto totalNDim = bufferInfo.ndim;
  strides_.resize(totalNDim);
  for (int i = 0; i < totalNDim; ++i) {
    strides_[i] = static_cast<py::ssize_t>(bufferInfo.strides[i] / sizeof(T));
  }
}

template <typename T>
py::ssize_t JointParametersAccessor<T>::computeOffset(
    std::span<const py::ssize_t> batchIndices) const {
  py::ssize_t offset = 0;

  // Apply strides for each dimension
  // Broadcasting is automatically handled: if stride is 0, the index doesn't matter
  for (size_t i = 0; i < batchIndices.size(); ++i) {
    offset += batchIndices[i] * strides_[i];
  }

  return offset;
}

template <typename T>
momentum::JointParametersT<T> JointParametersAccessor<T>::get(
    std::span<const py::ssize_t> batchIndices) const {
  const auto offset = computeOffset(batchIndices);
  const auto nJointParams = nJoints_ * 7;

  // Create a joint parameters vector
  Eigen::Matrix<T, Eigen::Dynamic, 1> jpVec(nJointParams);

  if (shape_ == JointParamsShape::Structured) {
    // Input is (..., nJoints, 7) - read each joint's 7 values using strides
    const auto rowStride = strides_[leadingNDim_]; // stride for joint dimension
    const auto colStride = strides_[leadingNDim_ + 1]; // stride for parameter dimension

    for (py::ssize_t iJoint = 0; iJoint < nJoints_; ++iJoint) {
      for (py::ssize_t iParam = 0; iParam < 7; ++iParam) {
        const auto elementOffset = offset + iJoint * rowStride + iParam * colStride;
        jpVec(iJoint * 7 + iParam) = data_[elementOffset];
      }
    }
  } else {
    // Input is (..., nJointParams) - read flat using strides
    const auto paramStride = strides_[leadingNDim_];

    for (py::ssize_t iParam = 0; iParam < nJointParams; ++iParam) {
      const auto elementOffset = offset + iParam * paramStride;
      jpVec(iParam) = data_[elementOffset];
    }
  }

  return momentum::JointParametersT<T>(jpVec);
}

template <typename T>
void JointParametersAccessor<T>::set(
    std::span<const py::ssize_t> batchIndices,
    const momentum::JointParametersT<T>& jointParams) {
  const auto offset = computeOffset(batchIndices);
  const auto& jpVec = jointParams.v;

  if (shape_ == JointParamsShape::Structured) {
    // Output is (..., nJoints, 7) - write each joint's 7 values using strides
    const auto rowStride = strides_[leadingNDim_]; // stride for joint dimension
    const auto colStride = strides_[leadingNDim_ + 1]; // stride for parameter dimension

    for (py::ssize_t iJoint = 0; iJoint < nJoints_; ++iJoint) {
      for (py::ssize_t iParam = 0; iParam < 7; ++iParam) {
        const auto elementOffset = offset + iJoint * rowStride + iParam * colStride;
        data_[elementOffset] = jpVec(iJoint * 7 + iParam);
      }
    }
  } else {
    // Output is (..., nJointParams) - write flat using strides
    const auto paramStride = strides_[leadingNDim_];
    const auto nJointParams = nJoints_ * 7;

    for (py::ssize_t iParam = 0; iParam < nJointParams; ++iParam) {
      const auto elementOffset = offset + iParam * paramStride;
      data_[elementOffset] = jpVec(iParam);
    }
  }
}

// Explicit template instantiations
template class JointParametersAccessor<float>;
template class JointParametersAccessor<double>;

// ============================================================================
// SkeletonStateAccessor implementation
// ============================================================================

template <typename T>
SkeletonStateAccessor<T>::SkeletonStateAccessor(
    const py::buffer_info& bufferInfo,
    const LeadingDimensions& leadingDims,
    py::ssize_t nJoints)
    : nJoints_(nJoints), leadingNDim_(leadingDims.ndim()) {
  data_ = static_cast<T*>(bufferInfo.ptr);

  // Extract strides (convert from bytes to elements)
  const auto totalNDim = bufferInfo.ndim;
  strides_.resize(totalNDim);
  for (int i = 0; i < totalNDim; ++i) {
    strides_[i] = static_cast<py::ssize_t>(bufferInfo.strides[i] / sizeof(T));
  }
}

template <typename T>
py::ssize_t SkeletonStateAccessor<T>::computeOffset(
    std::span<const py::ssize_t> batchIndices) const {
  py::ssize_t offset = 0;

  // Apply strides for each dimension
  // Broadcasting is automatically handled: if stride is 0, the index doesn't matter
  for (size_t i = 0; i < batchIndices.size(); ++i) {
    offset += batchIndices[i] * strides_[i];
  }

  return offset;
}

template <typename T>
momentum::TransformListT<T> SkeletonStateAccessor<T>::getTransforms(
    std::span<const py::ssize_t> batchIndices) const {
  const auto offset = computeOffset(batchIndices);

  // Create transforms using strides
  // Skeleton state format: (..., nJoints, 8) where each joint has
  // [tx, ty, tz, rx, ry, rz, rw, scale]
  momentum::TransformListT<T> transforms(nJoints_);

  const auto rowStride = strides_[leadingNDim_]; // stride for joint dimension
  const auto colStride = strides_[leadingNDim_ + 1]; // stride for parameter dimension

  for (py::ssize_t iJoint = 0; iJoint < nJoints_; ++iJoint) {
    const auto jointOffset = offset + iJoint * rowStride;

    // Read translation
    transforms[iJoint].translation.x() = data_[jointOffset + 0 * colStride];
    transforms[iJoint].translation.y() = data_[jointOffset + 1 * colStride];
    transforms[iJoint].translation.z() = data_[jointOffset + 2 * colStride];

    // Read rotation (quaternion in x,y,z,w format)
    const T qx = data_[jointOffset + 3 * colStride];
    const T qy = data_[jointOffset + 4 * colStride];
    const T qz = data_[jointOffset + 5 * colStride];
    const T qw = data_[jointOffset + 6 * colStride];
    transforms[iJoint].rotation = Eigen::Quaternion<T>(qw, qx, qy, qz);

    // Read scale
    const T scale = data_[jointOffset + 7 * colStride];
    transforms[iJoint].scale = scale;
  }

  return transforms;
}

template <typename T>
void SkeletonStateAccessor<T>::setTransforms(
    std::span<const py::ssize_t> batchIndices,
    const momentum::TransformListT<T>& transforms) {
  MT_THROW_IF(
      static_cast<py::ssize_t>(transforms.size()) != nJoints_,
      "setTransforms: expected {} transforms but got {}",
      nJoints_,
      transforms.size());

  const auto offset = computeOffset(batchIndices);
  const auto rowStride = strides_[leadingNDim_]; // stride for joint dimension
  const auto colStride = strides_[leadingNDim_ + 1]; // stride for parameter dimension

  for (py::ssize_t iJoint = 0; iJoint < nJoints_; ++iJoint) {
    const auto jointOffset = offset + iJoint * rowStride;
    const auto& transform = transforms[iJoint];

    // Write translation
    data_[jointOffset + 0 * colStride] = transform.translation.x();
    data_[jointOffset + 1 * colStride] = transform.translation.y();
    data_[jointOffset + 2 * colStride] = transform.translation.z();

    // Write rotation (quaternion in x,y,z,w format)
    data_[jointOffset + 3 * colStride] = transform.rotation.x();
    data_[jointOffset + 4 * colStride] = transform.rotation.y();
    data_[jointOffset + 5 * colStride] = transform.rotation.z();
    data_[jointOffset + 6 * colStride] = transform.rotation.w();

    // Write scale
    data_[jointOffset + 7 * colStride] = transform.scale;
  }
}

// Explicit template instantiations
template class SkeletonStateAccessor<float>;
template class SkeletonStateAccessor<double>;

// ============================================================================
// VectorArrayAccessor implementation
// ============================================================================

template <typename T, int Dim>
VectorArrayAccessor<T, Dim>::VectorArrayAccessor(
    const py::buffer_info& bufferInfo,
    const LeadingDimensions& leadingDims,
    py::ssize_t nElements)
    : nElements_(nElements), leadingNDim_(leadingDims.ndim()) {
  data_ = static_cast<T*>(bufferInfo.ptr);

  // Extract strides (convert from bytes to elements)
  const auto totalNDim = bufferInfo.ndim;
  strides_.resize(totalNDim);
  for (int i = 0; i < totalNDim; ++i) {
    strides_[i] = static_cast<py::ssize_t>(bufferInfo.strides[i] / sizeof(T));
  }

  // Cache the row and column strides for the trailing dimensions
  rowStride_ = strides_[leadingNDim_]; // stride for element dimension
  colStride_ = strides_[leadingNDim_ + 1]; // stride for vector component dimension
}

template <typename T, int Dim>
py::ssize_t VectorArrayAccessor<T, Dim>::computeOffset(
    const std::vector<py::ssize_t>& batchIndices) const {
  py::ssize_t offset = 0;

  // Apply strides for each dimension
  // Broadcasting is automatically handled: if stride is 0, the index doesn't matter
  for (size_t i = 0; i < batchIndices.size(); ++i) {
    offset += batchIndices[i] * strides_[i];
  }

  return offset;
}

template <typename T, int Dim>
typename VectorArrayAccessor<T, Dim>::ElementView VectorArrayAccessor<T, Dim>::view(
    const std::vector<py::ssize_t>& batchIndices) {
  const auto offset = computeOffset(batchIndices);
  return ElementView(data_ + offset, nElements_, rowStride_, colStride_);
}

template <typename T, int Dim>
typename VectorArrayAccessor<T, Dim>::ElementView VectorArrayAccessor<T, Dim>::view(
    const std::vector<py::ssize_t>& batchIndices) const {
  const auto offset = computeOffset(batchIndices);
  // const_cast is safe here because we control access at the call site
  return ElementView(const_cast<T*>(data_) + offset, nElements_, rowStride_, colStride_);
}

template <typename T, int Dim>
std::vector<typename VectorArrayAccessor<T, Dim>::VectorType> VectorArrayAccessor<T, Dim>::get(
    const std::vector<py::ssize_t>& batchIndices) const {
  const auto offset = computeOffset(batchIndices);

  std::vector<VectorType> result(nElements_);
  for (py::ssize_t i = 0; i < nElements_; ++i) {
    const auto elemOffset = offset + i * rowStride_;
    for (int d = 0; d < Dim; ++d) {
      result[i][d] = data_[elemOffset + d * colStride_];
    }
  }

  return result;
}

template <typename T, int Dim>
void VectorArrayAccessor<T, Dim>::set(
    const std::vector<py::ssize_t>& batchIndices,
    const std::vector<VectorType>& values) {
  MT_THROW_IF(
      static_cast<py::ssize_t>(values.size()) != nElements_,
      "set: expected {} elements but got {}",
      nElements_,
      values.size());

  const auto offset = computeOffset(batchIndices);
  for (py::ssize_t i = 0; i < nElements_; ++i) {
    const auto elemOffset = offset + i * rowStride_;
    for (int d = 0; d < Dim; ++d) {
      data_[elemOffset + d * colStride_] = values[i][d];
    }
  }
}

// Explicit template instantiations for VectorArrayAccessor
template class VectorArrayAccessor<float, 3>;
template class VectorArrayAccessor<double, 3>;
template class VectorArrayAccessor<int, 3>;

// ============================================================================
// IntVectorArrayAccessor implementation
// ============================================================================

template <int Dim>
IntVectorArrayAccessor<Dim>::IntVectorArrayAccessor(
    const py::buffer& buffer,
    const LeadingDimensions& leadingDims,
    py::ssize_t nElements)
    : nElements_(nElements), leadingNDim_(leadingDims.ndim()) {
  auto bufferInfo = buffer.request();
  data_ = bufferInfo.ptr;

  // Detect the source dtype based on format code and itemsize.
  // Format codes 'l' (long) and 'L' (unsigned long) are platform-dependent:
  // - LP64 (Linux/macOS 64-bit): long is 64-bit
  // - LLP64 (Windows 64-bit): long is 32-bit
  // We use itemsize to disambiguate these cases.
  const auto& fmt = bufferInfo.format;
  const auto itemsize = bufferInfo.itemsize;

  // Check for signed integer types
  const bool isSignedInt = fmt == py::format_descriptor<int32_t>::format() ||
      fmt == py::format_descriptor<int>::format() ||
      fmt == py::format_descriptor<int64_t>::format() ||
      fmt == "l"; // C long (size varies by platform)

  // Check for unsigned integer types
  const bool isUnsignedInt = fmt == py::format_descriptor<uint32_t>::format() ||
      fmt == py::format_descriptor<uint64_t>::format() ||
      fmt == "L"; // C unsigned long (size varies by platform)

  if (isSignedInt) {
    if (itemsize == 4) {
      dtype_ = SourceDtype::Int32;
    } else if (itemsize == 8) {
      dtype_ = SourceDtype::Int64;
    } else {
      MT_THROW(
          "IntVectorArrayAccessor: unexpected itemsize {} for signed integer format '{}'",
          itemsize,
          fmt);
    }
  } else if (isUnsignedInt) {
    if (itemsize == 4) {
      dtype_ = SourceDtype::UInt32;
    } else if (itemsize == 8) {
      dtype_ = SourceDtype::UInt64;
    } else {
      MT_THROW(
          "IntVectorArrayAccessor: unexpected itemsize {} for unsigned integer format '{}'",
          itemsize,
          fmt);
    }
  } else {
    MT_THROW(
        "IntVectorArrayAccessor: expected integer dtype (int32, int64, uint32, or uint64), got format '{}'",
        fmt);
  }

  // Store byte strides (not element strides) - we'll convert in get()
  const auto totalNDim = bufferInfo.ndim;
  byteStrides_.resize(totalNDim);
  for (int i = 0; i < totalNDim; ++i) {
    byteStrides_[i] = bufferInfo.strides[i];
  }

  // Cache the row and column byte strides for the trailing dimensions
  rowByteStride_ = byteStrides_[leadingNDim_];
  colByteStride_ = byteStrides_[leadingNDim_ + 1];
}

template <int Dim>
py::ssize_t IntVectorArrayAccessor<Dim>::computeByteOffset(
    const std::vector<py::ssize_t>& batchIndices) const {
  py::ssize_t offset = 0;

  // Apply byte strides for each dimension
  // Broadcasting is automatically handled: if stride is 0, the index doesn't matter
  for (size_t i = 0; i < batchIndices.size(); ++i) {
    offset += batchIndices[i] * byteStrides_[i];
  }

  return offset;
}

template <int Dim>
typename IntVectorArrayAccessor<Dim>::ElementView IntVectorArrayAccessor<Dim>::view(
    const std::vector<py::ssize_t>& batchIndices) const {
  const auto byteOffset = computeByteOffset(batchIndices);
  const auto* offsetData = static_cast<const char*>(data_) + byteOffset;
  return ElementView(offsetData, nElements_, rowByteStride_, colByteStride_, dtype_);
}

template <int Dim>
typename IntVectorArrayAccessor<Dim>::VectorType IntVectorArrayAccessor<Dim>::ElementView::get(
    py::ssize_t index) const {
  VectorType result;

  switch (dtype_) {
    case SourceDtype::Int32: {
      const auto elemStride = rowStride_ / static_cast<py::ssize_t>(sizeof(int32_t));
      const auto compStride = colStride_ / static_cast<py::ssize_t>(sizeof(int32_t));
      const auto* ptr = static_cast<const int32_t*>(data_);
      for (int d = 0; d < Dim; ++d) {
        result[d] = static_cast<int>(ptr[index * elemStride + d * compStride]);
      }
      break;
    }
    case SourceDtype::Int64: {
      const auto elemStride = rowStride_ / static_cast<py::ssize_t>(sizeof(int64_t));
      const auto compStride = colStride_ / static_cast<py::ssize_t>(sizeof(int64_t));
      const auto* ptr = static_cast<const int64_t*>(data_);
      for (int d = 0; d < Dim; ++d) {
        result[d] = static_cast<int>(ptr[index * elemStride + d * compStride]);
      }
      break;
    }
    case SourceDtype::UInt32: {
      const auto elemStride = rowStride_ / static_cast<py::ssize_t>(sizeof(uint32_t));
      const auto compStride = colStride_ / static_cast<py::ssize_t>(sizeof(uint32_t));
      const auto* ptr = static_cast<const uint32_t*>(data_);
      for (int d = 0; d < Dim; ++d) {
        result[d] = static_cast<int>(ptr[index * elemStride + d * compStride]);
      }
      break;
    }
    case SourceDtype::UInt64: {
      const auto elemStride = rowStride_ / static_cast<py::ssize_t>(sizeof(uint64_t));
      const auto compStride = colStride_ / static_cast<py::ssize_t>(sizeof(uint64_t));
      const auto* ptr = static_cast<const uint64_t*>(data_);
      for (int d = 0; d < Dim; ++d) {
        result[d] = static_cast<int>(ptr[index * elemStride + d * compStride]);
      }
      break;
    }
  }

  return result;
}

template <int Dim>
std::pair<int, int> IntVectorArrayAccessor<Dim>::ElementView::minmax() const {
  if (nElements_ == 0) {
    return {INT_MAX, INT_MIN};
  }

  int minVal = INT_MAX;
  int maxVal = INT_MIN;

  // Helper lambda to compute min/max for a given source type
  auto computeMinMax = [&]<typename SourceT>() {
    const auto elemStride = rowStride_ / static_cast<py::ssize_t>(sizeof(SourceT));
    const auto compStride = colStride_ / static_cast<py::ssize_t>(sizeof(SourceT));
    const auto* ptr = static_cast<const SourceT*>(data_);

    for (py::ssize_t i = 0; i < nElements_; ++i) {
      for (int d = 0; d < Dim; ++d) {
        const int val = static_cast<int>(ptr[i * elemStride + d * compStride]);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
      }
    }
  };

  switch (dtype_) {
    case SourceDtype::Int32:
      computeMinMax.template operator()<int32_t>();
      break;
    case SourceDtype::Int64:
      computeMinMax.template operator()<int64_t>();
      break;
    case SourceDtype::UInt32:
      computeMinMax.template operator()<uint32_t>();
      break;
    case SourceDtype::UInt64:
      computeMinMax.template operator()<uint64_t>();
      break;
  }

  return {minVal, maxVal};
}

// Explicit template instantiations for IntVectorArrayAccessor
template class IntVectorArrayAccessor<3>;

// ============================================================================
// IntScalarArrayAccessor implementation
// ============================================================================

IntScalarArrayAccessor::IntScalarArrayAccessor(const py::buffer& buffer, py::ssize_t expectedSize)
    : size_(expectedSize) {
  auto bufferInfo = buffer.request();
  data_ = bufferInfo.ptr;

  MT_THROW_IF(
      bufferInfo.ndim != 1, "IntScalarArrayAccessor: expected 1D array, got {}D", bufferInfo.ndim);
  MT_THROW_IF(
      bufferInfo.shape[0] != expectedSize,
      "IntScalarArrayAccessor: expected size {}, got {}",
      expectedSize,
      bufferInfo.shape[0]);

  byteStride_ = bufferInfo.strides[0];

  // Detect the source dtype based on format code and itemsize.
  // Format codes 'l' (long) and 'L' (unsigned long) are platform-dependent:
  // - LP64 (Linux/macOS 64-bit): long is 64-bit
  // - LLP64 (Windows 64-bit): long is 32-bit
  // We use itemsize to disambiguate these cases.
  const auto& fmt = bufferInfo.format;
  const auto itemsize = bufferInfo.itemsize;

  // Check for signed integer types
  const bool isSignedInt = fmt == py::format_descriptor<int32_t>::format() ||
      fmt == py::format_descriptor<int>::format() ||
      fmt == py::format_descriptor<int64_t>::format() ||
      fmt == "l"; // C long (size varies by platform)

  // Check for unsigned integer types
  const bool isUnsignedInt = fmt == py::format_descriptor<uint32_t>::format() ||
      fmt == py::format_descriptor<uint64_t>::format() ||
      fmt == "L"; // C unsigned long (size varies by platform)

  if (isSignedInt) {
    if (itemsize == 4) {
      dtype_ = SourceDtype::Int32;
    } else if (itemsize == 8) {
      dtype_ = SourceDtype::Int64;
    } else {
      MT_THROW(
          "IntScalarArrayAccessor: unexpected itemsize {} for signed integer format '{}'",
          itemsize,
          fmt);
    }
  } else if (isUnsignedInt) {
    if (itemsize == 4) {
      dtype_ = SourceDtype::UInt32;
    } else if (itemsize == 8) {
      dtype_ = SourceDtype::UInt64;
    } else {
      MT_THROW(
          "IntScalarArrayAccessor: unexpected itemsize {} for unsigned integer format '{}'",
          itemsize,
          fmt);
    }
  } else {
    MT_THROW(
        "IntScalarArrayAccessor: expected integer dtype (int32, int64, uint32, or uint64), got format '{}'",
        fmt);
  }
}

int IntScalarArrayAccessor::get(py::ssize_t index) const {
  const auto* bytePtr = static_cast<const char*>(data_) + index * byteStride_;

  switch (dtype_) {
    case SourceDtype::Int32:
      return static_cast<int>(*reinterpret_cast<const int32_t*>(bytePtr));
    case SourceDtype::Int64:
      return static_cast<int>(*reinterpret_cast<const int64_t*>(bytePtr));
    case SourceDtype::UInt32:
      return static_cast<int>(*reinterpret_cast<const uint32_t*>(bytePtr));
    case SourceDtype::UInt64:
      return static_cast<int>(*reinterpret_cast<const uint64_t*>(bytePtr));
  }
  // Unreachable, but silence compiler warning
  return 0;
}

std::pair<int, int> IntScalarArrayAccessor::minmax() const {
  if (size_ == 0) {
    return {INT_MAX, INT_MIN};
  }

  int minVal = INT_MAX;
  int maxVal = INT_MIN;

  // Helper lambda to compute min/max for a given source type
  auto computeMinMax = [&]<typename SourceT>() {
    const auto elemStride = byteStride_ / static_cast<py::ssize_t>(sizeof(SourceT));
    const auto* ptr = static_cast<const SourceT*>(data_);

    for (py::ssize_t i = 0; i < size_; ++i) {
      const int val = static_cast<int>(ptr[i * elemStride]);
      minVal = std::min(minVal, val);
      maxVal = std::max(maxVal, val);
    }
  };

  switch (dtype_) {
    case SourceDtype::Int32:
      computeMinMax.template operator()<int32_t>();
      break;
    case SourceDtype::Int64:
      computeMinMax.template operator()<int64_t>();
      break;
    case SourceDtype::UInt32:
      computeMinMax.template operator()<uint32_t>();
      break;
    case SourceDtype::UInt64:
      computeMinMax.template operator()<uint64_t>();
      break;
  }

  return {minVal, maxVal};
}

} // namespace pymomentum
