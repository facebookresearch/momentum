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
// VertexPositionsAccessor implementation
// ============================================================================

template <typename T>
VertexPositionsAccessor<T>::VertexPositionsAccessor(
    const py::buffer_info& bufferInfo,
    const LeadingDimensions& leadingDims,
    py::ssize_t nVertices)
    : nVertices_(nVertices), leadingNDim_(leadingDims.ndim()) {
  data_ = static_cast<T*>(bufferInfo.ptr);

  // Extract strides (convert from bytes to elements)
  const auto totalNDim = bufferInfo.ndim;
  strides_.resize(totalNDim);
  for (int i = 0; i < totalNDim; ++i) {
    strides_[i] = static_cast<py::ssize_t>(bufferInfo.strides[i] / sizeof(T));
  }
}

template <typename T>
py::ssize_t VertexPositionsAccessor<T>::computeOffset(
    const std::vector<py::ssize_t>& batchIndices) const {
  py::ssize_t offset = 0;

  // Apply strides for each dimension
  // Broadcasting is automatically handled: if stride is 0, the index doesn't matter
  for (size_t i = 0; i < batchIndices.size(); ++i) {
    offset += batchIndices[i] * strides_[i];
  }

  return offset;
}

template <typename T>
std::vector<Eigen::Vector3<T>> VertexPositionsAccessor<T>::get(
    const std::vector<py::ssize_t>& batchIndices) const {
  const auto offset = computeOffset(batchIndices);

  // Create a vector of Vector3<T> using strides
  // Vertex positions format: (..., nVertices, 3) where each vertex has [x, y, z]
  std::vector<Eigen::Vector3<T>> positions(nVertices_);

  const auto rowStride = strides_[leadingNDim_]; // stride for vertex dimension
  const auto colStride = strides_[leadingNDim_ + 1]; // stride for xyz dimension

  for (py::ssize_t iVert = 0; iVert < nVertices_; ++iVert) {
    const auto vertOffset = offset + iVert * rowStride;
    positions[iVert].x() = data_[vertOffset + 0 * colStride];
    positions[iVert].y() = data_[vertOffset + 1 * colStride];
    positions[iVert].z() = data_[vertOffset + 2 * colStride];
  }

  return positions;
}

template <typename T>
void VertexPositionsAccessor<T>::set(
    const std::vector<py::ssize_t>& batchIndices,
    const std::vector<Eigen::Vector3<T>>& positions) {
  MT_THROW_IF(
      static_cast<py::ssize_t>(positions.size()) != nVertices_,
      "set: expected {} vertices but got {}",
      nVertices_,
      positions.size());

  const auto offset = computeOffset(batchIndices);
  const auto rowStride = strides_[leadingNDim_]; // stride for vertex dimension
  const auto colStride = strides_[leadingNDim_ + 1]; // stride for xyz dimension

  for (py::ssize_t iVert = 0; iVert < nVertices_; ++iVert) {
    const auto vertOffset = offset + iVert * rowStride;
    data_[vertOffset + 0 * colStride] = positions[iVert].x();
    data_[vertOffset + 1 * colStride] = positions[iVert].y();
    data_[vertOffset + 2 * colStride] = positions[iVert].z();
  }
}

// Explicit template instantiations
template class VertexPositionsAccessor<float>;
template class VertexPositionsAccessor<double>;

} // namespace pymomentum
