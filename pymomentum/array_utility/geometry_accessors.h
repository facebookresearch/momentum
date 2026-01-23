/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pymomentum/array_utility/array_utility.h>

#include <momentum/character/joint.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/types.h>
#include <momentum/math/transform.h>

#include <pybind11/numpy.h>

#include <span>

namespace pymomentum {

namespace py = pybind11;

// Generic accessor for Eigen-vector-based strong types (ModelParametersT, BlendWeightsT, etc.)
// Provides access to the strong type directly, supporting arbitrary strides without
// requiring contiguous memory.
//
// StrongType must be an Eigen vector wrapper with:
//   - A member `v` of type Eigen::VectorX<T>
//   - A constructor that takes Eigen::VectorX<T>
//
// Supported strong types:
//   - momentum::ModelParametersT<T>
//   - momentum::BlendWeightsT<T>
//   - momentum::JointParametersT<T> (flat format only)
//
// Array format: (..., vectorSize) where vectorSize is the size of the Eigen vector.
template <typename T, template <typename> class StrongType>
class VectorAccessor {
 public:
  // Construct from buffer info with shape (..., vectorSize)
  VectorAccessor(
      const py::buffer_info& bufferInfo,
      const LeadingDimensions& leadingDims,
      py::ssize_t vectorSize);

  // Convenience constructor that takes a py::buffer and extracts buffer_info.
  VectorAccessor(py::buffer& buffer, const LeadingDimensions& leadingDims, py::ssize_t vectorSize)
      : VectorAccessor(buffer.request(), leadingDims, vectorSize) {}

  // Const convenience constructor
  VectorAccessor(
      const py::buffer& buffer,
      const LeadingDimensions& leadingDims,
      py::ssize_t vectorSize)
      : VectorAccessor(buffer.request(), leadingDims, vectorSize) {}

  // Get the vector for the given batch indices.
  // Returns a StrongType<T> object constructed from the array data.
  // Handles broadcasting: if a dimension has stride 0, its index is ignored.
  StrongType<T> get(std::span<const py::ssize_t> batchIndices) const;

  // Set the vector for the given batch indices.
  // Writes the StrongType<T> data to the array using strides.
  // Handles broadcasting: if a dimension has stride 0, its index is ignored.
  void set(std::span<const py::ssize_t> batchIndices, const StrongType<T>& value);

 private:
  T* data_;
  py::ssize_t vectorSize_{};
  std::vector<py::ssize_t> strides_;
  size_t leadingNDim_{};

  // Compute the flat offset into the data array for given batch indices.
  [[nodiscard]] py::ssize_t computeOffset(std::span<const py::ssize_t> batchIndices) const;
};

// Type aliases for common strong types
template <typename T>
using ModelParametersAccessor = VectorAccessor<T, momentum::ModelParametersT>;

template <typename T>
using BlendWeightsAccessor = VectorAccessor<T, momentum::BlendWeightsT>;

// Note: JointParametersAccessor is NOT implemented using VectorAccessor because it needs
// to handle two different array formats:
//   - Flat: (..., nJoints * 7) - could use VectorAccessor
//   - Structured: (..., nJoints, 7) - requires 2D trailing dimension handling
// The structured format is important for user-facing APIs where the per-joint structure
// is more intuitive. Since both formats need to be supported transparently, we keep
// JointParametersAccessor as a separate implementation.

// Accessor for joint parameters that handles both structured and flat formats.
// Provides access to JointParametersT<T> objects directly, supporting arbitrary
// strides without requiring contiguous memory.
//
// Supported formats:
//   - Structured: (..., nJoints, 7) where each joint has [tx, ty, tz, rx, ry, rz, rw]
//   - Flat: (..., nJointParams) where nJointParams = nJoints * 7
//
// The format is detected automatically and handled transparently.
template <typename T>
class JointParametersAccessor {
 public:
  // Construct from buffer info that has been validated by ArrayChecker.
  // The shape parameter indicates which format was detected during validation.
  JointParametersAccessor(
      const py::buffer_info& bufferInfo,
      const LeadingDimensions& leadingDims,
      py::ssize_t nJoints,
      JointParamsShape shape);

  // Convenience constructor that takes a py::buffer and extracts buffer_info.
  // This allows accepting any buffer-protocol object (numpy arrays, torch tensors, etc.)
  // for backward compatibility.
  JointParametersAccessor(
      py::buffer& buffer,
      const LeadingDimensions& leadingDims,
      py::ssize_t nJoints,
      JointParamsShape shape)
      : JointParametersAccessor(buffer.request(), leadingDims, nJoints, shape) {}

  // Const convenience constructor
  JointParametersAccessor(
      const py::buffer& buffer,
      const LeadingDimensions& leadingDims,
      py::ssize_t nJoints,
      JointParamsShape shape)
      : JointParametersAccessor(buffer.request(), leadingDims, nJoints, shape) {}

  // Get the joint parameters for the given batch indices.
  // Returns a JointParametersT<T> object constructed from the array data.
  // Handles broadcasting: if a dimension has stride 0, its index is ignored.
  momentum::JointParametersT<T> get(std::span<const py::ssize_t> batchIndices) const;

  // Set the joint parameters for the given batch indices.
  // Writes the JointParametersT<T> data to the array using strides.
  // Handles broadcasting: if a dimension has stride 0, its index is ignored.
  void set(
      std::span<const py::ssize_t> batchIndices,
      const momentum::JointParametersT<T>& jointParams);

 private:
  T* data_;
  py::ssize_t nJoints_{};
  const JointParamsShape shape_;
  std::vector<py::ssize_t> strides_;
  size_t leadingNDim_{};

  // Compute the flat offset into the data array for given batch indices.
  // Handles broadcasting by using stride=0 for broadcast dimensions.
  [[nodiscard]] py::ssize_t computeOffset(std::span<const py::ssize_t> batchIndices) const;
};

// Accessor for skeleton states.
// Provides access to either full SkeletonStateT<T> or vector<Transform> depending on options.
template <typename T>
class SkeletonStateAccessor {
 public:
  // Construct from buffer info with shape (..., nJoints, 8)
  // where each joint has [tx, ty, tz, rx, ry, rz, rw, scale]
  SkeletonStateAccessor(
      const py::buffer_info& bufferInfo,
      const LeadingDimensions& leadingDims,
      py::ssize_t nJoints);

  // Convenience constructor that takes a py::buffer and extracts buffer_info.
  // This allows accepting any buffer-protocol object (numpy arrays, torch tensors, etc.)
  // for backward compatibility.
  SkeletonStateAccessor(
      py::buffer& buffer,
      const LeadingDimensions& leadingDims,
      py::ssize_t nJoints)
      : SkeletonStateAccessor(buffer.request(), leadingDims, nJoints) {}

  // Const convenience constructor
  SkeletonStateAccessor(
      const py::buffer& buffer,
      const LeadingDimensions& leadingDims,
      py::ssize_t nJoints)
      : SkeletonStateAccessor(buffer.request(), leadingDims, nJoints) {}

  // Get the transforms for the given batch indices.
  // Returns a vector of transforms (one per joint).
  // Handles broadcasting: if a dimension has stride 0, its index is ignored.
  momentum::TransformListT<T> getTransforms(std::span<const py::ssize_t> batchIndices) const;

  // Set the transforms for the given batch indices.
  // Writes transform data to the array using strides.
  // Handles broadcasting: if a dimension has stride 0, its index is ignored.
  void setTransforms(
      std::span<const py::ssize_t> batchIndices,
      const momentum::TransformListT<T>& transforms);

 private:
  T* data_;
  py::ssize_t nJoints_{};
  std::vector<py::ssize_t> strides_;
  size_t leadingNDim_{};

  // Compute the flat offset into the data array for given batch indices.
  // Handles broadcasting by using stride=0 for broadcast dimensions.
  [[nodiscard]] py::ssize_t computeOffset(std::span<const py::ssize_t> batchIndices) const;
};

// Accessor for vertex positions with shape (..., nVertices, 3).
// Provides access to std::vector<Eigen::Vector3<T>> for mesh vertex operations.
template <typename T>
class VertexPositionsAccessor {
 public:
  // Construct from buffer info with shape (..., nVertices, 3)
  VertexPositionsAccessor(
      const py::buffer_info& bufferInfo,
      const LeadingDimensions& leadingDims,
      py::ssize_t nVertices);

  // Convenience constructor that takes a py::buffer and extracts buffer_info.
  VertexPositionsAccessor(
      py::buffer& buffer,
      const LeadingDimensions& leadingDims,
      py::ssize_t nVertices)
      : VertexPositionsAccessor(buffer.request(), leadingDims, nVertices) {}

  // Const convenience constructor
  VertexPositionsAccessor(
      const py::buffer& buffer,
      const LeadingDimensions& leadingDims,
      py::ssize_t nVertices)
      : VertexPositionsAccessor(buffer.request(), leadingDims, nVertices) {}

  // Get the vertex positions for the given batch indices.
  // Returns a vector of Vector3<T> (one per vertex).
  // Handles broadcasting: if a dimension has stride 0, its index is ignored.
  std::vector<Eigen::Vector3<T>> get(const std::vector<py::ssize_t>& batchIndices) const;

  // Set the vertex positions for the given batch indices.
  // Writes vertex data to the array using strides.
  // Handles broadcasting: if a dimension has stride 0, its index is ignored.
  void set(
      const std::vector<py::ssize_t>& batchIndices,
      const std::vector<Eigen::Vector3<T>>& positions);

 private:
  T* data_;
  py::ssize_t nVertices_{};
  std::vector<py::ssize_t> strides_;
  size_t leadingNDim_{};

  // Compute the flat offset into the data array for given batch indices.
  [[nodiscard]] py::ssize_t computeOffset(const std::vector<py::ssize_t>& batchIndices) const;
};

} // namespace pymomentum
