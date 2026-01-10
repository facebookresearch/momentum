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
#include <momentum/math/transform.h>

#include <pybind11/numpy.h>

namespace pymomentum {

namespace py = pybind11;

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
  momentum::JointParametersT<T> get(const std::vector<py::ssize_t>& batchIndices) const;

  // Set the joint parameters for the given batch indices.
  // Writes the JointParametersT<T> data to the array using strides.
  // Handles broadcasting: if a dimension has stride 0, its index is ignored.
  void set(
      const std::vector<py::ssize_t>& batchIndices,
      const momentum::JointParametersT<T>& jointParams);

 private:
  T* data_;
  py::ssize_t nJoints_{};
  const JointParamsShape shape_;
  std::vector<py::ssize_t> strides_;
  size_t leadingNDim_{};

  // Compute the flat offset into the data array for given batch indices.
  // Handles broadcasting by using stride=0 for broadcast dimensions.
  [[nodiscard]] py::ssize_t computeOffset(const std::vector<py::ssize_t>& batchIndices) const;
};

// Accessor for model parameters.
// Provides access to ModelParametersT<T> objects directly.
template <typename T>
class ModelParametersAccessor {
 public:
  // Construct from buffer info with shape (..., nModelParams)
  ModelParametersAccessor(
      const py::buffer_info& bufferInfo,
      const LeadingDimensions& leadingDims,
      py::ssize_t nModelParams);

  // Convenience constructor that takes a py::buffer and extracts buffer_info.
  // This allows accepting any buffer-protocol object (numpy arrays, torch tensors, etc.)
  // for backward compatibility.
  ModelParametersAccessor(
      py::buffer& buffer,
      const LeadingDimensions& leadingDims,
      py::ssize_t nModelParams)
      : ModelParametersAccessor(buffer.request(), leadingDims, nModelParams) {}

  // Const convenience constructor
  ModelParametersAccessor(
      const py::buffer& buffer,
      const LeadingDimensions& leadingDims,
      py::ssize_t nModelParams)
      : ModelParametersAccessor(buffer.request(), leadingDims, nModelParams) {}

  // Get the model parameters for the given batch indices.
  // Returns a ModelParametersT<T> object constructed from the array data.
  // Handles broadcasting: if a dimension has stride 0, its index is ignored.
  momentum::ModelParametersT<T> get(const std::vector<py::ssize_t>& batchIndices) const;

  // Set the model parameters for the given batch indices.
  // Writes the ModelParametersT<T> data to the array using strides.
  // Handles broadcasting: if a dimension has stride 0, its index is ignored.
  void set(
      const std::vector<py::ssize_t>& batchIndices,
      const momentum::ModelParametersT<T>& modelParams);

 private:
  T* data_;
  py::ssize_t nModelParams_{};
  std::vector<py::ssize_t> strides_;
  size_t leadingNDim_{};

  // Compute the flat offset into the data array for given batch indices.
  // Handles broadcasting by using stride=0 for broadcast dimensions.
  [[nodiscard]] py::ssize_t computeOffset(const std::vector<py::ssize_t>& batchIndices) const;
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
  momentum::TransformListT<T> getTransforms(const std::vector<py::ssize_t>& batchIndices) const;

  // Set the transforms for the given batch indices.
  // Writes transform data to the array using strides.
  // Handles broadcasting: if a dimension has stride 0, its index is ignored.
  void setTransforms(
      const std::vector<py::ssize_t>& batchIndices,
      const momentum::TransformListT<T>& transforms);

 private:
  T* data_;
  py::ssize_t nJoints_{};
  std::vector<py::ssize_t> strides_;
  size_t leadingNDim_{};

  // Compute the flat offset into the data array for given batch indices.
  // Handles broadcasting by using stride=0 for broadcast dimensions.
  [[nodiscard]] py::ssize_t computeOffset(const std::vector<py::ssize_t>& batchIndices) const;
};

} // namespace pymomentum
