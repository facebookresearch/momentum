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

#include <climits>
#include <cmath>
#include <span>
#include <utility>

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

// Generic accessor for arrays of fixed-size vectors with shape (..., nElements, Dim).
// Provides both bulk operations (get/set all elements) and element-level access via ElementView.
//
// Template parameters:
//   T   - scalar type (float, double)
//   Dim - dimension of each vector (e.g., 3 for xyz positions/normals)
//
// Usage pattern:
//   VectorArrayAccessor<float, 3> accessor(bufferInfo, leadingDims, nVertices);
//
//   // Bulk operation: get all vertices at once
//   auto positions = accessor.get(batchIndices);
//
//   // Element-level access: get a view for a specific batch index
//   auto view = accessor.view(batchIndices);
//   Eigen::Vector3f p0 = view.get(0);    // Get element at index 0
//   view.set(1, someVector);              // Set element at index 1
//   view.add(2, delta);                   // Accumulate to element at index 2
template <typename T, int Dim = 3>
class VectorArrayAccessor {
 public:
  using VectorType = Eigen::Vector<T, Dim>;

  // Lightweight view into a specific batch element.
  // Provides element-level access without additional memory allocation.
  class ElementView {
   public:
    ElementView(T* data, py::ssize_t nElements, py::ssize_t rowStride, py::ssize_t colStride)
        : data_(data), nElements_(nElements), rowStride_(rowStride), colStride_(colStride) {}

    // Get the number of elements in this view
    [[nodiscard]] py::ssize_t size() const {
      return nElements_;
    }

    // Get the vector at the given element index
    VectorType get(py::ssize_t index) const {
      VectorType result;
      const auto offset = index * rowStride_;
      for (int d = 0; d < Dim; ++d) {
        result[d] = data_[offset + d * colStride_];
      }
      return result;
    }

    // Set the vector at the given element index
    void set(py::ssize_t index, const VectorType& value) {
      const auto offset = index * rowStride_;
      for (int d = 0; d < Dim; ++d) {
        data_[offset + d * colStride_] = value[d];
      }
    }

    // Add to the vector at the given element index (for accumulation)
    void add(py::ssize_t index, const VectorType& delta) {
      const auto offset = index * rowStride_;
      for (int d = 0; d < Dim; ++d) {
        data_[offset + d * colStride_] += delta[d];
      }
    }

    // Set all elements to zero
    void setZero() {
      for (py::ssize_t i = 0; i < nElements_; ++i) {
        const auto offset = i * rowStride_;
        for (int d = 0; d < Dim; ++d) {
          data_[offset + d * colStride_] = T(0);
        }
      }
    }

    // Normalize all vectors (divide by their L2 norm)
    void normalize() {
      for (py::ssize_t i = 0; i < nElements_; ++i) {
        const auto offset = i * rowStride_;
        T sumSq = T(0);
        for (int d = 0; d < Dim; ++d) {
          T val = data_[offset + d * colStride_];
          sumSq += val * val;
        }
        if (sumSq > T(0)) {
          T invNorm = T(1) / std::sqrt(sumSq);
          for (int d = 0; d < Dim; ++d) {
            data_[offset + d * colStride_] *= invNorm;
          }
        }
      }
    }

   private:
    T* data_;
    py::ssize_t nElements_;
    py::ssize_t rowStride_;
    py::ssize_t colStride_;
  };

  // Construct from buffer info with shape (..., nElements, Dim)
  VectorArrayAccessor(
      const py::buffer_info& bufferInfo,
      const LeadingDimensions& leadingDims,
      py::ssize_t nElements);

  // Convenience constructor that takes a py::buffer and extracts buffer_info.
  VectorArrayAccessor(
      py::buffer& buffer,
      const LeadingDimensions& leadingDims,
      py::ssize_t nElements)
      : VectorArrayAccessor(buffer.request(), leadingDims, nElements) {}

  // Const convenience constructor
  VectorArrayAccessor(
      const py::buffer& buffer,
      const LeadingDimensions& leadingDims,
      py::ssize_t nElements)
      : VectorArrayAccessor(buffer.request(), leadingDims, nElements) {}

  // Get a view for the given batch indices.
  // Returns an ElementView that provides element-level access.
  ElementView view(const std::vector<py::ssize_t>& batchIndices);

  // Const version that returns a const-qualified view
  // Note: ElementView stores a non-const pointer but const correctness is enforced at call site
  ElementView view(const std::vector<py::ssize_t>& batchIndices) const;

  // Get all vectors for the given batch indices (bulk operation).
  // Returns a vector of VectorType (one per element).
  std::vector<VectorType> get(const std::vector<py::ssize_t>& batchIndices) const;

  // Set all vectors for the given batch indices (bulk operation).
  void set(const std::vector<py::ssize_t>& batchIndices, const std::vector<VectorType>& values);

 private:
  T* data_;
  py::ssize_t nElements_{};
  std::vector<py::ssize_t> strides_;
  size_t leadingNDim_{};
  py::ssize_t rowStride_{};
  py::ssize_t colStride_{};

  // Compute the flat offset into the data array for given batch indices.
  [[nodiscard]] py::ssize_t computeOffset(const std::vector<py::ssize_t>& batchIndices) const;
};

// Type alias for backward compatibility and convenience
template <typename T>
using VertexPositionsAccessor = VectorArrayAccessor<T, 3>;

// Type-erased integer vector array accessor.
// Accepts any integer dtype (int32, int64, uint32, uint64) and converts to int on-the-fly.
// This avoids templating on integer types while still supporting multiple input dtypes.
//
// Template parameter:
//   Dim - dimension of each vector (e.g., 3 for triangle indices)
//
// Usage:
//   IntVectorArrayAccessor<3> accessor(buffer, leadingDims, nElements);
//   auto view = accessor.view({});
//   Eigen::Vector3i vec = view.get(index);  // Converts from source dtype to int
template <int Dim = 3>
class IntVectorArrayAccessor {
 public:
  using VectorType = Eigen::Vector<int, Dim>;

  // Supported source integer dtypes
  enum class SourceDtype { Int32, Int64, UInt32, UInt64 };

  // Lightweight view into a specific batch element.
  // Provides element-level read access with on-the-fly type conversion.
  class ElementView {
   public:
    // Get the number of elements in this view
    [[nodiscard]] py::ssize_t size() const {
      return nElements_;
    }

    // Get the vector at the given element index, converting to int on-the-fly
    VectorType get(py::ssize_t index) const;

    // Compute the minimum and maximum values across all elements and components.
    // Returns {min, max}. For empty views, returns {INT_MAX, INT_MIN}.
    [[nodiscard]] std::pair<int, int> minmax() const;

   private:
    friend class IntVectorArrayAccessor;

    const void* data_; // Type-erased pointer
    py::ssize_t nElements_;
    py::ssize_t rowStride_; // In BYTES (not elements)
    py::ssize_t colStride_; // In BYTES
    SourceDtype dtype_;

    ElementView(
        const void* data,
        py::ssize_t nElements,
        py::ssize_t rowStride,
        py::ssize_t colStride,
        SourceDtype dtype)
        : data_(data),
          nElements_(nElements),
          rowStride_(rowStride),
          colStride_(colStride),
          dtype_(dtype) {}
  };

  // Construct from buffer, auto-detecting the integer dtype.
  // Throws if the buffer dtype is not a supported integer type.
  IntVectorArrayAccessor(
      const py::buffer& buffer,
      const LeadingDimensions& leadingDims,
      py::ssize_t nElements);

  // Get a view for the given batch indices.
  // Returns an ElementView that provides element-level access with type conversion.
  ElementView view(const std::vector<py::ssize_t>& batchIndices) const;

 private:
  const void* data_;
  py::ssize_t nElements_{};
  std::vector<py::ssize_t> byteStrides_; // Keep as byte strides
  size_t leadingNDim_{};
  py::ssize_t rowByteStride_{};
  py::ssize_t colByteStride_{};
  SourceDtype dtype_;

  // Compute the byte offset into the data for given batch indices.
  [[nodiscard]] py::ssize_t computeByteOffset(const std::vector<py::ssize_t>& batchIndices) const;
};

} // namespace pymomentum
