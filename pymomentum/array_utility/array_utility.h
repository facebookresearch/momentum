/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/common/exception.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <optional>
#include <unordered_map>
#include <vector>

namespace pymomentum {

namespace py = pybind11;

// Enum for joint parameters shape
enum class JointParamsShape {
  Structured, // (..., nJoints, 7)
  Flat // (..., nJointParams)
};

// Enum to specify expected dtype
enum class ArrayDtype {
  Float32,
  Float64,
  Auto // Auto-detect from first array
};

// Stores information about the leading batch dimensions
// For example, if an array has shape [5, 3, numModelParams], and numModelParams
// is the trailing dimension, the leading dimensions are [5, 3].
struct LeadingDimensions {
  std::vector<py::ssize_t> dims;

  // Returns the total number of batch elements (product of all leading dims)
  [[nodiscard]] py::ssize_t totalBatchElements() const;

  // Checks if this can be broadcast with another LeadingDimensions
  [[nodiscard]] bool broadcastCompatibleWith(const LeadingDimensions& other) const;

  // Computes the broadcasted result of this and other
  [[nodiscard]] LeadingDimensions broadcastWith(const LeadingDimensions& other) const;

  // Returns true if all dimensions are 1 (can be broadcast to any shape)
  [[nodiscard]] bool isScalar() const;

  // Returns number of leading dimensions
  [[nodiscard]] size_t ndim() const {
    return dims.size();
  }
};

// Main validation class for numpy arrays.
// Validates shape, dtype, and handles broadcasting logic for arrays with
// arbitrary leading dimensions and fixed trailing dimensions.
class ArrayChecker {
 public:
  explicit ArrayChecker(const char* functionName, ArrayDtype expectedDtype = ArrayDtype::Auto);

  // Validates array shape, dtype, and handles broadcasting.
  // Does NOT return a copy - the input array is validated but not modified.
  //
  // Parameters:
  //   array: Input numpy array to validate
  //   arrayName: Name of the array for error messages
  //   trailingDims: Expected trailing dimensions. Negative values indicate
  //       "variable" dimensions that are bound on first use and validated on
  //       subsequent uses (e.g., {-1, 3} means "variable size x 3").
  //   dimensionNames: Names for each trailing dimension for error messages
  //   allowEmpty: Whether to allow empty arrays
  //
  // The array is expected to have shape [..., trailingDims], where "..." are
  // the leading dimensions that must match or broadcast with other arrays
  // validated by this checker.
  void validateArray(
      const py::array& array,
      const char* arrayName,
      const std::vector<int>& trailingDims,
      const std::vector<const char*>& dimensionNames,
      bool allowEmpty = false);

  // Helper validation methods for common geometry types
  // These validate against a Character's dimensions

  // Validate skeleton state array with shape (..., nJoints, 8)
  void validateSkeletonState(
      const py::array& array,
      const char* arrayName,
      const momentum::Character& character);

  // Validate joint parameters array
  // Accepts EITHER:
  //   - (..., nJoints, 7): structured format
  //   - (..., nJointParams): flat format where nJointParams = nJoints * 7
  JointParamsShape validateJointParameters(
      const py::array& array,
      const char* arrayName,
      const momentum::Character& character);

  // Validate model parameters array with shape (..., nModelParams)
  void validateModelParameters(
      const py::array& array,
      const char* arrayName,
      const momentum::Character& character);

  // Get bound variable value (for negative indices in trailingDims)
  int64_t getBoundValue(int idx) const;

  // Get the detected dtype (Float32 or Float64)
  ArrayDtype getDetectedDtype() const {
    return detectedDtype_;
  }

  // Returns true if dtype is Float64
  bool isFloat64() const {
    return detectedDtype_ == ArrayDtype::Float64;
  }

  // Get leading dimensions (for iteration and output array creation)
  const LeadingDimensions& getLeadingDimensions() const {
    return leadingDims_;
  }

  // Get total number of batch elements for parallel iteration
  py::ssize_t getBatchSize() const {
    return leadingDims_.totalBatchElements();
  }

 private:
  const char* functionName_;
  ArrayDtype requestedDtype_;
  ArrayDtype detectedDtype_ = ArrayDtype::Auto;
  LeadingDimensions leadingDims_;
  bool leadingDimsSet_ = false;
  std::unordered_map<int, int64_t> boundVariableSizes_;

  void detectAndValidateDtype(const py::array& array, const char* arrayName);
  void validateAndUpdateLeadingDims(
      const py::array& array,
      const char* arrayName,
      size_t numTrailingDims);
};

// Helper to create output arrays with matching leading dimensions
template <typename T>
py::array_t<T> createOutputArray(
    const LeadingDimensions& leadingDims,
    const std::vector<py::ssize_t>& trailingDims);

// Format array dimensions for error messages
std::string formatArrayDims(const py::array& array);
std::string formatArrayDims(const std::vector<py::ssize_t>& dims);

std::string formatExpectedDims(
    const std::vector<int>& trailingDims,
    const std::vector<const char*>& dimensionNames,
    const std::unordered_map<int, int64_t>& boundVariableSizes);

// Convert Eigen vector to numpy array
template <typename T>
py::array_t<T> toArray(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vec);

// Convert std::vector to numpy array
template <typename T>
py::array_t<T> toArray(const std::vector<T>& vec);

// Convert Eigen matrix to numpy array (row-major output)
template <typename T, int Rows, int Cols>
py::array_t<T> toArray(const Eigen::Matrix<T, Rows, Cols>& mat);

// ============================================================================
// Template implementations
// ============================================================================

template <typename T>
py::array_t<T> createOutputArray(
    const LeadingDimensions& leadingDims,
    const std::vector<py::ssize_t>& trailingDims) {
  std::vector<py::ssize_t> shape;
  shape.reserve(leadingDims.dims.size() + trailingDims.size());
  shape.insert(shape.end(), leadingDims.dims.begin(), leadingDims.dims.end());
  shape.insert(shape.end(), trailingDims.begin(), trailingDims.end());
  return py::array_t<T>(shape);
}

template <typename T>
py::array_t<T> toArray(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vec) {
  py::array_t<T> result(vec.size());
  std::memcpy(result.mutable_data(), vec.data(), vec.size() * sizeof(T));
  return result;
}

template <typename T>
py::array_t<T> toArray(const std::vector<T>& vec) {
  py::array_t<T> result(vec.size());
  std::memcpy(result.mutable_data(), vec.data(), vec.size() * sizeof(T));
  return result;
}

template <typename T, int Rows, int Cols>
py::array_t<T> toArray(const Eigen::Matrix<T, Rows, Cols>& mat) {
  py::array_t<T> result({mat.rows(), mat.cols()});
  // Eigen is column-major by default, so we need to transpose for row-major output
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      result.mutable_data(), mat.rows(), mat.cols()) = mat;
  return result;
}

} // namespace pymomentum
