/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <axel/BoundingBox.h>
#include <axel/SignedDistanceField.h>
#include <axel/common/Types.h>

#include <fmt/format.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Core>

namespace py = pybind11;

namespace pymomentum {

/**
 * Helper function to get a string representation of array dimensions
 */
std::string getArrayDimStr(const py::array& array);

/**
 * Validate array shape for 3D positions.
 * Expected shape: (N, 3)
 */
void validatePositionArray(const py::array_t<float>& positions, const char* parameterName);

/**
 * Validate array shape for 3D indices (templated version).
 * Expected shape: (N, 3) for N triplets or (3,) for single triplet
 *
 * @tparam T Index type (e.g., int, axel::Index)
 */
template <typename T>
void validateIndexArray(const py::array_t<T>& indices, const char* parameterName) {
  if (indices.ndim() == 1) {
    if (indices.shape(0) != 3) {
      throw std::runtime_error(fmt::format(
          "Invalid shape for {}: expected (3,) for single index or (N, 3) for multiple indices, got {}",
          parameterName,
          getArrayDimStr(indices)));
    }
  } else if (indices.ndim() == 2) {
    if (indices.shape(1) != 3) {
      throw std::runtime_error(fmt::format(
          "Invalid shape for {}: expected (N, 3), got {}", parameterName, getArrayDimStr(indices)));
    }
  } else {
    throw std::runtime_error(fmt::format(
        "Invalid shape for {}: expected 1D array (3,) or 2D array (N, 3), got {}D array {}",
        parameterName,
        indices.ndim(),
        getArrayDimStr(indices)));
  }
}

/**
 * Validate array dimensions for grid coordinates.
 * Expected shape: (nx, ny, nz) or compatible with SDF resolution
 */
void validateGridDimensions(
    const py::array_t<float>& grid,
    const axel::SignedDistanceField<float>& sdf,
    const char* parameterName);

/**
 * Convert py::array_t<float> to Eigen::Vector3f.
 * Input should be shape (3,) or compatible.
 */
Eigen::Vector3f arrayToVector3f(const py::array_t<float>& array, const char* parameterName);

/**
 * Convert py::array_t<axel::Index> to Eigen::Vector3<axel::Index>.
 * Input should be shape (3,) or compatible.
 */
Eigen::Vector3<axel::Index> arrayToVector3i(
    const py::array_t<axel::Index>& array,
    const char* parameterName);

/**
 * Convert Eigen::Vector3f to py::array_t<float>.
 */
py::array_t<float> vector3fToArray(const Eigen::Vector3f& vec);

/**
 * Convert Eigen::Vector3<axel::Index> to py::array_t<axel::Index>.
 */
py::array_t<axel::Index> vector3iToArray(const Eigen::Vector3<axel::Index>& vec);

/**
 * Validate input positions array for batching operations.
 * Input can be either:
 * - 1D array of shape (3,) for single position
 * - 2D array of shape (N, 3) for batch of N positions
 */
void validateBatchPositionArray(const py::array_t<float>& positions, const char* parameterName);

/**
 * Specialized version for operations that return scalar values (like sample).
 */
template <typename SdfType, typename Operation>
py::array_t<float> applyBatchedScalarOperation(
    const SdfType& sdf,
    const py::array_t<float>& positions,
    Operation operation) {
  validateBatchPositionArray(positions, "positions");

  if (positions.ndim() == 1) {
    // Single position case: return scalar in 0D array
    Eigen::Vector3f pos(positions.data()[0], positions.data()[1], positions.data()[2]);
    float result = operation(sdf, pos);

    // For single position, return a scalar wrapped in an array
    auto resultArray = py::array_t<float>(1);
    *resultArray.mutable_data() = result;
    resultArray.resize({});
    return resultArray;
  } else {
    // Batch case: return 1D array of results
    const size_t numPositions = positions.shape(0);
    auto resultArray = py::array_t<float>(numPositions);
    auto resultData = resultArray.template mutable_unchecked<1>();
    auto posData = positions.unchecked<2>();

    for (size_t i = 0; i < numPositions; ++i) {
      Eigen::Vector3f pos(posData(i, 0), posData(i, 1), posData(i, 2));
      resultData(i) = operation(sdf, pos);
    }

    return resultArray;
  }
}

/**
 * Specialized version for operations that return Vector3f values (like
 * gradient).
 */
template <typename SdfType, typename Operation>
py::array_t<float> applyBatchedVectorOperation(
    const SdfType& sdf,
    const py::array_t<float>& positions,
    Operation operation) {
  validateBatchPositionArray(positions, "positions");

  if (positions.ndim() == 1) {
    // Single position case: return 1D array of shape (3,)
    Eigen::Vector3f pos(positions.data()[0], positions.data()[1], positions.data()[2]);
    Eigen::Vector3f result = operation(sdf, pos);
    return vector3fToArray(result);
  } else {
    // Batch case: return 2D array of shape (N, 3)
    const size_t numPositions = positions.shape(0);
    std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(numPositions), 3};
    auto resultArray = py::array_t<float>(shape);
    auto resultData = resultArray.template mutable_unchecked<2>();
    auto posData = positions.unchecked<2>();

    for (size_t i = 0; i < numPositions; ++i) {
      Eigen::Vector3f pos(posData(i, 0), posData(i, 1), posData(i, 2));
      Eigen::Vector3f result = operation(sdf, pos);
      resultData(i, 0) = result.x();
      resultData(i, 1) = result.y();
      resultData(i, 2) = result.z();
    }

    return resultArray;
  }
}

/**
 * Specialized version for operations that return pair<float, Vector3f> (like
 * sampleWithGradient).
 */
template <typename SdfType, typename Operation>
py::tuple applyBatchedSampleGradientOperation(
    const SdfType& sdf,
    const py::array_t<float>& positions,
    Operation operation) {
  validateBatchPositionArray(positions, "positions");

  if (positions.ndim() == 1) {
    // Single position case: return tuple of (scalar, vector)
    Eigen::Vector3f pos(positions.data()[0], positions.data()[1], positions.data()[2]);
    auto [value, gradient] = operation(sdf, pos);

    return py::make_tuple(py::cast(value), vector3fToArray(gradient));
  } else {
    // Batch case: return tuple of (1D values array, 2D gradients array)
    const size_t numPositions = positions.shape(0);
    auto valuesArray = py::array_t<float>(numPositions);
    std::vector<py::ssize_t> gradientsShape = {static_cast<py::ssize_t>(numPositions), 3};
    auto gradientsArray = py::array_t<float>(gradientsShape);

    auto valuesData = valuesArray.template mutable_unchecked<1>();
    auto gradientsData = gradientsArray.template mutable_unchecked<2>();
    auto posData = positions.unchecked<2>();

    for (size_t i = 0; i < numPositions; ++i) {
      Eigen::Vector3f pos(posData(i, 0), posData(i, 1), posData(i, 2));
      auto [value, gradient] = operation(sdf, pos);
      valuesData(i) = value;
      gradientsData(i, 0) = gradient.x();
      gradientsData(i, 1) = gradient.y();
      gradientsData(i, 2) = gradient.z();
    }

    return py::make_tuple(valuesArray, gradientsArray);
  }
}

} // namespace pymomentum
