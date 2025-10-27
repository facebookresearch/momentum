/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pymomentum/axel/axel_utility.h>

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <sstream>

namespace py = pybind11;

namespace pymomentum {

std::string getArrayDimStr(const py::array& array) {
  std::ostringstream oss;
  oss << "(";
  for (py::ssize_t i = 0; i < array.ndim(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << array.shape(i);
  }
  oss << ")";
  return oss.str();
}

void validatePositionArray(const py::array_t<float>& positions, const char* parameterName) {
  if (positions.ndim() != 2) {
    throw std::runtime_error(
        fmt::format(
            "Invalid shape for {}: expected 2D array (N, 3), got {}D array {}",
            parameterName,
            positions.ndim(),
            getArrayDimStr(positions)));
  }

  if (positions.shape(1) != 3) {
    throw std::runtime_error(
        fmt::format(
            "Invalid shape for {}: expected (N, 3), got {}",
            parameterName,
            getArrayDimStr(positions)));
  }
}

void validateGridDimensions(
    const py::array_t<float>& grid,
    const axel::SignedDistanceField<float>& sdf,
    const char* parameterName) {
  if (grid.ndim() != 3) {
    throw std::runtime_error(
        fmt::format(
            "Invalid shape for {}: expected 3D array (nx, ny, nz), got {}D array {}",
            parameterName,
            grid.ndim(),
            getArrayDimStr(grid)));
  }

  const auto& resolution = sdf.resolution();
  if (grid.shape(0) != resolution.x() || grid.shape(1) != resolution.y() ||
      grid.shape(2) != resolution.z()) {
    throw std::runtime_error(
        fmt::format(
            "Invalid shape for {}: expected ({}, {}, {}) to match SDF resolution, got {}",
            parameterName,
            resolution.x(),
            resolution.y(),
            resolution.z(),
            getArrayDimStr(grid)));
  }
}

Eigen::Vector3f arrayToVector3f(const py::array_t<float>& array, const char* parameterName) {
  if (array.ndim() != 1) {
    throw std::runtime_error(
        fmt::format(
            "Invalid shape for {}: expected 1D array (3,), got {}D array {}",
            parameterName,
            array.ndim(),
            getArrayDimStr(array)));
  }

  if (array.shape(0) != 3) {
    throw std::runtime_error(
        fmt::format(
            "Invalid shape for {}: expected (3,), got {}", parameterName, getArrayDimStr(array)));
  }

  auto acc = array.unchecked<1>();
  return {acc(0), acc(1), acc(2)};
}

Eigen::Vector3<axel::Index> arrayToVector3i(
    const py::array_t<axel::Index>& array,
    const char* parameterName) {
  if (array.ndim() != 1) {
    throw std::runtime_error(
        fmt::format(
            "Invalid shape for {}: expected 1D array (3,), got {}D array {}",
            parameterName,
            array.ndim(),
            getArrayDimStr(array)));
  }

  if (array.shape(0) != 3) {
    throw std::runtime_error(
        fmt::format(
            "Invalid shape for {}: expected (3,), got {}", parameterName, getArrayDimStr(array)));
  }

  auto acc = array.unchecked<1>();
  return {acc(0), acc(1), acc(2)};
}

py::array_t<float> vector3fToArray(const Eigen::Vector3f& vec) {
  auto result = py::array_t<float>(3);
  auto acc = result.mutable_unchecked<1>();
  acc(0) = vec.x();
  acc(1) = vec.y();
  acc(2) = vec.z();
  return result;
}

py::array_t<axel::Index> vector3iToArray(const Eigen::Vector3<axel::Index>& vec) {
  auto result = py::array_t<axel::Index>(3);
  auto acc = result.mutable_unchecked<1>();
  acc(0) = vec.x();
  acc(1) = vec.y();
  acc(2) = vec.z();
  return result;
}

void validateBatchPositionArray(const py::array_t<float>& positions, const char* parameterName) {
  if (positions.ndim() == 1) {
    // Single position: shape (3,)
    if (positions.shape(0) != 3) {
      throw std::runtime_error(
          fmt::format(
              "Invalid shape for {}: expected (3,) for single position or (N, 3) for batch of positions, got {}",
              parameterName,
              getArrayDimStr(positions)));
    }
  } else if (positions.ndim() == 2) {
    // Batch of positions: shape (N, 3)
    if (positions.shape(1) != 3) {
      throw std::runtime_error(
          fmt::format(
              "Invalid shape for {}: expected (N, 3) for batch of positions, got {}",
              parameterName,
              getArrayDimStr(positions)));
    }
  } else {
    throw std::runtime_error(
        fmt::format(
            "Invalid shape for {}: expected 1D array (3,) for single position or 2D array (N, 3) for batch of positions, got {}D array {}",
            parameterName,
            positions.ndim(),
            getArrayDimStr(positions)));
  }
}

// Template implementations are now in the header file

} // namespace pymomentum
