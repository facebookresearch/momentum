/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/math/mesh.h>
#include <pymomentum/solver2/solver2_utility.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

std::string getDimStr(const py::array& array) {
  std::ostringstream oss;
  oss << "[";
  bool first = true;
  for (py::ssize_t iDim = 0; iDim < array.ndim(); ++iDim) {
    auto dim = array.shape(iDim);
    if (!first) {
      oss << ", ";
    }
    first = false;
    oss << dim;
  }
  oss << "]";
  return oss.str();
}

std::string getDimStr(
    const std::vector<int>& dims,
    const std::vector<std::string>& dimNames) {
  std::ostringstream oss;
  oss << "[";
  bool first = true;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (!first) {
      oss << ", ";
    }
    first = false;

    if (dims[i] < 0 && i < dimNames.size()) {
      oss << dimNames[i];
    } else {
      oss << dims[i];
    }
  }
  oss << "]";
  return oss.str();
}

void ArrayShapeValidator::validate(
    const std::optional<py::array>& array,
    const std::string& name,
    const std::vector<int>& expectedShape,
    const std::vector<std::string>& expectedNames) {
  if (!array.has_value()) {
    return;
  }

  validate(array.value(), name, expectedShape, expectedNames);
}

void ArrayShapeValidator::validate(
    const py::array& array,
    const std::string& name,
    const std::vector<int>& expectedShape,
    const std::vector<std::string>& expectedNames) {
  if (array.ndim() != expectedShape.size()) {
    throw std::runtime_error(
        "Invalid shape for " + name + ": expected " +
        getDimStr(expectedShape, expectedNames) + ", got " + getDimStr(array));
  }

  for (size_t i = 0; i < expectedShape.size(); ++i) {
    if (expectedShape[i] >= 0) {
      if (array.shape(i) != expectedShape[i]) {
        throw std::runtime_error(
            "Invalid shape for " + name + ": expected " +
            getDimStr(expectedShape, expectedNames) + ", got " +
            getDimStr(array));
      }
    } else if (expectedShape[i] < 0) {
      auto bindingIdx = expectedShape[i];
      auto itr = boundShapes_.find(bindingIdx);
      if (itr == boundShapes_.end()) {
        boundShapes_.emplace(bindingIdx, array.shape(i));
      } else if (itr->second != array.shape(i)) {
        throw std::runtime_error(
            "Invalid shape for " + name + ": expected " +
            getDimStr(expectedShape, expectedNames) + ", got " +
            getDimStr(array));
      }
    }
  }
};

mm::ParameterSet arrayToParameterSet(
    const py::array_t<bool>& array,
    const size_t nParameters,
    bool defaultValue) {
  if (nParameters > mm::kMaxModelParams) {
    throw std::runtime_error(
        "Parameter set size exceeds maximum allowed size of " +
        std::to_string(mm::kMaxModelParams));
  }

  if (array.size() == 0) {
    mm::ParameterSet result;
    for (size_t i = 0; i < nParameters; ++i) {
      result.set(i, defaultValue);
    }
    return result;
  }

  if (array.ndim() != 1) {
    throw std::runtime_error("Expected a 1D array for parameter set");
  }

  if (array.shape(0) != nParameters) {
    throw std::runtime_error(
        "Parameter set size does not match parameter transform, expected " +
        std::to_string(nParameters) + " but got " +
        std::to_string(array.shape(0)));
  }

  auto a = array.unchecked<1>();
  mm::ParameterSet result;
  for (size_t i = 0; i < array.shape(0); ++i) {
    result.set(i, a(i) != 0);
  }
  return result;
}

Eigen::VectorXf arrayToVec(
    const py::array_t<float>& array,
    py::ssize_t expectedSize,
    const char* parameterName) {
  if (array.ndim() != 1) {
    throw std::runtime_error(
        "Expected a 1D array for " + std::string(parameterName) + "; got " +
        getDimStr(array));
  }

  if (expectedSize >= 0 && array.shape(0) != expectedSize) {
    throw std::runtime_error(
        "Invalid size for " + std::string(parameterName) + "; expected " +
        std::to_string(expectedSize) + " but got " + getDimStr(array));
  }

  auto a = array.unchecked<1>();
  Eigen::VectorXf result(array.shape(0));
  for (size_t i = 0; i < array.shape(0); ++i) {
    result(i) = a(i);
  }
  return result;
}

mm::ParameterSet arrayToParameterSet(
    const py::array_t<bool>& array,
    const mm::ParameterTransform& parameterTransform,
    bool defaultValue) {
  return arrayToParameterSet(
      array, parameterTransform.numAllModelParameters(), defaultValue);
}

Eigen::VectorXf arrayToVec(
    const std::optional<py::array_t<float>>& array,
    py::ssize_t expectedSize,
    float defaultValue,
    const char* parameterName) {
  if (!array.has_value()) {
    return Eigen::VectorXf::Constant(expectedSize, defaultValue);
  }

  return arrayToVec(array.value(), expectedSize, parameterName);
}

mm::ParameterSet arrayToParameterSet(
    const std::optional<py::array_t<bool>>& array,
    const momentum::ParameterTransform& transform,
    bool defaultValue) {
  if (!array.has_value()) {
    mm::ParameterSet result;
    for (size_t i = 0; i < transform.numAllModelParameters(); ++i) {
      result.set(i, defaultValue);
    }
    return result;
  }

  return arrayToParameterSet(array.value(), transform, defaultValue);
}

void validateIndexArray(
    const py::array_t<int>& indexArray,
    const char* name,
    const char* type,
    size_t maxIndex) {
  auto validateIndex = [&](int idx) {
    if (idx < 0 || idx >= maxIndex) {
      throw std::runtime_error(
          "Invalid " + std::string(type) + " for " + name + ": " +
          std::to_string(idx) + "; expected a value between 0 and " +
          std::to_string(maxIndex));
    }
  };

  if (indexArray.ndim() == 1) {
    auto a = indexArray.unchecked<1>();
    for (py::ssize_t i = 0; i < indexArray.shape(0); ++i) {
      validateIndex(a(i));
    }
  } else if (indexArray.ndim() == 2) {
    auto a = indexArray.unchecked<2>();
    for (py::ssize_t i = 0; i < indexArray.shape(0); ++i) {
      for (py::ssize_t j = 0; j < indexArray.shape(1); ++j) {
        validateIndex(a(i, j));
      }
    }
  } else {
    throw std::runtime_error(
        "Invalid " + std::string(name) +
        " array; expected 1D or 2D array, got " + getDimStr(indexArray));
  }
}

void validateJointIndex(
    int jointIndex,
    const char* name,
    const mm::Skeleton& skeleton) {
  if (jointIndex < 0) {
    throw std::runtime_error(
        std::string("Invalid ") + name +
        " index: " + std::to_string(jointIndex));
  }

  if (jointIndex >= static_cast<int>(skeleton.joints.size())) {
    throw std::runtime_error(
        std::string("Invalid ") + name +
        " index: " + std::to_string(jointIndex) + "; skeleton has only " +
        std::to_string(skeleton.joints.size()) + " joints");
  }
}

void validateJointIndex(
    const py::array_t<int>& jointIndex,
    const char* name,
    const mm::Skeleton& skeleton) {
  validateIndexArray(jointIndex, name, "joint index", skeleton.joints.size());
}

void validateJointIndex(
    const py::array_t<int>& jointIndex,
    const char* name,
    const mm::Character& character) {
  validateJointIndex(jointIndex, name, character.skeleton);
}

void validateVertexIndex(
    int vertexIndex,
    const char* name,
    const momentum::Character& character) {
  if (!character.mesh) {
    throw std::runtime_error(
        "Character does not have a mesh; cannot validate " + std::string(name));
  }

  if (vertexIndex < 0 || vertexIndex >= character.mesh->vertices.size()) {
    throw std::runtime_error(
        "Invalid " + std::string(name) + " index: " +
        std::to_string(vertexIndex) + "; expected a value between 0 and " +
        std::to_string(character.mesh->vertices.size()));
  }
}

void validateVertexIndex(
    const pybind11::array_t<int>& vertexIndex,
    const char* name,
    const momentum::Character& character) {
  if (!character.mesh) {
    throw std::runtime_error(
        "Character does not have a mesh; cannot validate " + std::string(name));
  }

  validateIndexArray(
      vertexIndex, name, "vertex index", character.mesh->vertices.size());
}

mm::TransformList toTransformList(const py::array_t<float>& array) {
  if (array.ndim() != 2 || array.shape(1) != 8) {
    throw std::runtime_error(
        "Expected (nJoints x 8) skeleton state tensor; got " +
        getDimStr(array));
  }

  const auto nTransforms = array.shape(0);

  mm::TransformList result(nTransforms);

  auto acc = array.unchecked<2>();

  for (py::ssize_t i = 0; i < nTransforms; ++i) {
    Eigen::Vector3f position(acc(i, 0), acc(i, 1), acc(i, 2));
    // Quaternions in pymomentum are: (x=3, y=4, z=5, w=6)
    // Eigen quaternion constructor takes (w, x, y, z)
    Eigen::Quaternionf rotation(acc(i, 6), acc(i, 3), acc(i, 4), acc(i, 5));
    float scale = acc(i, 7);

    result.at(i) = mm::Transform(position, rotation.normalized(), scale);
  }

  return result;
}

momentum::ModelParameters toModelParameters(
    const py::array_t<float>& array,
    const mm::ParameterTransform& pt) {
  if (array.ndim() != 1) {
    throw std::runtime_error(
        "Expected a 1D array for model parameters; got " + getDimStr(array));
  }

  const auto nParams = array.shape(0);

  if (nParams != pt.numAllModelParameters()) {
    throw std::runtime_error(
        "Invalid size for model parameters; expected " +
        std::to_string(pt.numAllModelParameters()) + " but got " +
        getDimStr(array));
  }

  auto a = array.unchecked<1>();
  momentum::ModelParameters result(nParams);
  for (size_t i = 0; i < nParams; ++i) {
    result(i) = a(i);
  }
  return result;
}

Eigen::Quaternionf toQuaternion(const Eigen::Vector4f& q) {
  return Eigen::Quaternionf(q(3), q(0), q(1), q(2)).normalized();
}

} // namespace pymomentum
