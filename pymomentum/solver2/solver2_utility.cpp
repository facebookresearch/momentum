/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/skeleton_solver_function.h>

#include <momentum/math/mesh.h>
#include <pymomentum/solver2/solver2_utility.h>

#include <fmt/format.h>
#include <pybind11/eigen.h>
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

std::string getDimStr(const std::vector<int>& dims, const std::vector<std::string>& dimNames) {
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
    throw std::runtime_error(fmt::format(
        "Invalid shape for {}: expected {}, got {}",
        name,
        getDimStr(expectedShape, expectedNames),
        getDimStr(array)));
  }

  for (size_t i = 0; i < expectedShape.size(); ++i) {
    if (expectedShape[i] >= 0) {
      if (array.shape(i) != expectedShape[i]) {
        throw std::runtime_error(fmt::format(
            "Invalid shape for {}: expected {}, got {}",
            name,
            getDimStr(expectedShape, expectedNames),
            getDimStr(array)));
      }
    } else if (expectedShape[i] < 0) {
      auto bindingIdx = expectedShape[i];
      auto itr = boundShapes_.find(bindingIdx);
      if (itr == boundShapes_.end()) {
        boundShapes_.emplace(bindingIdx, array.shape(i));
      } else if (itr->second != array.shape(i)) {
        throw std::runtime_error(fmt::format(
            "Invalid shape for {}: expected {}, got {}",
            name,
            getDimStr(expectedShape, expectedNames),
            getDimStr(array)));
      }
    }
  }
}

mm::ParameterSet
arrayToParameterSet(const py::array_t<bool>& array, const size_t nParameters, bool defaultValue) {
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
    throw std::runtime_error(fmt::format(
        "Parameter set size does not match parameter transform, expected {} but got {}",
        nParameters,
        array.shape(0)));
  }

  auto a = array.unchecked<1>();
  mm::ParameterSet result;
  for (size_t i = 0; i < array.shape(0); ++i) {
    result.set(i, a(i) != 0);
  }
  return result;
}

Eigen::VectorXf
arrayToVec(const py::array_t<float>& array, py::ssize_t expectedSize, const char* parameterName) {
  if (array.ndim() != 1) {
    throw std::runtime_error(
        fmt::format("Expected a 1D array for {}; got {}", parameterName, getDimStr(array)));
  }

  if (expectedSize >= 0 && array.shape(0) != expectedSize) {
    throw std::runtime_error(fmt::format(
        "Invalid size for {}; expected {} but got {}",
        parameterName,
        expectedSize,
        getDimStr(array)));
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
  return arrayToParameterSet(array, parameterTransform.numAllModelParameters(), defaultValue);
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
      throw std::runtime_error(fmt::format(
          "Invalid {} for {}: {}; expected a value between 0 and {}", type, name, idx, maxIndex));
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
    throw std::runtime_error(fmt::format(
        "Invalid {} array; expected 1D or 2D array, got {}", name, getDimStr(indexArray)));
  }
}

void validateJointIndex(int jointIndex, const char* name, const mm::Skeleton& skeleton) {
  if (jointIndex < 0) {
    throw std::runtime_error(fmt::format("Invalid {} index: {}", name, jointIndex));
  }

  if (jointIndex >= static_cast<int>(skeleton.joints.size())) {
    throw std::runtime_error(fmt::format(
        "Invalid {} index: {}; skeleton has only {} joints",
        name,
        jointIndex,
        skeleton.joints.size()));
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

void validateVertexIndex(int vertexIndex, const char* name, const momentum::Character& character) {
  if (!character.mesh) {
    throw std::runtime_error(
        fmt::format("Character does not have a mesh; cannot validate {}", name));
  }

  if (vertexIndex < 0 || vertexIndex >= character.mesh->vertices.size()) {
    throw std::runtime_error(fmt::format(
        "Invalid {} index: {}; expected a value between 0 and {}",
        name,
        vertexIndex,
        character.mesh->vertices.size()));
  }
}

void validateVertexIndex(
    const pybind11::array_t<int>& vertexIndex,
    const char* name,
    const momentum::Character& character) {
  if (!character.mesh) {
    throw std::runtime_error(
        fmt::format("Character does not have a mesh; cannot validate {}", name));
  }

  validateIndexArray(vertexIndex, name, "vertex index", character.mesh->vertices.size());
}

mm::TransformList toTransformList(const py::array_t<float>& array) {
  if (array.ndim() != 2 || array.shape(1) != 8) {
    throw std::runtime_error(
        fmt::format("Expected (nJoints x 8) skeleton state tensor; got {}", getDimStr(array)));
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
        fmt::format("Expected a 1D array for model parameters; got {}", getDimStr(array)));
  }

  const auto nParams = array.shape(0);

  if (nParams != pt.numAllModelParameters()) {
    throw std::runtime_error(fmt::format(
        "Invalid size for model parameters; expected {} but got {}",
        pt.numAllModelParameters(),
        getDimStr(array)));
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

void validateWeight(float weight, const char* name) {
  if (weight < 0.0f) {
    throw py::value_error(fmt::format("Invalid {}: {}; weight must be non-negative", name, weight));
  }
}

void validateWeights(const py::array_t<float>& weights, const char* name) {
  auto weightsAcc = weights.unchecked<1>();
  for (py::ssize_t i = 0; i < weights.shape(0); ++i) {
    if (weightsAcc(i) < 0.0f) {
      throw py::value_error(fmt::format(
          "Invalid {} at index {}: {}; all weights must be non-negative", name, i, weightsAcc(i)));
    }
  }
}

void validateWeights(const std::optional<py::array_t<float>>& weights, const char* name) {
  if (weights.has_value()) {
    validateWeights(weights.value(), name);
  }
}

void validateWeights(const Eigen::VectorXf& weights, const char* name) {
  for (int i = 0; i < weights.size(); ++i) {
    if (weights(i) < 0.0f) {
      throw py::value_error(fmt::format(
          "Invalid {} at index {}: {}; all weights must be non-negative", name, i, weights(i)));
    }
  }
}

void validateWeights(const std::optional<Eigen::VectorXf>& weights, const char* name) {
  if (weights.has_value()) {
    validateWeights(weights.value(), name);
  }
}

void validateErrorFunctionMatchesCharacter(
    const momentum::SkeletonSolverFunction& solverFunction,
    const momentum::SkeletonErrorFunction& errorFunction) {
  if (solverFunction.getSkeleton() != &errorFunction.getSkeleton()) {
    throw std::runtime_error(
        "Skeleton in solver function does not match skeleton in error function; did you use the correct Character when constructing it?");
  }

  if (solverFunction.getParameterTransform() != &errorFunction.getParameterTransform()) {
    throw std::runtime_error(
        "Parameter transform in solver function does not match parameter transform in error function; did you use the correct Character when constructing it?");
  }
}

Eigen::VectorXf getJointWeights(
    const std::optional<pybind11::array_t<float>>& weights,
    const momentum::Skeleton& character,
    const char* name) {
  if (!weights.has_value()) {
    return Eigen::VectorXf::Ones(character.joints.size());
  }

  return getJointWeights(weights.value(), character, name);
}

Eigen::VectorXf getJointWeights(
    const pybind11::array_t<float>& weights,
    const momentum::Skeleton& skeleton,
    const char* name) {
  MT_THROW_IF_T(weights.ndim() != 1, py::value_error, "{} weights must be a 1D array.", name);

  auto wAcc = weights.unchecked<1>();
  MT_THROW_IF_T(
      wAcc.shape(0) != skeleton.joints.size(),
      py::value_error,
      "{} weights size does not match the number of joints in the skeleton, expected {} but got {}.",
      name,
      skeleton.joints.size(),
      wAcc.shape(0));
  Eigen::VectorXf result(wAcc.shape(0));
  for (py::ssize_t i = 0; i < wAcc.shape(0); ++i) {
    result[i] = wAcc(i);
  }
  return result;
}

} // namespace pymomentum
