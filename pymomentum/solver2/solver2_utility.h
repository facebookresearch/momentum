/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/fwd.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <optional>
#include <vector>

namespace pymomentum {

std::string getDimStr(const pybind11::array& array);

std::string getDimStr(const std::vector<int>& dims, const std::vector<std::string>& dimNames);

class ArrayShapeValidator {
 public:
  void validate(
      const pybind11::array& array,
      const std::string& name,
      const std::vector<int>& expectedShape,
      const std::vector<std::string>& expectedNames);

  void validate(
      const std::optional<pybind11::array>& array,
      const std::string& name,
      const std::vector<int>& expectedShape,
      const std::vector<std::string>& expectedNames);

 private:
  std::unordered_map<int, int64_t> boundShapes_;
};

momentum::ParameterSet
arrayToParameterSet(const pybind11::array_t<bool>& array, size_t nParameters, bool defaultValue);

momentum::ParameterSet arrayToParameterSet(
    const pybind11::array_t<bool>& array,
    const momentum::ParameterTransform& parameterTransform,
    bool defaultValue);

Eigen::VectorXf arrayToVec(
    const pybind11::array_t<float>& array,
    pybind11::ssize_t expectedSize,
    const char* parameterName);

Eigen::VectorXf arrayToVec(
    const std::optional<pybind11::array_t<float>>& array,
    pybind11::ssize_t expectedSize,
    float defaultValue,
    const char* parameterName);

momentum::ParameterSet arrayToParameterSet(
    const std::optional<pybind11::array_t<bool>>& array,
    const momentum::ParameterTransform& transform,
    bool defaultValue);

void validateJointIndex(int jointIndex, const char* name, const momentum::Skeleton& skeleton);

void validateJointIndex(
    const pybind11::array_t<int>& jointIndex,
    const char* name,
    const momentum::Character& character);

void validateJointIndex(
    const pybind11::array_t<int>& jointIndex,
    const char* name,
    const momentum::Skeleton& skeleton);

void validateVertexIndex(int vertexIndex, const char* name, const momentum::Character& character);

void validateVertexIndex(int vertexIndex, const char* name, const momentum::Character* character);

void validateVertexIndex(
    const pybind11::array_t<int>& vertexIndex,
    const char* name,
    const momentum::Character& character);

void validateVertexIndex(
    const pybind11::array_t<int>& vertexIndex,
    const char* name,
    const momentum::Character* character);

void validateWeight(float weight, const char* name);

void validateWeights(const pybind11::array_t<float>& weights, const char* name);

void validateWeights(const std::optional<pybind11::array_t<float>>& weights, const char* name);

void validateWeights(const Eigen::VectorXf& weights, const char* name);

void validateWeights(const std::optional<Eigen::VectorXf>& weights, const char* name);

Eigen::VectorXf getJointWeights(
    const std::optional<pybind11::array_t<float>>& weights,
    const momentum::Skeleton& character,
    const char* name);
Eigen::VectorXf getJointWeights(
    const pybind11::array_t<float>& weights,
    const momentum::Skeleton& skeleton,
    const char* name);

void validateErrorFunctionMatchesCharacter(
    const momentum::SkeletonSolverFunction& solverFunction,
    const momentum::SkeletonErrorFunction& errorFunction);

momentum::TransformList toTransformList(const pybind11::array_t<float>& array);

momentum::ModelParameters toModelParameters(
    const pybind11::array_t<float>& array,
    const momentum::ParameterTransform& pt);

// Convert vector of floats to quaternion, using the correct (x, y, z, w) order.
Eigen::Quaternionf toQuaternion(const Eigen::Vector4f& q);

} // namespace pymomentum
