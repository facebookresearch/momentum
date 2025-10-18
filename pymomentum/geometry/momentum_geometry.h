/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/math/mppca.h>

#include <ATen/ATen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>

namespace pymomentum {

using RowMatrixf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixb = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

at::Tensor mapModelParameters_names(
    at::Tensor motion_in,
    const std::vector<std::string>& srcParameterNames,
    const momentum::Character& tgtCharacter,
    bool verbose);

at::Tensor mapModelParameters(
    at::Tensor motion_in,
    const momentum::Character& srcCharacter,
    const momentum::Character& tgtCharacter,
    bool verbose);

at::Tensor mapJointParameters(
    at::Tensor motion_in,
    const momentum::Character& srcCharacter,
    const momentum::Character& tgtCharacter);

momentum::Character loadFBXCharacterFromFile(
    const std::string& fbxPath,
    const std::optional<std::string>& configPath = {},
    const std::optional<std::string>& locatorsPath = {},
    bool permissive = false);
std::tuple<momentum::Character, std::vector<Eigen::MatrixXf>, float>
loadFBXCharacterWithMotionFromFile(const std::string& fbxPath, bool permissive = false);

std::tuple<momentum::Character, std::vector<Eigen::MatrixXf>, float>
loadFBXCharacterWithMotionFromBytes(const pybind11::bytes& fbxBytes, bool permissive = false);

momentum::Character loadLocatorsFromFile(
    const momentum::Character& character,
    const std::string& locatorsPath);
momentum::Character loadConfigFromFile(
    const momentum::Character& character,
    const std::string& configPath);

momentum::Character loadFBXCharacterFromBytes(
    const pybind11::bytes& bytes,
    bool permissive = false);
momentum::Character loadLocatorsFromBytes(
    const momentum::Character& character,
    const pybind11::bytes& bytes);
momentum::Character loadConfigFromBytes(
    const momentum::Character& character,
    const pybind11::bytes& bytes);

momentum::Character loadURDFCharacterFromFile(const std::string& urdfPath);
momentum::Character loadURDFCharacterFromBytes(const pybind11::bytes& urdfBytes);

// Convert uniform noise to meaningful model parameters.
// unifNoise: size: batchSize (optional) x #modelParameters, each entry
//   range in [0, 1].
// Return a Tensor of size batchSize (optional) x #modelParameters.
//   Scale parameters range in [-0.5, 0.5].
//   Translate parameters range in [-2.5, 2.5].
//   Rotation angles range in [-Pi/8, Pi/8].
at::Tensor uniformRandomToModelParameters(
    const momentum::Character& character,
    at::Tensor unifNoise);

std::shared_ptr<const momentum::Mppca> loadPosePriorFromFile(const std::string& path);
void savePosePriorToFile(const momentum::Mppca& mppca, const std::string& path);
std::shared_ptr<const momentum::Mppca> loadPosePriorFromBytes(const pybind11::bytes& bytes);

std::shared_ptr<momentum::BlendShape> loadBlendShapeFromFile(
    const std::string& path,
    int nExpectedShapes = -1,
    int nExpectedVertices = -1);
std::shared_ptr<momentum::BlendShape> loadBlendShapeFromBytes(
    const pybind11::bytes& bytes,
    int nExpectedShapes = -1,
    int nExpectedVertices = -1);
pybind11::bytes saveBlendShapeToBytes(const momentum::BlendShape& blendShape);
void saveBlendShapeToFile(const momentum::BlendShape& blendShape, const std::string& path);

std::shared_ptr<momentum::BlendShape> loadBlendShapeFromTensors(
    pybind11::array_t<float> baseShape,
    pybind11::array_t<float> shapeVectors);

// Strips out vertices attached to the lower body.
// Does not actually change the skeleton, this is so you can
// simply apply the same motion to the new character as you did to the
// old character.
momentum::Character stripLowerBodyVertices(const momentum::Character& character);

std::vector<size_t> getUpperBodyJoints(const momentum::Skeleton& skeleton);

std::tuple<float, Eigen::VectorXf, Eigen::MatrixXf, float> getMppcaModel(
    const momentum::Mppca& mppca,
    int iModel);

std::shared_ptr<momentum::Mppca> createMppcaModel(
    const Eigen::VectorXd& pi,
    const Eigen::MatrixXd& mu,
    const pybind11::array_t<double>& W,
    const Eigen::VectorXd& sigma,
    const std::vector<std::string>& parameterNames);

// Flatten a vector of Eigen::Vector3f or similar into an Eigen::MatrixXf,
// which can then be converted to a numpy array by pybind11's automatic
// conversion.
template <typename T, int N>
pybind11::array_t<T> asArray(const std::vector<Eigen::Matrix<T, N, 1>>& v) {
  pybind11::array_t<T> result(
      {static_cast<pybind11::ssize_t>(v.size()), static_cast<pybind11::ssize_t>(N)});
  auto r = result.template mutable_unchecked<2>();
  for (size_t i = 0; i < v.size(); ++i) {
    for (int j = 0; j < N; ++j) {
      r(i, j) = v[i](j);
    }
  }
  return result;
}

template <typename T, int N, typename Derived>
std::vector<Eigen::Matrix<T, N, 1>> asVectorList(const Eigen::DenseBase<Derived>& b) {
  std::vector<Eigen::Matrix<T, N, 1>> result;
  if (b.size() == 0) {
    return result;
  }

  MT_THROW_IF(b.cols() != N, "Expected a matrix with {} columns, but got {}", N, b.cols());

  result.reserve(b.rows());
  for (Eigen::Index i = 0; i < b.rows(); ++i) {
    result.emplace_back(b.row(i).template cast<T>());
  }
  return result;
}

template <typename T, int N>
std::vector<Eigen::Vector<T, N>> asVectorList(const pybind11::array_t<T>& b) {
  std::vector<Eigen::Vector<T, N>> result;
  if (b.shape(0) == 0) {
    return result;
  }

  MT_THROW_IF(b.shape(1) != N, "Expected a matrix with {} columns, but got {}", N, b.shape(1));

  result.reserve(b.shape(0));
  auto r = b.template unchecked<2>();
  for (pybind11::ssize_t i = 0; i < b.shape(0); ++i) {
    Eigen::Vector<T, N> v;
    for (pybind11::ssize_t j = 0; j < N; ++j) {
      v(j) = r(i, j);
    }
    result.push_back(v);
  }
  return result;
}

std::unique_ptr<momentum::Mesh> getPosedMesh(
    const momentum::Character& character,
    Eigen::Ref<const Eigen::VectorXf> jointParameters);

momentum::Character replaceRestMesh(const momentum::Character& character, RowMatrixf positions);

std::vector<bool> jointListToBitset(
    const momentum::Character& character,
    const std::vector<int>& jointIndices);
std::vector<int> bitsetToJointList(const std::vector<bool>& jointMask);

/// Matches the locator names from the available locators in the character
/// object and returns parents and offsets for the each of the locator names
/// @param[in] character A momentum character object with locators
/// @param[in] names A vector of locator names
/// @return A tuple of (locator_parents, locator_offsets)
/// locator_parents has the same size as names vector and contains parent
/// index of the locator if found otherwise -1. locator_offsets is a matrix of
/// size (names.size(), 3) containing the offset of the locator wrt locators'
/// parent joint
std::tuple<Eigen::VectorXi, RowMatrixf> getLocators(
    const momentum::Character& character,
    const std::vector<std::string>& names);

// Clamp body model params to be within limits applied by
// character.parameterLimits and return clamped model params.
// Work with batched model params. The returned tensor's batch
// dimension is dependent on the input tensor.
// Differentiable, support both float and double dtypes.
at::Tensor applyModelParameterLimits(const momentum::Character& character, at::Tensor modelParams);

std::tuple<Eigen::VectorXf, Eigen::VectorXf> modelParameterLimits(
    const momentum::Character& character);

using MatrixX7f =
    Eigen::Matrix<float, Eigen::Dynamic, momentum::kParametersPerJoint, Eigen::RowMajor>;

std::tuple<MatrixX7f, MatrixX7f> jointParameterLimits(const momentum::Character& character);

pybind11::array_t<float> getBindPose(const momentum::Character& character);
pybind11::array_t<float> getInverseBindPose(const momentum::Character& character);

std::string formatDimensions(const pybind11::array& array);

std::vector<bool> boolArrayToVector(const pybind11::array_t<bool>& array);

} // namespace pymomentum
