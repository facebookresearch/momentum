/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/momentum_geometry.h"

#include "pymomentum/python_utility/python_utility.h"
#include "pymomentum/tensor_momentum/tensor_parameter_transform.h"
#include "pymomentum/tensor_utility/autograd_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/character/character.h>
#include <momentum/character/character_utility.h>
#include <momentum/character/joint.h>
#include <momentum/character/linear_skinning.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/skin_weights.h>
#include <momentum/common/checks.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/shape/blend_shape_io.h>
#include <momentum/io/skeleton/locator_io.h>
#include <momentum/io/skeleton/mppca_io.h>
#include <momentum/io/skeleton/parameter_transform_io.h>
#include <momentum/io/urdf/urdf_io.h>
#include <momentum/math/constants.h>
#include <momentum/math/mesh.h>

#ifndef PYMOMENTUM_LIMITED_TORCH_API
#include <torch/csrc/jit/python/python_ivalue.h>
#endif
#include <Eigen/Core>

#include <cstdint>

namespace py = pybind11;

#ifndef PYMOMENTUM_LIMITED_TORCH_API
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
#endif

namespace pymomentum {

template <typename T>
gsl::span<const T> toSpan(const pybind11::bytes& bytes) {
  pybind11::gil_scoped_acquire acquire;
  py::buffer_info info(py::buffer(bytes).request());
  const T* data = reinterpret_cast<const T*>(info.ptr);
  const size_t length = static_cast<size_t>(info.size);

  MT_THROW_IF(data == nullptr, "Unable to extract contents from bytes.");

  return gsl::make_span<const T>(data, length);
}

// Use tensor.index_select() and tensor.index_copy() to copy data from srcTensor
// to form a new tensor. The dimension the mapping is applied on is `dimension`.
// The mapping is created by matching `srcNames` to `tgtNames`. The target slice
// which has no corresponding source is filled with zero.
at::Tensor mapTensor(
    int64_t dimension,
    const std::vector<std::string>& srcNames,
    const std::vector<std::string>& tgtNames,
    at::Tensor srcTensor) {
  if (dimension < 0) {
    dimension += srcTensor.ndimension();
  }

  MT_THROW_IF(
      dimension < 0 || srcTensor.ndimension() <= dimension ||
          srcTensor.size(dimension) != srcNames.size(),
      "Unexpected error in mapTensor(): dimensions don't match.");

  std::unordered_map<std::string, size_t> tgtNameMap;
  for (size_t iTgt = 0; iTgt < tgtNames.size(); ++iTgt) {
    tgtNameMap.insert({tgtNames[iTgt], iTgt});
  }

  const size_t expectedSize = std::min(srcNames.size(), tgtNames.size());

  std::vector<int64_t> tgtIndices;
  tgtIndices.reserve(expectedSize);
  std::vector<int64_t> srcIndices;
  srcIndices.reserve(expectedSize);

  for (size_t iSrc = 0; iSrc < srcNames.size(); ++iSrc) {
    auto itr = tgtNameMap.find(srcNames[iSrc]);
    if (itr == tgtNameMap.end()) {
      continue;
    }

    tgtIndices.push_back(itr->second);
    srcIndices.push_back(iSrc);
  }

  at::Tensor srcIndicesTensor = to1DTensor(srcIndices).to(srcTensor.device());
  at::Tensor tgtIndicesTensor = to1DTensor(tgtIndices).to(srcTensor.device());

  std::vector<int64_t> tgtSizes;
  for (int64_t i = 0; i < srcTensor.ndimension(); ++i) {
    if (i == dimension) {
      tgtSizes.push_back(tgtNames.size());
    } else {
      tgtSizes.push_back(srcTensor.size(i));
    }
  }

  at::Tensor result = at::zeros(tgtSizes, srcTensor.scalar_type());
  return result.index_copy(
      dimension, tgtIndicesTensor, srcTensor.index_select(dimension, srcIndicesTensor));
}

at::Tensor mapModelParameters(
    at::Tensor motion_in,
    const momentum::Character& srcCharacter,
    const momentum::Character& tgtCharacter,
    bool verbose) {
  return mapModelParameters_names(
      motion_in, srcCharacter.parameterTransform.name, tgtCharacter, verbose);
}

at::Tensor mapModelParameters_names(
    at::Tensor motion_in,
    const std::vector<std::string>& parameterNames_in,
    const momentum::Character& character_remap,
    bool verbose) {
  MT_THROW_IF(
      motion_in.size(-1) != parameterNames_in.size(),
      "Mismatch between motion size and parameter name count.");

  std::vector<std::string> missingParams;
  for (size_t iParamSource = 0; iParamSource < parameterNames_in.size(); ++iParamSource) {
    auto iParamRemap =
        character_remap.parameterTransform.getParameterIdByName(parameterNames_in[iParamSource]);
    if (iParamRemap == momentum::kInvalidIndex) {
      missingParams.push_back(parameterNames_in[iParamSource]);
    }
  }

  if (verbose && !missingParams.empty()) {
    // TODO better logging:
    py::print("WARNING: missing parameters found during map_model_parameters: ", missingParams);
  }

  // Map tensor at dimension -1.
  return mapTensor(-1, parameterNames_in, character_remap.parameterTransform.name, motion_in);
}

at::Tensor mapJointParameters(
    at::Tensor srcMotion,
    const momentum::Character& srcCharacter,
    const momentum::Character& tgtCharacter) {
  // Make tensor into shape: [... x n_joint_params x 7]
  bool unflattened;
  srcMotion = unflattenJointParameters(srcCharacter, srcMotion, &unflattened);

  // Map tensor at dimension -2.
  at::Tensor result = mapTensor(
      -2, srcCharacter.skeleton.getJointNames(), tgtCharacter.skeleton.getJointNames(), srcMotion);
  if (unflattened) {
    result = flattenJointParameters(tgtCharacter, result);
  }
  return result;
}

constexpr momentum::KeepLocators keepLocators = momentum::KeepLocators::Yes;

momentum::Character loadFBXCharacterFromFile(
    const std::string& fbxPath,
    const std::optional<std::string>& configPath,
    const std::optional<std::string>& locatorsPath,
    bool permissive) {
  momentum::Character result =
      momentum::loadFbxCharacter(filesystem::path(fbxPath), keepLocators, permissive);
  if (configPath && !configPath->empty()) {
    result = loadConfigFromFile(result, *configPath);
  }

  if (locatorsPath && !locatorsPath->empty()) {
    result = loadLocatorsFromFile(result, *locatorsPath);
  }

  return result;
}

void transposeMotionInPlace(std::vector<Eigen::MatrixXf>& motion) {
  // transpose the motion matrix due to different conventions in momentum vs
  // pymomentum
  for (auto& m : motion) {
    if (m.cols() > 0) {
      m.transposeInPlace();
    }
  }
}

std::tuple<momentum::Character, std::vector<Eigen::MatrixXf>, float>
loadFBXCharacterWithMotionFromFile(const std::string& fbxPath, bool permissive) {
  auto [character, motion, fps] =
      momentum::loadFbxCharacterWithMotion(filesystem::path(fbxPath), keepLocators, permissive);
  transposeMotionInPlace(motion);
  return {character, motion, fps};
}

std::tuple<momentum::Character, std::vector<Eigen::MatrixXf>, float>
loadFBXCharacterWithMotionFromBytes(const py::bytes& fbxBytes, bool permissive) {
  auto [character, motion, fps] =
      momentum::loadFbxCharacterWithMotion(toSpan<std::byte>(fbxBytes), keepLocators, permissive);
  transposeMotionInPlace(motion);
  return {character, motion, fps};
}

momentum::Character loadLocatorsFromFile(
    const momentum::Character& character,
    const std::string& locatorsPath) {
  MT_THROW_IF(locatorsPath.empty(), "Missing locators path.");
  momentum::Character result = character;
  auto locators = momentum::loadLocators(
      filesystem::path(locatorsPath), character.skeleton, character.parameterTransform);
  std::copy(locators.begin(), locators.end(), std::back_inserter(result.locators));
  return result;
}

momentum::Character loadConfigFromFile(
    const momentum::Character& character,
    const std::string& configPath) {
  MT_THROW_IF(configPath.empty(), "Missing model definition path.");

  const auto [parameterTransform, parameterLimits] =
      momentum::loadModelDefinition(filesystem::path(configPath), character.skeleton);

  return momentum::Character(
      character.skeleton,
      parameterTransform,
      parameterLimits,
      character.locators,
      character.mesh.get(),
      character.skinWeights.get(),
      character.collision.get(),
      character.poseShapes.get(),
      character.blendShape,
      character.faceExpressionBlendShape,
      character.name,
      character.inverseBindPose,
      character.skinnedLocators);
}

momentum::Character loadFBXCharacterFromBytes(const pybind11::bytes& bytes, bool permissive) {
  return momentum::loadFbxCharacter(toSpan<std::byte>(bytes), keepLocators, permissive);
}

momentum::Character loadConfigFromBytes(
    const momentum::Character& character,
    const pybind11::bytes& bytes) {
  const auto [parameterTransform, parameterLimits] =
      momentum::loadModelDefinition(toSpan<std::byte>(bytes), character.skeleton);

  return momentum::Character(
      character.skeleton,
      parameterTransform,
      parameterLimits,
      character.locators,
      character.mesh.get(),
      character.skinWeights.get(),
      character.collision.get(),
      character.poseShapes.get(),
      character.blendShape,
      character.faceExpressionBlendShape,
      character.name,
      character.inverseBindPose,
      character.skinnedLocators);
}

momentum::Character loadLocatorsFromBytes(
    const momentum::Character& character,
    const pybind11::bytes& bytes) {
  momentum::Character result = character;
  auto locators = momentum::loadLocatorsFromBuffer(
      toSpan<std::byte>(bytes), character.skeleton, character.parameterTransform);
  std::copy(locators.begin(), locators.end(), std::back_inserter(result.locators));
  return result;
}

momentum::Character loadURDFCharacterFromFile(const std::string& urdfPath) {
  return momentum::loadUrdfCharacter<float>(urdfPath);
}

momentum::Character loadURDFCharacterFromBytes(const pybind11::bytes& urdfBytes) {
  return momentum::loadUrdfCharacter<float>(toSpan<std::byte>(urdfBytes));
}

std::shared_ptr<const momentum::Mppca> loadPosePriorFromFile(const std::string& path) {
  return momentum::loadMppca(path);
}

void savePosePriorToFile(const momentum::Mppca& mppca, const std::string& path) {
  return momentum::saveMppca(mppca, path);
}

std::shared_ptr<const momentum::Mppca> loadPosePriorFromBytes(const py::bytes& bytes) {
  return momentum::loadMppca(toSpan<unsigned char>(bytes));
}

std::shared_ptr<momentum::BlendShape>
loadBlendShapeFromFile(const std::string& path, int nExpectedShapes, int nExpectedVertices) {
  auto result = momentum::loadBlendShape(path, nExpectedShapes, nExpectedVertices);
  MT_THROW_IF(result.getBaseShape().empty(), "Error loading blend shape from '{}'.", path);
  return std::make_shared<momentum::BlendShape>(std::move(result));
}

std::shared_ptr<momentum::BlendShape>
loadBlendShapeFromBytes(const pybind11::bytes& bytes, int nExpectedShapes, int nExpectedVertices) {
  PyBytesStreamBuffer streambuf(bytes);
  std::istream is(&streambuf);
  momentum::BlendShape result = momentum::loadBlendShape(is, nExpectedShapes, nExpectedVertices);
  MT_THROW_IF(result.getBaseShape().empty(), "Error loading blend shape from bytes.");
  return std::make_shared<momentum::BlendShape>(std::move(result));
}

py::bytes saveBlendShapeToBytes(const momentum::BlendShape& blendShape) {
  std::ostringstream oss;
  momentum::saveBlendShape(oss, blendShape);
  return py::bytes(oss.str());
}

void saveBlendShapeToFile(const momentum::BlendShape& blendShape, const std::string& path) {
  momentum::saveBlendShape(filesystem::path(path), blendShape);
}

std::string formatDimensions(const py::array& array) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < array.ndim(); ++i) {
    if (i > 0) {
      oss << " x ";
    }
    oss << array.shape(i);
  }
  oss << "]";
  return oss.str();
}

std::shared_ptr<momentum::BlendShape> loadBlendShapeFromTensors(
    pybind11::array_t<float> baseShape,
    pybind11::array_t<float> shapeVectors) {
  MT_THROW_IF(
      baseShape.ndim() != 2 || baseShape.shape(1) != 3,
      "In BlendShape.from_tensors(), expected base_shape to be [n_pts x 3] but got {}",
      formatDimensions(baseShape));
  MT_THROW_IF(
      shapeVectors.ndim() != 3 || shapeVectors.shape(2) != 3,
      "In BlendShape.from_tensors(), expected shape_vectors shape to be [n_shapes x n_pts x 3] but got {}",
      formatDimensions(shapeVectors));

  const auto nPts = baseShape.shape(0);
  const auto nShapes = shapeVectors.shape(0);
  MT_THROW_IF(
      shapeVectors.shape(1) != nPts,
      "In BlendShape.from_tensors(), expected match in n_pts dimensions. "
      "Expected base_shape to be [n_pts x 3] but got {}. "
      "Expected shape_vectors to be [n_shapes x n_pts x 3] but got {}.",
      formatDimensions(baseShape),
      formatDimensions(shapeVectors));

  std::vector<Eigen::Vector3f> baseShapeRes(nPts, Eigen::Vector3f::Zero());
  auto baseShapeAccess = baseShape.unchecked<2>();
  for (py::ssize_t i = 0; i < baseShapeAccess.shape(0); i++) {
    for (py::ssize_t j = 0; j < 3; j++) {
      baseShapeRes[i](j) = baseShapeAccess(i, j);
    }
  }

  Eigen::MatrixXf shapeVectorsRes(3 * nPts, nShapes);
  auto shapeVectorsAccess = shapeVectors.unchecked<3>();
  for (py::ssize_t i = 0; i < nShapes; i++) {
    for (py::ssize_t j = 0; j < nPts; j++) {
      for (py::ssize_t k = 0; k < 3; k++) {
        shapeVectorsRes(3 * j + k, i) = shapeVectorsAccess(i, j, k);
      }
    }
  }

  auto result = std::make_shared<momentum::BlendShape>(baseShapeRes, nShapes);
  result->setShapeVectors(shapeVectorsRes);
  return result;
}

momentum::Character replaceRestMesh(const momentum::Character& character, RowMatrixf positions) {
  MT_THROW_IF(
      !character.mesh, "Can't replace vertex positions because the Character lacks a mesh.");

  MT_THROW_IF(
      positions.cols() != 3 || positions.rows() != character.mesh->vertices.size(),
      "Expected a mesh position vector of size {} x 3; got {} x {}",
      character.mesh->vertices.size(),
      positions.rows(),
      positions.cols());

  momentum::Mesh newMesh(*character.mesh);
  for (size_t i = 0; i < newMesh.vertices.size(); ++i) {
    newMesh.vertices[i] = positions.row(i);
  }
  newMesh.updateNormals();

  return momentum::Character(
      character.skeleton,
      character.parameterTransform,
      character.parameterLimits,
      character.locators,
      &newMesh,
      character.skinWeights.get(),
      character.collision.get(),
      character.poseShapes.get(),
      character.blendShape,
      character.faceExpressionBlendShape,
      character.name,
      character.inverseBindPose,
      character.skinnedLocators);
}

at::Tensor uniformRandomToModelParameters(
    const momentum::Character& character,
    at::Tensor unifNoise) {
  unifNoise = unifNoise.contiguous().to(at::DeviceType::CPU, at::ScalarType::Float);

  const auto& paramTransform = character.parameterTransform;

  const int64_t nModelParam = (int64_t)paramTransform.numAllModelParameters();
  assert(paramTransform.name.size() == nModelParam);

  bool squeeze = false;
  if (unifNoise.ndimension() == 1) {
    unifNoise = unifNoise.unsqueeze(0);
    squeeze = true;
  }
  const auto nBatch = unifNoise.size(0);

  MT_THROW_IF(
      unifNoise.size(1) != nModelParam,
      "In uniformRandomToModelParameters(), expected array with size [nBatch, nModelParameters] with nModelParameters={}; got array with size {}",
      nModelParam,
      formatTensorSizes(unifNoise));

  at::Tensor result = at::zeros({nBatch, nModelParam}, at::kFloat);

  for (py::ssize_t iBatch = 0; iBatch < nBatch; ++iBatch) {
    auto res_i = toEigenMap<float>(result.select(0, iBatch));
    auto unif_i = toEigenMap<float>(unifNoise.select(0, iBatch));

    for (size_t iParam = 0; iParam < nModelParam; ++iParam) {
      // TODO apply limits
      const auto& name = paramTransform.name[iParam];
      if (name.find("scale_") != std::string::npos) {
        res_i[iParam] = 1.0f * (unif_i[iParam] - 0.5f);
      } else if (
          name.find("_tx") != std::string::npos || name.find("_ty") != std::string::npos ||
          name.find("_tz") != std::string::npos) {
        res_i[iParam] = 5.0f * (unif_i[iParam] - 0.5f);
      } else {
        // Assume it's a rotation parameter
        res_i[iParam] = (momentum::pi<float>() / 4.0f) * (unif_i[iParam] - 0.5f);
      }
    }
  }

  if (squeeze) {
    result = result.squeeze(0);
  }

  return result;
}

// Get a boolean vector for selected vertices from selected bones.
// The criteria are the bone with max weight on the vertex is selected and the
// weights from selected bones >= 0.5.
std::vector<bool> bonesToVertices(
    const momentum::Mesh& mesh,
    const momentum::SkinWeights& skinWeights,
    const std::vector<bool>& bones) {
  const auto nVerts = mesh.vertices.size();
  MT_THROW_IF(skinWeights.index.rows() != nVerts, "Skinning weights don't match mesh vertices.");

  std::vector<bool> result(nVerts);
  for (size_t iVert = 0; iVert < nVerts; ++iVert) {
    float maxWeight = 0.f; // Record max weight from a bone.
    float sumWeight = 0.f; // Record sum of weights from selected bones.
    size_t maxCoefIdx = momentum::kInvalidIndex;
    for (size_t jCoefIdx = 0; jCoefIdx < skinWeights.weight.row(iVert).size(); ++jCoefIdx) {
      float w = skinWeights.weight.row(iVert)[jCoefIdx];
      if (w > maxWeight) {
        maxWeight = w;
        maxCoefIdx = jCoefIdx;
      }
      if (bones[skinWeights.index(iVert, jCoefIdx)]) {
        sumWeight += w;
      }
    }
    size_t maxBoneIdx = skinWeights.index(iVert, maxCoefIdx);

    result[iVert] = (bones[maxBoneIdx] && sumWeight >= 0.5f);
  }

  return result;
}

std::vector<size_t> bitsetToList(const std::vector<bool>& bits) {
  std::vector<size_t> result;

  for (size_t i = 0; i < bits.size(); ++i) {
    if (bits[i]) {
      result.push_back(i);
    }
  }

  return result;
}

std::vector<bool> listToBitset(const std::vector<size_t>& list, const size_t sz) {
  std::vector<bool> result(sz);
  for (const auto& x : list) {
    result[x] = true;
  }
  return result;
}

std::vector<size_t> getUpperBodyJoints(const momentum::Skeleton& skeleton) {
  auto upperBodyRoot_idx = skeleton.getJointIdByName("b_spine0");
  MT_THROW_IF(upperBodyRoot_idx == momentum::kInvalidIndex, "Missing 'b_spine0' joint.");

  std::vector<size_t> result;

  // mark all joints above this joint:
  size_t cur = upperBodyRoot_idx;
  while (cur != momentum::kInvalidIndex) {
    result.push_back(cur);
    assert(cur < skeleton.joints.size());
    cur = skeleton.joints[cur].parent;
  }

  // and all joints below it:
  for (const auto child : skeleton.getChildrenJoints(upperBodyRoot_idx, true)) {
    result.push_back(child);
  }

  std::sort(result.begin(), result.end());

  return result;
}

momentum::Character stripLowerBodyVertices(const momentum::Character& character) {
  if (!character.mesh || !character.skinWeights) {
    return character;
  }

  const std::vector<bool> jointsToKeep =
      listToBitset(getUpperBodyJoints(character.skeleton), character.skeleton.joints.size());

  const std::vector<bool> verticesToKeep =
      bonesToVertices(*character.mesh, *character.skinWeights, jointsToKeep);

  return momentum::reduceMeshByVertices(character, verticesToKeep);
}

std::tuple<float, Eigen::VectorXf, Eigen::MatrixXf, float> getMppcaModel(
    const momentum::Mppca& mppca,
    int iModel) {
  MT_THROW_IF(iModel >= mppca.p, "Out of range iModel in Mppca.getModel()");

  const Eigen::Index dim = mppca.mu.cols();

  // The MPPCA model in momentum only stores the final covariance, not
  // the intermediate quantities, so to get these we need to do an eigenvalue
  // decomposition:
  assert((size_t)iModel < mppca.Cinv.size());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> Cinv_eigs(mppca.Cinv[iModel]);

  // Eigenvalues of the inverse are the inverse of the eigenvalues:
  Eigen::VectorXf C_eigenvalues = Cinv_eigs.eigenvalues().cwiseInverse();

  // Assume that it's not full rank and hence the last eigenvalue is sigma^2.
  const float sigma2 = C_eigenvalues(C_eigenvalues.size() - 1);

  // (sigma^2*I + W^T*W) has eigenvalues (sigma^2 + lambda)
  // where the lambda are the eigenvalues for W^T*W (which we want):
  C_eigenvalues.array() -= sigma2;

  const Eigen::VectorXf mu = mppca.mu.row(iModel);

  // Find the rank of W:
  Eigen::Index W_rank = C_eigenvalues.size();
  for (Eigen::Index i = 0; i < C_eigenvalues.size(); ++i) {
    if (C_eigenvalues(i) < 0.0001) {
      W_rank = i;
      break;
    }
  }

  Eigen::MatrixXf W(dim, W_rank);
  for (Eigen::Index i = 0; i < W_rank; ++i) {
    W.col(i) = std::sqrt(C_eigenvalues(i)) * Cinv_eigs.eigenvectors().col(i);
  }

  const float C_logDeterminant = -Cinv_eigs.eigenvalues().array().log().sum();

  // We have:
  //   Rpre(c) = std::log(pi(c))
  //       - 0.5 * C_logDeterminant
  //       - 0.5 * static_cast<double>(d) * std::log(2.0 * PI));
  // so std::log(pi(c)) = Rpre(c) + 0.5 * C_logDeterminant + 0.5 *
  //      d * std::log(2.0 * PI));
  const float log_pi = mppca.Rpre(iModel) + 0.5f * C_logDeterminant +
      0.5f * static_cast<float>(dim) * std::log(2.0 * momentum::pi<float>());
  const float pi = exp(log_pi);

  return {pi, mu, W, sigma2};
}

template <typename T>
std::string formatNumpyDims(const pybind11::array_t<T>& arr) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < arr.ndim(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << arr.shape(i);
  }
  oss << "]";
  return oss.str();
}

std::shared_ptr<momentum::Mppca> createMppcaModel(
    const Eigen::VectorXd& pi,
    const Eigen::MatrixXd& mu,
    const pybind11::array_t<double>& W,
    const Eigen::VectorXd& sigma,
    const std::vector<std::string>& parameterNames) {
  const Eigen::Index dimension = parameterNames.size();
  const Eigen::Index nModels = pi.size();

  MT_THROW_IF(
      mu.cols() != dimension || mu.rows() != nModels,
      "Invalid dimensions for mu; expected {} x {} but got {} x {}",
      nModels,
      dimension,
      mu.rows(),
      mu.cols());

  MT_THROW_IF(sigma.size() != nModels, "Mismatch between mixture counts in pi and sigma2.");

  MT_THROW_IF(
      W.ndim() != 3 || W.shape(0) != nModels || W.shape(2) != dimension,
      "Expected W of size [nMixtures={} x nPCA x d={}]; got {}",
      nModels,
      dimension,
      formatNumpyDims(W));

  std::vector<Eigen::MatrixXf> W_in;

  {
    auto r = W.unchecked<3>();

    const auto nPCA = W.shape(1);

    W_in.resize(nModels);
    for (py::ssize_t iMix = 0; iMix < nModels; iMix++) {
      assert(iMix < W_in.size());
      W_in[iMix].resize(dimension, nPCA);
      for (py::ssize_t jPCA = 0; jPCA < nPCA; jPCA++) {
        for (py::ssize_t kDim = 0; kDim < dimension; kDim++) {
          W_in[iMix](kDim, jPCA) = r(iMix, jPCA, kDim);
        }
      }
    }
  }

  auto result = std::make_shared<momentum::Mppca>();
  result->set(pi.cast<float>(), mu.cast<float>(), W_in, sigma.array().square().cast<float>());
  result->names = parameterNames;
  return result;
}

std::unique_ptr<momentum::Mesh> getPosedMesh(
    const momentum::Character& character,
    Eigen::Ref<const Eigen::VectorXf> jointParameters) {
  MT_THROW_IF(!character.mesh, "Character has no mesh.");

  auto result = std::make_unique<momentum::Mesh>(*character.mesh);

  using momentum::kParametersPerJoint;

  MT_THROW_IF(
      jointParameters.size() != kParametersPerJoint * character.skeleton.joints.size(),
      "Mismatched jointParameters size in getPosedMesh(); expected {}x{} parameters but got {}",
      kParametersPerJoint,
      character.skeleton.joints.size(),
      jointParameters.size());
  momentum::SkeletonState skelState;

  skelState.set(jointParameters, character.skeleton);

  result->vertices = momentum::applySSD(
      character.inverseBindPose, *character.skinWeights, character.mesh->vertices, skelState);
  result->updateNormals();
  return result;
}

std::tuple<Eigen::VectorXi, RowMatrixf> getLocators(
    const momentum::Character& character,
    const std::vector<std::string>& names) {
  Eigen::VectorXi parents = Eigen::VectorXi::Constant(names.size(), -1);
  RowMatrixf offsets = RowMatrixf::Zero(names.size(), 3);

  // locator name -> indices of the name in `names`. We use a vector here
  // to allow for duplicate names.
  std::unordered_map<std::string, std::vector<size_t>> nameMap;
  for (size_t i = 0; i < names.size(); ++i) {
    const auto itr = nameMap.find(names[i]);
    if (itr != nameMap.end()) {
      itr->second.push_back(i);
    } else {
      nameMap.emplace(names[i], std::vector<size_t>{i});
    }
  }

  for (const auto& l : character.locators) {
    const auto itr = nameMap.find(l.name);
    if (itr != nameMap.end()) {
      for (size_t idx : itr->second) {
        parents(idx) = l.parent;
        offsets.row(idx) = l.offset;
      }
    }
  }

  for (size_t iBone = 0; iBone < character.skeleton.joints.size(); ++iBone) {
    const auto& name = character.skeleton.joints[iBone].name;
    const auto itr = nameMap.find(name);
    if (itr == nameMap.end()) {
      continue;
    }
    for (size_t idx : itr->second) {
      MT_THROW_IF(
          parents(idx) != -1,
          "Duplicate joint '{}' found in both locators list and skeleton.",
          name);
      parents(idx) = iBone;
      // Offset in this case relative to parent bone is just 0
    }
  }

  for (size_t i = 0; i < names.size(); ++i) {
    MT_THROW_IF(parents(i) == -1, "Missing joint/locator '{}'.", names[i]);
  }

  return {parents, offsets};
}

// Template for applyModelParameterLimits(character, modelParams)
// so that it works with both float and double dtypes.
template <typename T>
at::Tensor applyModelParameterLimitsTemplate(
    const momentum::Character& character,
    at::Tensor modelParams) {
  TensorChecker checker("applyBodyParameterLimits");
  bool squeeze = false;
  const int nModelParamsID = -1;
  modelParams = checker.validateAndFixTensor(
      modelParams,
      "model_params",
      {nModelParamsID},
      {"nModelParams"},
      toScalarType<T>(),
      true,
      false,
      &squeeze);

  const int64_t nModelParams = checker.getBoundValue(nModelParamsID);
  MT_THROW_IF(
      character.parameterTransform.numAllModelParameters() != nModelParams,
      "pymomentum::applyModelParameterLimits(): model param ssize in input tensor does not match character")

  // character.parameterLimits can be empty. In this case, there is no
  // limit for all the model params.
  if (!character.parameterLimits.empty()) {
    // Store model param indices that have limits.
    std::vector<int64_t> limitedIndices;
    // Store the limit values.
    std::vector<T> minLimits, maxLimits;
    for (const auto& l : character.parameterLimits) {
      if (l.type == momentum::LimitType::MinMax) {
        limitedIndices.push_back(l.data.minMax.parameterIndex);
        minLimits.push_back(l.data.minMax.limits.x());
        maxLimits.push_back(l.data.minMax.limits.y());
      }
    }

    // Build tensors for doing the differentiable computation:
    // - We need an index tensor to call tensor.index_select() to get the slices
    // from modelParams that have limits.
    // - We then need to call torch.minimum() using a 1D tensor of upper
    //   bounds to clamp those model parameters from above.
    // - We also need torch.maximum() with a 1D tensor of lower bounds to
    //   clamp those model parameters from below.
    // - Finally, call tensor.index_copy() to combine those clamped model
    //   parameters with the bounds-free model parameters to build the returned
    //   tensor. The indices tensor used in this step is the same as the one in
    //   tensor.index_select().

    at::Tensor limitedIndicesTensor = to1DTensor(limitedIndices).to(modelParams.device());
    at::Tensor minLimitsTensor =
        to1DTensor(minLimits).to(modelParams.device(), modelParams.dtype());
    at::Tensor maxLimitsTensor =
        to1DTensor(maxLimits).to(modelParams.device(), modelParams.dtype());
    modelParams = modelParams.index_copy(
        -1,
        limitedIndicesTensor,
        torch::clamp(
            modelParams.index_select(-1, limitedIndicesTensor), minLimitsTensor, maxLimitsTensor));
  }

  if (squeeze) {
    modelParams = modelParams.squeeze(0);
  }

  return modelParams;
}

at::Tensor applyModelParameterLimits(const momentum::Character& character, at::Tensor modelParams) {
  if (hasFloat64(modelParams)) {
    return applyModelParameterLimitsTemplate<double>(character, modelParams);
  } else {
    return applyModelParameterLimitsTemplate<float>(character, modelParams);
  }
}

std::tuple<Eigen::VectorXf, Eigen::VectorXf> modelParameterLimits(
    const momentum::Character& character) {
  Eigen::VectorXf minLimits = Eigen::VectorXf::Constant(
      character.parameterTransform.numAllModelParameters(), std::numeric_limits<float>::lowest());
  Eigen::VectorXf maxLimits = Eigen::VectorXf::Constant(
      character.parameterTransform.numAllModelParameters(), std::numeric_limits<float>::max());

  for (const auto& l : character.parameterLimits) {
    if (l.type == momentum::LimitType::MinMax) {
      const auto& limitVal = l.data.minMax;
      minLimits(limitVal.parameterIndex) = limitVal.limits.x();
      maxLimits(limitVal.parameterIndex) = limitVal.limits.y();
    }
  }

  return {minLimits, maxLimits};
}

std::tuple<MatrixX7f, MatrixX7f> jointParameterLimits(const momentum::Character& character) {
  MatrixX7f minLimits = MatrixX7f::Constant(
      character.skeleton.joints.size(),
      momentum::kParametersPerJoint,
      std::numeric_limits<float>::lowest());
  MatrixX7f maxLimits = MatrixX7f::Constant(
      character.skeleton.joints.size(),
      momentum::kParametersPerJoint,
      std::numeric_limits<float>::max());

  for (const auto& l : character.parameterLimits) {
    if (l.type == momentum::LimitType::MinMaxJoint ||
        l.type == momentum::LimitType::MinMaxJointPassive) {
      const auto& limitVal = l.data.minMaxJoint;
      minLimits(limitVal.jointIndex, limitVal.jointParameter) = limitVal.limits.x();
      maxLimits(limitVal.jointIndex, limitVal.jointParameter) = limitVal.limits.y();
    }
  }

  return {minLimits, maxLimits};
}

py::array_t<float> getBindPose(const momentum::Character& character) {
  const auto& inverseBindPose = character.inverseBindPose;
  py::array_t<float> result =
      py::array_t<float>(std::vector<py::ssize_t>{(py::ssize_t)inverseBindPose.size(), 4, 4});
  auto r = result.mutable_unchecked<3>(); // Will throw if ndim != 3 or
                                          // flags.writable is false
  for (py::ssize_t i = 0; i < inverseBindPose.size(); i++) {
    const Eigen::Affine3f bindPose = inverseBindPose[i].inverse();
    for (py::ssize_t j = 0; j < 4; j++) {
      for (py::ssize_t k = 0; k < 4; k++) {
        r(i, j, k) = bindPose(j, k);
      }
    }
  }
  return result;
}

py::array_t<float> getInverseBindPose(const momentum::Character& character) {
  const auto& inverseBindPose = character.inverseBindPose;
  py::array_t<float> result =
      py::array_t<float>(std::vector<py::ssize_t>{(py::ssize_t)inverseBindPose.size(), 4, 4});
  auto r = result.mutable_unchecked<3>(); // Will throw if ndim != 3 or
                                          // flags.writable is false
  for (py::ssize_t i = 0; i < inverseBindPose.size(); i++) {
    for (py::ssize_t j = 0; j < 4; j++) {
      for (py::ssize_t k = 0; k < 4; k++) {
        r(i, j, k) = inverseBindPose[i](j, k);
      }
    }
  }

  return result;
}

std::vector<bool> jointListToBitset(
    const momentum::Character& character,
    const std::vector<int>& jointIndices) {
  std::vector<bool> activeJoints(character.skeleton.joints.size(), false);

  for (const auto& jointIndex : jointIndices) {
    if (jointIndex < 0 || jointIndex >= activeJoints.size()) {
      throw pybind11::index_error(fmt::format("Invalid joint index {}", jointIndex));
    }
    activeJoints.at(jointIndex) = true;
  }

  return activeJoints;
}

std::vector<int> bitsetToJointList(const std::vector<bool>& jointMask) {
  std::vector<int> jointIndices;
  for (size_t i = 0; i < jointMask.size(); ++i) {
    if (jointMask[i]) {
      jointIndices.push_back(static_cast<int>(i));
    }
  }
  return jointIndices;
}

} // namespace pymomentum
