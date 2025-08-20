/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_momentum/tensor_momentum_utility.h"

#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/math/mesh.h>

namespace pymomentum {

void checkValidBoneIndex(
    at::Tensor idx,
    const momentum::Character& character,
    const char* name) {
  if (isEmpty(idx)) {
    return;
  }

  const int64_t nJoints = character.skeleton.joints.size();
  MT_THROW_IF(
      idx.less((int64_t)0).any().cpu().item<bool>() ||
          idx.greater_equal(nJoints).any().cpu().item<bool>(),
      "Invalid joint index found in {}; expected all values in the range [0, {}).",
      name,
      nJoints);
}

void checkValidParameterIndex(
    at::Tensor idx,
    const momentum::Character& character,
    const char* name,
    bool allow_missing) {
  if (isEmpty(idx)) {
    return;
  }

  const int64_t nParams = character.parameterTransform.numAllModelParameters();
  const int64_t minVal = allow_missing ? -1 : 0;
  MT_THROW_IF(
      idx.less((int64_t)minVal).any().cpu().item<bool>() ||
          idx.greater_equal(nParams).any().cpu().item<bool>(),
      "Invalid parameter index found in {}; expected all values in the range [0, {}).",
      name,
      nParams);
}

// allow_missing means -1 is allowed:
void checkValidVertexIndex(
    at::Tensor idx,
    const momentum::Character& character,
    const char* name,
    bool allow_missing) {
  if (isEmpty(idx)) {
    return;
  }

  MT_THROW_IF(
      !character.mesh && !allow_missing,
      "Vertex indices invalid for empty mesh.");

  const int64_t nVertices =
      (character.mesh) ? character.mesh->vertices.size() : 0;
  const int64_t minVal = allow_missing ? -1 : 0;
  MT_THROW_IF(
      idx.less((int64_t)minVal).any().cpu().template item<bool>() ||
          idx.greater_equal(nVertices).any().cpu().template item<bool>(),
      "Invalid vertex index found in {}; expected all values in the range [0, {}).",
      name,
      nVertices);
}

std::vector<bool> tensorToJointSet(
    const momentum::Skeleton& skeleton,
    at::Tensor jointSet,
    DefaultJointSet defaultValue) {
  const auto nJoints = skeleton.joints.size();

  if (isEmpty(jointSet)) {
    switch (defaultValue) {
      case DefaultJointSet::ALL_ZEROS: {
        return std::vector<bool>(nJoints, false);
        case DefaultJointSet::ALL_ONES:
          return std::vector<bool>(nJoints, true);
        case DefaultJointSet::NO_DEFAULT:
        default:
            // fall through to the check below:
            ;
      }
    }
  }

  MT_THROW_IF(
      isEmpty(jointSet) || jointSet.ndimension() != 1 ||
          jointSet.size(0) != nJoints,
      "Mismatch between joint set size and number of joints in skeleton.");

  jointSet = jointSet.to(at::DeviceType::CPU, at::ScalarType::Bool);
  auto ptr = (uint8_t*)jointSet.data_ptr();
  std::vector<bool> result(nJoints);
  for (int k = 0; k < nJoints; ++k) {
    result[k] = (ptr[k] != 0);
  }
  return result;
}

} // namespace pymomentum
