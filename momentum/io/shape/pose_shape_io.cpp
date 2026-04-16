/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/shape/pose_shape_io.h"

#include "momentum/character/character.h"
#include "momentum/common/exception.h"
#include "momentum/common/log.h"
#include "momentum/math/mesh.h"

#include <fstream>
#include <limits>

namespace momentum {

PoseShape loadPoseShape(const std::string& filename, const Character& character) {
  PoseShape result;

  MT_CHECK(character.mesh);

  std::ifstream data(filename, std::ios::in | std::ios::binary);
  if (!data.is_open()) {
    return result;
  }

  // read dimensions
  uint64_t numRows = 0;
  uint64_t numJoints = 0;
  data.read((char*)&numRows, sizeof(numRows));
  data.read((char*)&numJoints, sizeof(numJoints));

  if (!data.good()) {
    MT_LOGW("{}: Failed to read dimensions from file {}", __func__, filename);
    return result;
  }

  // Validate sizes to prevent excessive allocation from malformed files
  constexpr uint64_t kMaxDimension = 10'000'000;
  constexpr uint64_t kMaxStringLength = 10'000;
  if (numRows > kMaxDimension || numJoints > kMaxDimension) {
    MT_LOGW(
        "{}: Unreasonable dimensions in file {}: numRows={}, numJoints={}",
        __func__,
        filename,
        numRows,
        numJoints);
    return result;
  }

  // Guard against integer overflow in numJoints * 4
  if (numJoints > std::numeric_limits<uint64_t>::max() / 4) {
    MT_LOGW("{}: Integer overflow in numCols calculation for file {}", __func__, filename);
    return result;
  }
  const uint64_t numCols = numJoints * 4;

  // Guard against integer overflow in numRows * numCols
  if (numRows != 0 && numCols != 0 && numRows > std::numeric_limits<uint64_t>::max() / numCols) {
    MT_LOGW("{}: Integer overflow in shape matrix size for file {}", __func__, filename);
    return result;
  }

  MT_CHECK(
      character.mesh->vertices.size() * 3 == numRows,
      "{}, {}",
      character.mesh->vertices.size() * 3,
      numRows);

  uint64_t count = 0;
  data.read((char*)&count, sizeof(uint64_t));
  if (!data.good() || count > kMaxStringLength) {
    MT_LOGW("{}: Invalid base joint name length in file {}", __func__, filename);
    return result;
  }
  std::string base;
  base.resize(count);
  data.read((char*)base.data(), count);
  result.baseJoint = character.skeleton.getJointIdByName(base);
  MT_CHECK(result.baseJoint != kInvalidIndex);
  MT_CHECK(
      0 <= result.baseJoint && result.baseJoint < character.skeleton.joints.size(),
      "Invalid joint index");
  // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
  result.baseRot = character.skeleton.joints[result.baseJoint].preRotation;

  // load names
  std::vector<std::string> names(numJoints);
  for (size_t i = 0; i < numJoints; i++) {
    data.read((char*)&count, sizeof(uint64_t));
    if (!data.good() || count > kMaxStringLength) {
      MT_LOGW("{}: Invalid joint name length in file {}", __func__, filename);
      return result;
    }
    names[i].resize(count);
    data.read((char*)names[i].data(), count);
  }

  // read mean shape
  result.baseShape.resize(numRows);
  data.read((char*)result.baseShape.data(), sizeof(float) * numRows);
  if (!data.good()) {
    MT_LOGW("{}: Failed to read base shape data from file {}", __func__, filename);
    return result;
  }

  // add character vertices
  const Map<const VectorXf> mesh(
      &character.mesh->vertices[0][0], character.mesh->vertices.size() * 3);
  result.baseShape += mesh;

  // load shapeVectors
  result.shapeVectors.resize(numRows, numCols);
  data.read((char*)result.shapeVectors.data(), sizeof(float) * numRows * numCols);
  if (!data.good()) {
    MT_LOGW("{}: Failed to read shape vectors data from file {}", __func__, filename);
    return result;
  }

  // generate mapping from names
  result.jointMap.resize(numJoints);
  for (size_t i = 0; i < names.size(); i++) {
    result.jointMap[i] = character.skeleton.getJointIdByName(names[i]);
    MT_CHECK(result.jointMap[i] != kInvalidIndex);
  }

  return result;
}

} // namespace momentum
