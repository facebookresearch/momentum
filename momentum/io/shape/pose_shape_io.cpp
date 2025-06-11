/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/shape/pose_shape_io.h"

#include "momentum/character/character.h"
#include "momentum/common/log.h"
#include "momentum/math/mesh.h"

#include <fstream>

namespace momentum {

PoseShape loadPoseShape(const std::string& filename, const Character& character) {
  PoseShape result;

  MT_CHECK(character.mesh);

  std::ifstream data(filename, std::ios::in | std::ios::binary);
  if (!data.is_open()) {
    return result;
  }

  // read dimensions
  uint64_t numRows;
  uint64_t numJoints;
  data.read((char*)&numRows, sizeof(numRows));
  data.read((char*)&numJoints, sizeof(numJoints));
  const uint64_t numCols = numJoints * 4;

  MT_CHECK(
      character.mesh->vertices.size() * 3 == numRows,
      "{}, {}",
      character.mesh->vertices.size() * 3,
      numRows);

  uint64_t count = 0;
  data.read((char*)&count, sizeof(uint64_t));
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
    names[i].resize(count);
    data.read((char*)names[i].data(), count);
  }

  // read mean shape
  result.baseShape.resize(numRows);
  data.read((char*)result.baseShape.data(), sizeof(float) * numRows);

  // add character vertices
  const Map<const VectorXf> mesh(
      &character.mesh->vertices[0][0], character.mesh->vertices.size() * 3);
  result.baseShape += mesh;

  // load shapeVectors
  result.shapeVectors.resize(numRows, numCols);
  data.read((char*)result.shapeVectors.data(), sizeof(float) * numRows * numCols);

  // generate mapping from names
  result.jointMap.resize(numJoints);
  for (size_t i = 0; i < names.size(); i++) {
    result.jointMap[i] = character.skeleton.getJointIdByName(names[i]);
    MT_CHECK(result.jointMap[i] != kInvalidIndex);
  }

  return result;
}

} // namespace momentum
