/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/pose_shape.h"

#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"

namespace momentum {

std::vector<Vector3f> PoseShape::compute(const SkeletonState& state) const {
  MT_CHECK(
      baseShape.size() == shapeVectors.rows(),
      "{} is not {}",
      baseShape.size(),
      shapeVectors.rows());

  VectorXf coefficients(shapeVectors.cols());

  // If baseJoint is out of range, fall back to baseRot directly (no relative rotation).
  const Quaternionf base = (baseJoint < state.jointState.size())
      ? baseRot * state.jointState.at(baseJoint).rotation().inverse()
      : baseRot;

  // Pack each driving joint's rotation (relative to base) as a 4-vector of quaternion
  // coefficients (x, y, z, w). Out-of-range joints leave their segment uninitialized.
  // TODO: zero-initialize `coefficients` so out-of-range joint indices don't read garbage.
  for (size_t i = 0; i < jointMap.size(); i++) {
    const auto& jid = jointMap[i];
    if (jid < state.jointState.size()) {
      coefficients.segment<4>(i * 4) = (base * state.jointState.at(jid).rotation()).coeffs();
    }
  }

  std::vector<Vector3f> output(baseShape.size() / 3);
  if (output.empty()) {
    return output;
  }
  Map<VectorXf> outputVec(&output[0][0], output.size() * 3);

  outputVec = baseShape + shapeVectors * coefficients;

  return output;
}

} // namespace momentum
