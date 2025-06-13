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

  // compute coefficients
  VectorXf coefficients(shapeVectors.cols());

  // calculate base transform
  const Quaternionf base = (baseJoint < state.jointState.size())
      ? baseRot * state.jointState.at(baseJoint).rotation().inverse()
      : baseRot;

  // set up pose shape coefficients
  for (size_t i = 0; i < jointMap.size(); i++) {
    const auto& jid = jointMap[i];
    if (jid < state.jointState.size()) {
      coefficients.segment<4>(i * 4) = (base * state.jointState.at(jid).rotation()).coeffs();
    }
  }

  // use the coefficients to calculate the new base shape
  std::vector<Vector3f> output(baseShape.size() / 3);
  Map<VectorXf> outputVec(&output[0][0], output.size() * 3);

  outputVec = baseShape + shapeVectors * coefficients;

  return output;
}

} // namespace momentum
