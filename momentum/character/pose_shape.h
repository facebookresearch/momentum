/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/math/types.h>

namespace momentum {

/// Computes shape deformations based on joint rotations.
///
/// PoseShape implements a linear blend shape system where shape deformations
/// are driven by joint rotations. This is useful for modeling effects like
/// muscle bulging or skin sliding that occur during animation.
struct PoseShape {
  /// Index of the joint used as reference for computing relative rotations.
  size_t baseJoint;

  /// Base rotation used as reference for computing rotation differences.
  ///
  /// The effective rotation used for deformation is baseRot * joint.rotation().inverse()
  Quaternionf baseRot;

  /// Indices of joints whose rotations drive the shape deformation.
  std::vector<size_t> jointMap;

  /// Base shape vertices stored as a flat vector [x1,y1,z1,x2,y2,z2,...].
  ///
  /// The size must be a multiple of 3 (3 coordinates per vertex).
  VectorXf baseShape;

  /// Shape deformation vectors for each quaternion component of each joint.
  ///
  /// Matrix dimensions are [baseShape.size() × (jointMap.size() * 4)].
  /// Each column corresponds to a quaternion component (x,y,z,w) of a joint.
  MatrixXf shapeVectors;

  /// Computes the deformed shape based on the current skeleton state.
  ///
  /// @param state Current state of the skeleton containing joint rotations
  /// @return Vector of 3D vertex positions representing the deformed shape
  /// @throws If baseShape.size() != shapeVectors.rows()
  [[nodiscard]] std::vector<Vector3f> compute(const SkeletonState& state) const;

  /// Checks if this PoseShape is approximately equal to another.
  ///
  /// Two PoseShape objects are considered approximately equal if all their
  /// corresponding members are approximately equal.
  ///
  /// @param poseShape The PoseShape to compare with
  /// @return True if the PoseShape objects are approximately equal
  [[nodiscard]] inline bool isApprox(const PoseShape& poseShape) const {
    return (
        (baseJoint == poseShape.baseJoint) && baseRot.isApprox(poseShape.baseRot) &&
        (jointMap == poseShape.jointMap) && baseShape.isApprox(poseShape.baseShape) &&
        shapeVectors.isApprox(poseShape.shapeVectors));
  }
};

} // namespace momentum
