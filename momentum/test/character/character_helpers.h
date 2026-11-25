/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/math/mppca.h>

namespace momentum {

/// Creates a character with a customizable number of joints.
///
/// @param numJoints The number of joints in the resulting character.
/// @note: The `numJoints` parameter should be equal to 3 or greater.
[[nodiscard]] Character createTestCharacter(size_t numJoints = 3);
// TODO: Currently, the number of parameters is fixed to 10, affecting only the first 3 joints.
// Additional work is needed to scale the number of parameters with the number of joints.

template <typename T>
[[nodiscard]] std::shared_ptr<const MppcaT<T>> createDefaultPosePrior();

[[nodiscard]] Character withTestBlendShapes(const Character& character);

[[nodiscard]] Character withTestFaceExpressionBlendShapes(const Character& character);

/// Returns a copy of `character` with deterministic physical mass properties attached to its first
/// two joints, for exercising physical-properties I/O round trips.
///
/// @pre `character.skeleton.joints.size() >= 2`.
[[nodiscard]] Character withTestPhysicalProperties(const Character& character);

/// The joint parameters of every frame of `motion`, with the identity `offsets` added as
/// ParameterTransform::apply(CharacterParameters) adds them: `transform * motion.col(i) + offsets`.
///
/// @param parameterTransform The parameter transform of the motion's character.
/// @param motion Model parameters (numAllModelParameters x frames).
/// @param offsets Identity joint parameters (numJointParameters values), or empty for none.
/// @return Joint parameters (numJointParameters x frames).
[[nodiscard]] MatrixXf jointMotionOf(
    const ParameterTransform& parameterTransform,
    const MatrixXf& motion,
    const VectorXf& offsets = VectorXf());

/// Zeroes the model parameters that drive a joint scale in every frame of `motion`, for formats
/// that cannot store joint scales.
///
/// @param parameterTransform The parameter transform of the motion's character.
/// @param motion Model parameters (numAllModelParameters x frames).
void removeScaleMotion(const ParameterTransform& parameterTransform, MatrixXf& motion);

/// The largest differences between the global joint transforms of two skeletons in any frame.
struct JointTransformDifference {
  /// The largest distance between the positions of two joints with the same name.
  float translation = 0.0f;
  /// The largest Frobenius norm of the difference between the rotations of two joints with the
  /// same name.
  float rotation = 0.0f;
};

/// Compares the global transforms of the joints of `expected` with those of the joints of
/// `actual` that have the same names, in every frame. Throws when a joint of `expected` has no
/// joint with its name in `actual`, or when the frame counts differ.
///
/// @param expected The expected skeleton.
/// @param expectedJointMotion Joint parameters of `expected` (numJointParameters x frames).
/// @param actual The skeleton to compare, which can list its joints in another order.
/// @param actualJointMotion Joint parameters of `actual` (numJointParameters x frames).
/// @return The largest differences.
[[nodiscard]] JointTransformDifference maxGlobalJointTransformDifference(
    const Skeleton& expected,
    const MatrixXf& expectedJointMotion,
    const Skeleton& actual,
    const MatrixXf& actualJointMotion);

} // namespace momentum
