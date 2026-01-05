/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/skeleton_state.h>
#include <momentum/character_sequence_solver/finite_difference_sequence_error_function.h>
#include <momentum/character_sequence_solver/fwd.h>

namespace momentum {

/// Error function that penalizes the jerk (third derivative) of joint positions across four
/// consecutive frames using a standard finite difference stencil [1, -3, 3, -1].
///
/// The jerk residual for each joint is computed as:
///   jerk = pos[t-1] - 3*pos[t] + 3*pos[t+1] - pos[t+2] - targetJerk
///
/// This is useful for enforcing smoothness in acceleration changes, which produces more
/// natural-looking motion. By default, the target jerk is zero, which penalizes any non-zero
/// jerk (smoothness constraint on acceleration).
///
/// Note: This error function only constrains position jerk, not rotation jerk, as rotation
/// derivatives involve significantly more complex mathematics.
template <typename T>
class JerkSequenceErrorFunctionT : public FiniteDifferenceSequenceErrorFunctionT<T> {
 public:
  JerkSequenceErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);
  explicit JerkSequenceErrorFunctionT(const Character& character);

  /// Set the per-joint weights for the jerk error.
  /// @param weights Per-joint weights vector. Size must match the number of joints.
  void setTargetWeights(const Eigen::VectorX<T>& weights) {
    FiniteDifferenceSequenceErrorFunctionT<T>::setTargetWeights(weights);
  }

  /// Set a single target jerk applied to all joints.
  /// @param jerk The target jerk vector.
  void setTargetJerk(const Eigen::Vector3<T>& jerk) {
    FiniteDifferenceSequenceErrorFunctionT<T>::setTargetValue(jerk);
  }

  /// Set per-joint target jerks.
  /// @param jerks Vector of target jerks, one per joint.
  void setTargetJerks(const std::vector<Eigen::Vector3<T>>& jerks) {
    FiniteDifferenceSequenceErrorFunctionT<T>::setTargetValues(jerks);
  }

  /// Reset weights to ones and target jerks to zero.
  void reset() {
    FiniteDifferenceSequenceErrorFunctionT<T>::reset();
  }

  [[nodiscard]] const Eigen::VectorX<T>& getTargetWeights() const {
    return FiniteDifferenceSequenceErrorFunctionT<T>::getTargetWeights();
  }

  [[nodiscard]] const std::vector<Eigen::Vector3<T>>& getTargetJerks() const {
    return FiniteDifferenceSequenceErrorFunctionT<T>::getTargetValues();
  }
};

} // namespace momentum
