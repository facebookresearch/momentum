/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/types.h>
#include <momentum/math/types.h>

#include <Eigen/Sparse>

namespace momentum {

/// Maps from joint parameters back to the model parameters.
///
/// Because the joint parameters have a much higher dimensionality (7*nJoints) than the number of
/// model parameters, in general we can't always find a set of model parameters that reproduce the
/// passed-in joint parameters. Therefore, the mapping is done in a least squares sense: we find
/// the model parameters that bring us closest to the passed-in joint parameters under L2.
///
/// @pre The parameter transform must be well-formed, in the sense that it has rank equal to the
/// number of model parameters. Any sane parameter transform should have this property (if not, it
/// would be possible to generate the same set of joint parameters with two different model
/// parameter vectors).
template <typename T>
struct InverseParameterTransformT {
 public:
  const SparseRowMatrix<T> transform;
  const Eigen::SparseQR<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> inverseTransform;
  // TODO: `offsets` is unused by the inverse mapping below and is documented as deprecated; remove
  // it and update all constructors/consumers.
  /// @deprecated Constant offset factor for each joint; retained for ABI compatibility but no
  /// longer consulted by `apply()`.
  const Eigen::VectorX<T> offsets;

  explicit InverseParameterTransformT(const ParameterTransformT<T>& paramTransform);

  /// Map joint parameters to model parameters.
  ///
  /// @return The fit model parameters; the residual from the least-squares fit is left in the
  ///   result's `offsets` vector.
  [[nodiscard]] CharacterParametersT<T> apply(const JointParametersT<T>& parameters) const;

  /// Dimension of the output model parameters vector.
  [[nodiscard]] Eigen::Index numAllModelParameters() const {
    return transform.cols();
  }

  /// Dimension of the input joint parameters vector.
  [[nodiscard]] Eigen::Index numJointParameters() const {
    return transform.rows();
  }
};

using InverseParameterTransform = InverseParameterTransformT<float>;
using InverseParameterTransformd = InverseParameterTransformT<double>;

/// Apply model parameter scale conversion to motion data.
///
/// Extracts scaling parameters from the parameter transform, converts joint-level identity
/// parameters (bone scales) to model-level parameters using InverseParameterTransform, and bakes
/// the scale values into each frame of the motion matrix.
///
/// @param[in] parameterTransform The character's parameter transform.
/// @param[in,out] motion The motion matrix (numModelParams x numFrames, column-major). Scale
///   parameters are added to each frame in-place.
/// @param[in] jointIdentity Joint-level identity parameters (bone scales).
/// @return Model-level identity parameters with scale values.
[[nodiscard]] ModelParameters applyModelParameterScales(
    const ParameterTransform& parameterTransform,
    MatrixXf& motion,
    const JointParameters& jointIdentity);

} // namespace momentum
