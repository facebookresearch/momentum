/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/types.h>
#include <momentum/math/types.h>

#include <Eigen/Sparse>

namespace momentum {

// Maps from joint parameters back to the model parameters.  Because the joint
// parameters have a much higher dimensionality (7*nJoints) than the number of
// model parameters, in general we can't always find a set of model parameters
// that reproduce the passed-in joint parameters.  Therefore, the mapping is
// done in a least squares sense: we find the model parameters that bring use
// closest to the passed-in joint parameters under L2.
//
// Note that this assumes the parameter transform is well-formed, in the sense
// that it has rank equal to the number of model parameters.  Any sane
// parameter transform should have this property (if not, it would be possible
// to generate the same set of joint parameters with two different model
// parameter vectors).

template <typename T>
struct InverseParameterTransformT {
 public:
  const SparseRowMatrix<T> transform;
  const Eigen::SparseQR<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> inverseTransform;
  const Eigen::VectorX<T> offsets; // DEPRECATED: constant offset factor for each joint

  explicit InverseParameterTransformT(const ParameterTransformT<T>& paramTransform);

  // map joint parameters to model parameters.  The residual from the fit is
  // left in the result's offsets vector.
  [[nodiscard]] CharacterParametersT<T> apply(const JointParametersT<T>& parameters) const;

  // Dimension of the output model parameters vector:
  [[nodiscard]] Eigen::Index numAllModelParameters() const {
    return transform.cols();
  }

  // Dimension of the input joint parameters vector:
  [[nodiscard]] Eigen::Index numJointParameters() const {
    return transform.rows();
  }
};

using InverseParameterTransform = InverseParameterTransformT<float>;
using InverseParameterTransformd = InverseParameterTransformT<double>;

} // namespace momentum
