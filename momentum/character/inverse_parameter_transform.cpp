/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/inverse_parameter_transform.h>

#include <momentum/character/parameter_transform.h>
#include <momentum/common/checks.h>

namespace momentum {

template <typename T>
Eigen::SparseMatrix<T> toColumnMajor(const SparseRowMatrix<T>& mat) {
  Eigen::SparseMatrix<T> result = mat;
  result.makeCompressed();
  return result;
}

template <typename T>
InverseParameterTransformT<T>::InverseParameterTransformT(
    const ParameterTransformT<T>& paramTransform)
    : transform(paramTransform.transform),
      inverseTransform(toColumnMajor<T>(paramTransform.transform)),
      offsets(paramTransform.offsets) {}

template <typename T>
CharacterParametersT<T> InverseParameterTransformT<T>::apply(
    const JointParametersT<T>& jointParameters) const {
  MT_CHECK(
      jointParameters.size() == transform.rows(),
      "{} is not {}",
      jointParameters.size(),
      transform.rows());
  MT_CHECK(offsets.size() == transform.rows(), "{} is not {}", offsets.size(), transform.rows());
  CharacterParametersT<T> result;
  result.pose = inverseTransform.solve(jointParameters.v - offsets);
  result.offsets = jointParameters.v - transform * result.pose.v;
  return result;
}

template struct InverseParameterTransformT<float>;
template struct InverseParameterTransformT<double>;

} // namespace momentum
