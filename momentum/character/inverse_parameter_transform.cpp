/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/inverse_parameter_transform.h"

#include "momentum/character/parameter_transform.h"
#include "momentum/common/checks.h"
#include "momentum/common/exception.h"
#include "momentum/common/log.h"

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

ModelParameters applyModelParameterScales(
    const ParameterTransform& parameterTransform,
    MatrixXf& motion,
    const JointParameters& jointIdentity) {
  const auto scalingParams = parameterTransform.getScalingParameters();

  // Check if motion already contains non-zero scale parameters
  bool hasScaleParametersInMotion = false;
  for (Eigen::Index iFrame = 0; iFrame < motion.cols() && !hasScaleParametersInMotion; ++iFrame) {
    for (Eigen::Index iParam = 0; iParam < parameterTransform.numAllModelParameters(); ++iParam) {
      if (scalingParams.test(iParam) && motion(iParam, iFrame) != 0) {
        hasScaleParametersInMotion = true;
        break;
      }
    }
  }

  // Convert joint identity parameters to model identity parameters
  ModelParameters modelIdentity = ModelParameters::Zero(parameterTransform.numAllModelParameters());
  if (jointIdentity.size() > 0 && !jointIdentity.v.isZero()) {
    if (hasScaleParametersInMotion) {
      MT_LOGW("Motion already contains scale parameters, not adding joint identity parameters.");
    }
    const auto scalingTransform = parameterTransform.simplify(scalingParams);
    const ModelParameters scaleParametersSubset =
        InverseParameterTransform(scalingTransform).apply(jointIdentity).pose;

    MT_CHECK(
        static_cast<Eigen::Index>(parameterTransform.name.size()) ==
            parameterTransform.numAllModelParameters(),
        "Parameter name count ({}) does not match model parameter count ({})",
        parameterTransform.name.size(),
        parameterTransform.numAllModelParameters());
    for (Eigen::Index iParam = 0; iParam < parameterTransform.numAllModelParameters(); iParam++) {
      if (scalingParams.test(iParam)) {
        MT_THROW_IF(
            static_cast<size_t>(iParam) >= parameterTransform.name.size(),
            "Parameter index {} exceeds parameter name count {}",
            iParam,
            parameterTransform.name.size());
        auto subsetParamIdx =
            scalingTransform.getParameterIdByName(parameterTransform.name[iParam]);
        MT_THROW_IF(
            subsetParamIdx >= static_cast<size_t>(scaleParametersSubset.size()),
            "Subset parameter index {} exceeds scaleParametersSubset size {}",
            subsetParamIdx,
            scaleParametersSubset.size());
        modelIdentity[iParam] = scaleParametersSubset(subsetParamIdx);

        // Add the identity parameters into the motion.
        if (!hasScaleParametersInMotion) {
          for (Eigen::Index jFrame = 0; jFrame < motion.cols(); ++jFrame) {
            motion(iParam, jFrame) += scaleParametersSubset(subsetParamIdx);
          }
        }
      }
    }
  }

  return modelIdentity;
}

} // namespace momentum
