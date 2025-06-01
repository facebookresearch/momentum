/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/parameter_transform.h"

#include "momentum/common/checks.h"
#include "momentum/common/exception.h"
#include "momentum/common/profile.h"

#include <fmt/format.h>

namespace momentum {

template <typename T>
size_t ParameterTransformT<T>::getParameterIdByName(const std::string& nm) const {
  for (size_t d = 0; d < name.size(); d++) {
    if (name[d] == nm)
      return d;
  }

  return kInvalidIndex;
}

template <typename T>
ParameterSet ParameterTransformT<T>::getParameterSet(
    const std::string& parameterSetName,
    bool allowMissing) const {
  const auto itr = this->parameterSets.find(parameterSetName);
  if (itr == this->parameterSets.end()) {
    MT_THROW_IF(!allowMissing, "Missing parameter set: {}", parameterSetName);
    return ParameterSet{};
  }
  return itr->second;
}

template <typename T>
ParameterTransformT<T> ParameterTransformT<T>::empty(size_t nJointParameters) {
  ParameterTransformT<T> result;
  result.offsets.setZero(nJointParameters);
  result.activeJointParams.setConstant(nJointParameters, false);
  result.transform.resize(nJointParameters, 0);
  result.name.clear();
  return result;
}

template <typename T>
ParameterTransformT<T> ParameterTransformT<T>::identity(gsl::span<const std::string> jointNames) {
  ParameterTransformT<T> result;
  const size_t nJoints = jointNames.size();
  const size_t nJointParameters = nJoints * kParametersPerJoint;
  result.offsets.setZero(nJointParameters);
  result.activeJointParams.setConstant(nJointParameters, true);
  result.transform.resize(nJointParameters, nJointParameters);
  result.transform.setIdentity();
  result.name.resize(nJointParameters);
  for (size_t i = 0; i < nJoints; i++) {
    for (size_t j = 0; j < kParametersPerJoint; ++j) {
      result.name[i * kParametersPerJoint + j] =
          fmt::format("{}_{}", jointNames[i], kJointParameterNames[j]);
    }
  }
  return result;
}

template <typename T>
VectorX<bool> ParameterTransformT<T>::computeActiveJointParams(const ParameterSet& ps) const {
  VectorX<bool> result = VectorX<bool>::Constant(this->transform.rows(), false);
  for (int i = 0; i < this->transform.outerSize(); ++i) {
    for (typename SparseRowMatrix<T>::InnerIterator it(this->transform, i); it; ++it) {
      if (ps.test(it.col())) {
        result[it.row()] = 1;
      }
    }
  }
  return result;
}

// map model parameters to joint parameters using a linear transformation
template <typename T>
JointParametersT<T> ParameterTransformT<T>::apply(const ModelParametersT<T>& parameters) const {
  MT_PROFILE_FUNCTION();
  MT_CHECK(
      parameters.size() == transform.cols(),
      "Size mismatch: The size of the parameters vector '{}' does not match the number of columns in the transform matrix '{}'.",
      parameters.size(),
      transform.cols());
  MT_CHECK(
      offsets.size() == transform.rows(),
      "Size mismatch: The size of the offsets vector '{}' does not match the number of rows in the transform matrix '{}'.",
      offsets.size(),
      transform.rows());
  return transform * parameters.v + offsets;
}

// map model parameters to joint parameters using a linear transformation
template <typename T>
JointParametersT<T> ParameterTransformT<T>::apply(const CharacterParametersT<T>& parameters) const {
  MT_PROFILE_FUNCTION();
  MT_CHECK(
      parameters.pose.size() == transform.cols(),
      "{} is not {}",
      parameters.pose.size(),
      transform.cols());
  JointParametersT<T> result = transform * parameters.pose.v;
  if (parameters.offsets.size() > 0) {
    MT_CHECK(
        parameters.offsets.size() == transform.rows(),
        "{} is not {}",
        parameters.offsets.size(),
        transform.rows());
    result.v += parameters.offsets.v;
  }
  return result;
}

// return rest pose parameters
template <typename T>
JointParametersT<T> ParameterTransformT<T>::zero() const {
  MT_CHECK(offsets.size() == transform.rows(), "{} is not {}", offsets.size(), transform.rows());
  return offsets;
}

// return rest pose parameters
template <typename T>
JointParametersT<T> ParameterTransformT<T>::bindPose() const {
  return JointParametersT<T>::Zero(transform.rows());
}

// get a list of scaling parameters
template <typename T>
ParameterSet ParameterTransformT<T>::getScalingParameters() const {
  ParameterSet result;

  for (size_t i = 0; i < name.size(); i++) {
    if (name[i].find("scale_") != std::string::npos)
      result.set(i);
  }

  return result;
}

// get a list of root parameters
template <typename T>
ParameterSet ParameterTransformT<T>::getRigidParameters() const {
  ParameterSet result;

  for (size_t i = 0; i < name.size(); i++) {
    if (name[i].find("root_") != std::string::npos || name[i].find("hips") != std::string::npos)
      result.set(i);
  }

  return result;
}

template <typename T>
ParameterSet ParameterTransformT<T>::getPoseParameters() const {
  return ~getScalingParameters() & ~getBlendShapeParameters() & ~getFaceExpressionParameters();
}

template <typename T>
ParameterTransformT<T> ParameterTransformT<T>::simplify(
    const ParameterSet& enabledParameters) const {
  return std::get<0>(subsetParameterTransform(*this, ParameterLimits{}, enabledParameters));
}

template <typename T>
ParameterTransformT<T> mapParameterTransformJoints(
    const ParameterTransformT<T>& parameterTransform,
    size_t numTargetJoints,
    const std::vector<size_t>& jointMapping) {
  ParameterTransformT<T> mappedTransform;

  // No change in the parameters:
  mappedTransform.name = parameterTransform.name;

  // resize the offset matrix to the right size
  mappedTransform.offsets.setZero(numTargetJoints * kParametersPerJoint);
  mappedTransform.activeJointParams.setConstant(numTargetJoints * kParametersPerJoint, false);

  // map the offset from the anim skel to the new one
  for (size_t i = 0; i < jointMapping.size(); i++) {
    if (jointMapping[i] != kInvalidIndex) {
      for (size_t d = 0; d < kParametersPerJoint; d++) {
        mappedTransform.offsets[jointMapping[i] * kParametersPerJoint + d] =
            parameterTransform.offsets(i * kParametersPerJoint + d);
      }
    }
  }

  // map the transform from the anim skel to the new one
  std::vector<Eigen::Triplet<float>> triplets;
  for (int k = 0; k < parameterTransform.transform.outerSize(); ++k) {
    for (typename SparseRowMatrix<T>::InnerIterator it(parameterTransform.transform, k); it; ++it) {
      // row is joint + type index, col is parameter
      const int jIndex = static_cast<int>(it.row()) / kParametersPerJoint;
      const int jOffset = static_cast<int>(it.row()) % kParametersPerJoint;

      const auto mappedJoint = jointMapping[jIndex];
      if (mappedJoint == kInvalidIndex) {
        continue;
      }

      triplets.push_back(Eigen::Triplet<float>(
          mappedJoint * kParametersPerJoint + jOffset, static_cast<int>(it.col()), it.value()));

      // enable joint channels
      mappedTransform.activeJointParams[mappedJoint * kParametersPerJoint + jOffset] = true;
    }
  }

  // resize the Transform matrix to the correct size
  mappedTransform.transform.resize(
      static_cast<int>(numTargetJoints) * kParametersPerJoint,
      static_cast<int>(mappedTransform.name.size()));

  // create sparse matrix from triplet
  mappedTransform.transform.setFromTriplets(triplets.begin(), triplets.end());

  // copy over the parameterSets
  mappedTransform.parameterSets = parameterTransform.parameterSets;
  mappedTransform.poseConstraints = parameterTransform.poseConstraints;

  return mappedTransform;
}

template <typename T>
std::tuple<ParameterTransformT<T>, ParameterLimits> subsetParameterTransform(
    const ParameterTransformT<T>& paramTransformOld,
    const ParameterLimits& paramLimitsOld,
    const ParameterSet& paramSet) {
  const auto nOldParam = paramTransformOld.numAllModelParameters();

  std::vector<size_t> oldParamToNewParam(nOldParam, kInvalidIndex);
  std::vector<size_t> newParamToOldParam;

  for (Eigen::Index iOldParam = 0; iOldParam < nOldParam; ++iOldParam) {
    if (!paramSet.test(iOldParam))
      continue;

    oldParamToNewParam[iOldParam] = newParamToOldParam.size();
    newParamToOldParam.push_back(iOldParam);
  }

  const auto nNewParam = newParamToOldParam.size();

  ParameterTransformT<T> paramTransformNew;

  paramTransformNew.name.reserve(nNewParam);
  for (const auto iOldParam : newParamToOldParam)
    paramTransformNew.name.push_back(paramTransformOld.name[iOldParam]);

  // Offsets are unchanged because they are the size #joints, not #param.
  paramTransformNew.offsets = paramTransformOld.offsets;

  paramTransformNew.activeJointParams.setConstant(paramTransformOld.numJointParameters(), false);

  // Remap the transformation.
  using IndexType = typename SparseRowMatrix<T>::Index;
  std::vector<Eigen::Triplet<T>> tripletsNew;
  for (int k = 0; k < paramTransformOld.transform.outerSize(); ++k) {
    for (typename SparseRowMatrix<T>::InnerIterator it(paramTransformOld.transform, k); it; ++it) {
      const auto iOldParam = it.col();
      const auto iNewParam = oldParamToNewParam[iOldParam];

      if (iNewParam == kInvalidIndex)
        continue;

      tripletsNew.emplace_back((int)it.row(), (int)iNewParam, it.value());
      paramTransformNew.activeJointParams[it.row()] = true;
    }
  }

  paramTransformNew.transform.resize(paramTransformOld.transform.rows(), (IndexType)nNewParam);
  paramTransformNew.transform.setFromTriplets(tripletsNew.begin(), tripletsNew.end());

  for (const auto& [name, constraintsOld] : paramTransformOld.poseConstraints) {
    PoseConstraint pcNew;
    for (const auto& [idxOld, weight] : constraintsOld.parameterIdValue) {
      auto idxNew = oldParamToNewParam[idxOld];
      if (idxNew != kInvalidIndex) {
        pcNew.parameterIdValue.emplace_back(idxNew, weight);
      }
    }

    paramTransformNew.poseConstraints.insert(std::make_pair(name, pcNew));
  }

  paramTransformNew.blendShapeParameters =
      Eigen::VectorXi::Constant(paramTransformOld.blendShapeParameters.size(), -1);
  for (Eigen::Index iBlendShape = 0; iBlendShape < paramTransformOld.blendShapeParameters.size();
       ++iBlendShape) {
    auto paramOld = paramTransformOld.blendShapeParameters(iBlendShape);
    if (paramOld >= 0) {
      auto paramNew = oldParamToNewParam[paramOld];
      if (paramNew != kInvalidIndex) {
        paramTransformNew.blendShapeParameters(iBlendShape) = paramNew;
      }
    }
  }

  paramTransformNew.faceExpressionParameters =
      Eigen::VectorXi::Constant(paramTransformOld.faceExpressionParameters.size(), -1);
  for (Eigen::Index iBlendShape = 0;
       iBlendShape < paramTransformOld.faceExpressionParameters.size();
       ++iBlendShape) {
    auto paramOld = paramTransformOld.faceExpressionParameters(iBlendShape);
    if (paramOld >= 0) {
      auto paramNew = oldParamToNewParam[paramOld];
      if (paramNew != kInvalidIndex) {
        paramTransformNew.faceExpressionParameters(iBlendShape) = paramNew;
      }
    }
  }

  for (const auto& paramSetOld : paramTransformOld.parameterSets) {
    ParameterSet paramSetNew;
    paramSetNew.reset();

    for (size_t iNewParam = 0; iNewParam < nNewParam; ++iNewParam) {
      const auto iOldParam = newParamToOldParam[iNewParam];
      paramSetNew.set(iNewParam, paramSetOld.second.test(iOldParam));
    }

    paramTransformNew.parameterSets.emplace(paramSetOld.first, paramSetNew);
  }

  ParameterLimits paramLimitsNew;
  paramLimitsNew.reserve(paramLimitsOld.size());
  for (const auto& limitOld : paramLimitsOld) {
    auto limitNew = limitOld;
    switch (limitOld.type) {
      case MinMax: {
        auto& data = limitNew.data.minMax;
        data.parameterIndex = oldParamToNewParam[data.parameterIndex];
        if (data.parameterIndex == kInvalidIndex) {
          continue;
        }
        break;
      }

      case MinMaxJoint: {
        break;
      }

      case Linear: {
        auto& data = limitNew.data.linear;
        data.referenceIndex = oldParamToNewParam[data.referenceIndex];
        data.targetIndex = oldParamToNewParam[data.targetIndex];
        if (data.referenceIndex == kInvalidIndex || data.targetIndex == kInvalidIndex) {
          continue;
        }
        break;
      }

      case LinearJoint: {
        break;
      }

      case HalfPlane: {
        auto& data = limitNew.data.halfPlane;
        data.param1 = oldParamToNewParam[data.param1];
        data.param2 = oldParamToNewParam[data.param2];
        if (data.param1 == kInvalidIndex || data.param2 == kInvalidIndex) {
          continue;
        }
        break;
      }

      case Ellipsoid:
      case MinMaxJointPassive: {
        // nothing to do here.
        break;
      }
    }

    paramLimitsNew.push_back(limitNew);
  }

  return std::make_tuple(paramTransformNew, paramLimitsNew);
}

template <typename T>
template <typename T2>
ParameterTransformT<T2> ParameterTransformT<T>::cast() const {
  ParameterTransformT<T2> result;
  result.name = name;
  result.transform = transform.template cast<T2>();
  result.offsets = offsets.template cast<T2>();
  result.activeJointParams = activeJointParams;
  result.parameterSets = parameterSets;
  result.poseConstraints = poseConstraints;
  return result;
}

template <typename T>
ParameterSet ParameterTransformT<T>::getBlendShapeParameters() const {
  ParameterSet result;
  for (Eigen::Index i = 0; i < blendShapeParameters.size(); ++i) {
    if (blendShapeParameters(i) >= 0) {
      result.set(blendShapeParameters(i));
    }
  }

  return result;
}

template <typename T>
ParameterSet ParameterTransformT<T>::getFaceExpressionParameters() const {
  ParameterSet result;
  for (Eigen::Index i = 0; i < faceExpressionParameters.size(); ++i) {
    if (faceExpressionParameters(i) >= 0) {
      result.set(faceExpressionParameters(i));
    }
  }

  return result;
}

std::tuple<ParameterTransform, ParameterLimits> addBlendShapeParameters(
    ParameterTransform paramTransform,
    ParameterLimits paramLimits,
    Eigen::Index nBlendShapes) {
  // First, strip out any existing blend shape parameters, to make sure this
  // operation is idempotent.
  std::tie(paramTransform, paramLimits) = subsetParameterTransform(
      paramTransform, paramLimits, ~paramTransform.getBlendShapeParameters());

  // Now add in the additional parameters:
  paramTransform.blendShapeParameters.setZero(nBlendShapes);
  for (Eigen::Index iBlend = 0; iBlend < nBlendShapes; ++iBlend) {
    std::ostringstream oss;
    oss << "blend_" << iBlend;
    paramTransform.blendShapeParameters(iBlend) = (int)paramTransform.name.size();
    paramTransform.name.push_back(oss.str());
  }

  paramTransform.transform.conservativeResize(
      paramTransform.transform.rows(), paramTransform.name.size());
  paramTransform.transform.makeCompressed();

  return {paramTransform, paramLimits};
}

std::tuple<ParameterTransform, ParameterLimits> addFaceExpressionParameters(
    ParameterTransform paramTransform,
    ParameterLimits paramLimits,
    Eigen::Index nFaceExpressionBlendShapes) {
  // First, strip out any existing face expression parameters, to make sure this
  // operation is idempotent.
  std::tie(paramTransform, paramLimits) = subsetParameterTransform(
      paramTransform, paramLimits, ~paramTransform.getFaceExpressionParameters());

  // Now add in the additional parameters:
  paramTransform.faceExpressionParameters.setZero(nFaceExpressionBlendShapes);
  for (Eigen::Index iBlend = 0; iBlend < nFaceExpressionBlendShapes; ++iBlend) {
    std::ostringstream oss;
    oss << "face_expre_" << iBlend;
    paramTransform.faceExpressionParameters(iBlend) = (int)paramTransform.name.size();
    paramTransform.name.push_back(oss.str());
  }

  paramTransform.transform.conservativeResize(
      paramTransform.transform.rows(), paramTransform.name.size());
  paramTransform.transform.makeCompressed();

  return {paramTransform, paramLimits};
}

template struct ParameterTransformT<float>;
template struct ParameterTransformT<double>;

template ParameterTransformT<float> ParameterTransformT<float>::cast() const;
template ParameterTransformT<double> ParameterTransformT<float>::cast() const;
template ParameterTransformT<float> ParameterTransformT<double>::cast() const;
template ParameterTransformT<double> ParameterTransformT<double>::cast() const;

template std::tuple<ParameterTransformT<float>, ParameterLimits> subsetParameterTransform(
    const ParameterTransformT<float>& paramTransform,
    const ParameterLimits& parameterLimits,
    const ParameterSet& paramSet);
template std::tuple<ParameterTransformT<double>, ParameterLimits> subsetParameterTransform(
    const ParameterTransformT<double>& paramTransform,
    const ParameterLimits& parameterLimits,
    const ParameterSet& paramSet);

template ParameterTransformT<float> mapParameterTransformJoints(
    const ParameterTransformT<float>& paramTransform,
    size_t numTargetJoints,
    const std::vector<size_t>& jointMapping);

template ParameterTransformT<double> mapParameterTransformJoints(
    const ParameterTransformT<double>& paramTransform,
    size_t numTargetJoints,
    const std::vector<size_t>& jointMapping);

} // namespace momentum
