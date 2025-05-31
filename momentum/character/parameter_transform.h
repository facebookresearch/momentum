/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/parameter_limits.h>
#include <momentum/character/types.h>
#include <momentum/math/utility.h>

#include <gsl/span>

#include <bitset>
#include <string>
#include <unordered_map>

namespace momentum {

struct PoseConstraint {
  /// A vector of tuple of type: (model parameter index, parameter value).
  /// The model parameter index must be in [0, numModelParameters.size()).
  /// The value of the model parameters specified here are kept constant (= parameter value) during
  /// optimization. The ordering of elements in the vector doesn't matter (making it an unordered
  /// map would be more semantically correct) since it stores an index to the model parameters
  std::vector<std::pair<size_t, float>> parameterIdValue;

  inline bool operator==(const PoseConstraint& poseConstraint) const {
    // Compare the parameterIdValue as sets
    std::map<size_t, float> paramIdToValue1;
    std::copy(
        parameterIdValue.begin(),
        parameterIdValue.end(),
        std::inserter(paramIdToValue1, paramIdToValue1.begin()));

    std::map<size_t, float> paramIdToValue2;
    std::copy(
        poseConstraint.parameterIdValue.begin(),
        poseConstraint.parameterIdValue.end(),
        std::inserter(paramIdToValue2, paramIdToValue2.begin()));

    if (paramIdToValue1.size() != paramIdToValue2.size()) {
      return false;
    }

    auto pred = [](const auto& l, const auto& r) {
      return ((l.first == r.first) && isApprox(l.second, r.second));
    };
    return std::equal(
        paramIdToValue1.begin(), paramIdToValue1.end(), paramIdToValue2.begin(), pred);
  }
};

using ParameterSets = std::unordered_map<std::string, ParameterSet>;
using PoseConstraints = std::unordered_map<std::string, PoseConstraint>;

/// A parameter transform is an abstraction of the joint parameters that maps a model_parameter
/// vector to a joint_parameter vector. It allows mapping a single model_parameter to multiple
/// joints, and a single joint being influenced by multiple model_parameters joint parameters are
/// calculated from parameters in the following way : <joint_parameters> = <transform> *
/// <model_parameters> + <offsets>
template <typename T>
struct ParameterTransformT {
  /// The list of model parameter names.
  std::vector<std::string> name;

  /// The sparse mapping matrix that maps model parameters to joint parameters.
  SparseRowMatrix<T> transform;

  /// @deprecated Constant offset factor for each joint.
  Eigen::VectorX<T> offsets;

  /// The list of joint *parameters* that are actually active and influenced from the transform.
  VectorX<bool> activeJointParams;

  /// Convenience grouping of model parameters.
  ParameterSets parameterSets;

  /// A set of predefined poses.
  PoseConstraints poseConstraints;

  /// The indices of the parameters that influence blend shapes; blendShapeParameters(0) is the
  /// parameter that controls the 0th blend shape, etc.
  VectorXi blendShapeParameters;

  /// The indices of the parameters that influence face expressions; faceExpressionParameters(0) is
  /// the parameter that controls the 0th face expression parameter, etc.
  VectorXi faceExpressionParameters;

  /// Return a ParameterTransform object with no model parameters. The model can still perform FK
  /// with JointParameters, but it does not have any degrees of freedom for IK.
  static ParameterTransformT<T> empty(size_t nJointParameters);

  /// Return a ParameterTransform object where the model parameters are identical to the joint
  /// parameters.
  static ParameterTransformT<T> identity(gsl::span<const std::string> jointNames);

  /// Compute activeJointParams based on the transform and the input ParameterSet.
  VectorX<bool> computeActiveJointParams(const ParameterSet& ps = allParams()) const;

  /// Return the index of a model parameter from its name.
  size_t getParameterIdByName(const std::string& nm) const;

  /// Map model parameters to joint parameters using a linear transformation.
  JointParametersT<T> apply(const ModelParametersT<T>& parameters) const;

  /// Map model parameters to joint parameters using a linear transformation.
  JointParametersT<T> apply(const CharacterParametersT<T>& parameters) const;

  /// Return rest pose joint parameters.
  JointParametersT<T> zero() const;

  /// Return bind pose joint parameters (same as the rest pose here).
  JointParametersT<T> bindPose() const;

  /// Get a list of scaling parameters (with prefix "scale_").
  ParameterSet getScalingParameters() const;

  /// Get a list of root parameters (with prefix "root_")
  ParameterSet getRigidParameters() const;

  /// Return all parameters used for posing the body (excludes scaling parameters, blend shape
  /// parameters, or any parameters used for physics).
  ParameterSet getPoseParameters() const;

  /// Get a list of blend shape parameters.
  ParameterSet getBlendShapeParameters() const;

  /// Get a list of face expression parameters.
  ParameterSet getFaceExpressionParameters() const;

  /// Get a parameter set, if allowMissing is set then it will return an empty parameter set if no
  /// such parameterset is found in the file.
  ParameterSet getParameterSet(const std::string& parameterSetName, bool allowMissing = false)
      const;

  template <typename T2>
  ParameterTransformT<T2> cast() const;

  /// Create a simplified transform given the enabled parameters.
  ParameterTransformT<T> simplify(const ParameterSet& enabledParameters) const;

  /// Dimension of all model parameters, including pose, scaling, marker joints, and blendshape
  /// parameters for id and expressions.
  Eigen::Index numAllModelParameters() const {
    return transform.cols();
  }

  /// Dimension of the output jointParameters vector.
  Eigen::Index numJointParameters() const {
    return transform.rows();
  }

  /// Dimension of identity blendshape parameters.
  Eigen::Index numBlendShapeParameters() const {
    return blendShapeParameters.size();
  }

  /// Dimension of facial expression parameters.
  Eigen::Index numFaceExpressionParameters() const {
    return faceExpressionParameters.size();
  }

  /// Dimension of skeletal model parameters, including pose parameters,
  /// scaling parameters, locator joint parameters etc. basically everything
  /// but blendshapes (ids and expressions).
  Eigen::Index numSkeletonParameters() const {
    return numAllModelParameters() - numBlendShapeParameters() - numFaceExpressionParameters();
  }

  inline bool isApprox(const ParameterTransformT<T>& parameterTransform) const {
    // special handling of zero sparse matrix
    bool isTransformEqual = false;
    if (transform.cols() > 0 && transform.rows() > 0 && parameterTransform.transform.cols() > 0 &&
        parameterTransform.transform.rows() > 0) {
      isTransformEqual = transform.isApprox(parameterTransform.transform);
    } else {
      isTransformEqual = (transform.cols() == parameterTransform.transform.cols()) &&
          (transform.rows() == parameterTransform.transform.rows());
    }
    if (!isTransformEqual)
      return false;

    return (
        (name == parameterTransform.name) &&
        activeJointParams.isApprox(parameterTransform.activeJointParams) &&
        (parameterSets == parameterTransform.parameterSets) &&
        (poseConstraints == parameterTransform.poseConstraints) &&
        (blendShapeParameters == parameterTransform.blendShapeParameters));
  }
};

using ParameterTransform = ParameterTransformT<float>;
using ParameterTransformd = ParameterTransformT<double>;

/// Return a parameter mapping that only includes the listed parameters.
template <typename T>
std::tuple<ParameterTransformT<T>, ParameterLimits> subsetParameterTransform(
    const ParameterTransformT<T>& paramTransform,
    const ParameterLimits& parameterLimits,
    const ParameterSet& paramSet);

/// Construct a new parameter transform where the joints have been mapped to a
/// new skeleton.  Joints that are mapped to kInvalidIndex will be simply skipped.
/// Note that this does _not_ delete any parameters so it's possible if you remove
/// enough joints to have "orphan" parameters still kicking around; to avoid this
/// consider also applying an appropriate subsetParameterTransform() operation.
template <typename T>
ParameterTransformT<T> mapParameterTransformJoints(
    const ParameterTransformT<T>& parameterTransform,
    size_t numTargetJoints,
    const std::vector<size_t>& jointMapping);

std::tuple<ParameterTransform, ParameterLimits> addBlendShapeParameters(
    ParameterTransform paramTransform,
    ParameterLimits parameterLimits,
    Eigen::Index nBlendShapes);

std::tuple<ParameterTransform, ParameterLimits> addFaceExpressionParameters(
    ParameterTransform paramTransform,
    ParameterLimits parameterLimits,
    Eigen::Index nFaceExpressionBlendShapes);

} // namespace momentum
