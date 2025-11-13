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

#include <span>

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

  bool operator==(const PoseConstraint& poseConstraint) const;
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

  /// Parameters that control the rest-space positions of the skinned locators.
  /// This array will be either empty or have the same size as the number of
  /// skinned locators.  Each entry maps from a skinned locator index to the
  /// first of the three (x,y,z) parameters that control its position.  Skinned
  /// locators that aren't controlled by the paramete transform will have
  /// skinnedLocatorParameters[i] = -1.
  VectorXi skinnedLocatorParameters;

  /// Return a ParameterTransform object with no model parameters. The model can still perform FK
  /// with JointParameters, but it does not have any degrees of freedom for IK.
  [[nodiscard]] static ParameterTransformT<T> empty(size_t nJointParameters);

  /// Return a ParameterTransform object where the model parameters are identical to the joint
  /// parameters.
  [[nodiscard]] static ParameterTransformT<T> identity(std::span<const std::string> jointNames);

  /// Compute activeJointParams based on the transform and the input ParameterSet.
  [[nodiscard]] VectorX<bool> computeActiveJointParams(const ParameterSet& ps = allParams()) const;

  /// Return the index of a model parameter from its name.
  [[nodiscard]] size_t getParameterIdByName(const std::string& nm) const;

  /// Map model parameters to joint parameters using a linear transformation.
  [[nodiscard]] JointParametersT<T> apply(const ModelParametersT<T>& parameters) const;

  /// Map model parameters to joint parameters using a linear transformation.
  [[nodiscard]] JointParametersT<T> apply(const CharacterParametersT<T>& parameters) const;

  /// Return rest pose joint parameters.
  [[nodiscard]] JointParametersT<T> zero() const;

  /// Return bind pose joint parameters (same as the rest pose here).
  [[nodiscard]] JointParametersT<T> bindPose() const;

  /// Get a list of scaling parameters (with prefix "scale_").
  [[nodiscard]] ParameterSet getScalingParameters() const;

  /// Get a list of root parameters (with prefix "root_")
  [[nodiscard]] ParameterSet getRigidParameters() const;

  /// Return all parameters used for posing the body (excludes scaling parameters, blend shape
  /// parameters, or any parameters used for physics).
  [[nodiscard]] ParameterSet getPoseParameters() const;

  /// Get a list of blend shape parameters.
  [[nodiscard]] ParameterSet getBlendShapeParameters() const;

  /// Get a list of face expression parameters.
  [[nodiscard]] ParameterSet getFaceExpressionParameters() const;

  /// Get a list of skinned locator parameters
  [[nodiscard]] ParameterSet getSkinnedLocatorParameters() const;

  /// Get a parameter set, if allowMissing is set then it will return an empty parameter set if no
  /// such parameterset is found in the file.
  [[nodiscard]] ParameterSet getParameterSet(
      const std::string& parameterSetName,
      bool allowMissing = false) const;

  template <typename T2>
  [[nodiscard]] ParameterTransformT<T2> cast() const;

  /// Create a simplified transform given the enabled parameters.
  [[nodiscard]] ParameterTransformT<T> simplify(const ParameterSet& enabledParameters) const;

  /// Dimension of all model parameters, including pose, scaling, marker joints, and blendshape
  /// parameters for id and expressions.
  [[nodiscard]] Eigen::Index numAllModelParameters() const {
    return transform.cols();
  }

  /// Dimension of the output jointParameters vector.
  [[nodiscard]] Eigen::Index numJointParameters() const {
    return transform.rows();
  }

  /// Dimension of identity blendshape parameters.
  [[nodiscard]] Eigen::Index numBlendShapeParameters() const;

  /// Dimension of facial expression parameters.
  [[nodiscard]] Eigen::Index numFaceExpressionParameters() const;

  /// Dimension of facial expression parameters.
  [[nodiscard]] Eigen::Index numSkinnedLocatorParameters() const;

  /// Dimension of skeletal model parameters, including pose parameters,
  /// scaling parameters, locator joint parameters etc. basically everything
  /// but blendshapes (ids and expressions).
  [[nodiscard]] Eigen::Index numSkeletonParameters() const {
    return numAllModelParameters() - numBlendShapeParameters() - numFaceExpressionParameters() -
        numSkinnedLocatorParameters();
  }

  [[nodiscard]] bool isApprox(const ParameterTransformT<T>& parameterTransform) const;
};

using ParameterTransform = ParameterTransformT<float>;
using ParameterTransformd = ParameterTransformT<double>;

/// Return a parameter mapping that only includes the listed parameters.
template <typename T>
[[nodiscard]] std::tuple<ParameterTransformT<T>, ParameterLimits> subsetParameterTransform(
    const ParameterTransformT<T>& paramTransform,
    const ParameterLimits& paramLimitsOld,
    const ParameterSet& paramSet);

/// Construct a new parameter transform where the joints have been mapped to a
/// new skeleton.  Joints that are mapped to kInvalidIndex will be simply skipped.
/// Note that this does _not_ delete any parameters so it's possible if you remove
/// enough joints to have "orphan" parameters still kicking around; to avoid this
/// consider also applying an appropriate subsetParameterTransform() operation.
template <typename T>
[[nodiscard]] ParameterTransformT<T> mapParameterTransformJoints(
    const ParameterTransformT<T>& parameterTransform,
    size_t numTargetJoints,
    const std::vector<size_t>& jointMapping);

[[nodiscard]] std::tuple<ParameterTransform, ParameterLimits> addBlendShapeParameters(
    ParameterTransform paramTransform,
    ParameterLimits paramLimits,
    Eigen::Index nBlendShapes);

[[nodiscard]] std::tuple<ParameterTransform, ParameterLimits> addFaceExpressionParameters(
    ParameterTransform paramTransform,
    ParameterLimits paramLimits,
    Eigen::Index nFaceExpressionBlendShapes);

/// Add a set of parameters that control the rest-space positions of the skinned locators.
/// This function will add 3 parameters for each locator, one for each of the x, y, and z
/// components of the locator position.  The parameters are added to the end of the
/// parameter transform.  The parameter transform is returned along with the updated
/// list of parameter limits.  Note that you pass in the locator names rather than the
/// actual SkinnedLocator objects because this avoids a circular dependency.
[[nodiscard]] std::tuple<ParameterTransform, ParameterLimits> addSkinnedLocatorParameters(
    ParameterTransform paramTransform,
    ParameterLimits paramLimits,
    const std::vector<bool>& activeLocators,
    const std::vector<std::string>& locatorNames = {});

} // namespace momentum
