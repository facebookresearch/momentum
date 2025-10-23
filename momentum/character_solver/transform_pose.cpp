/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/transform_pose.h"
#include "momentum/character/character.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character_solver/orientation_error_function.h"
#include "momentum/character_solver/position_error_function.h"
#include "momentum/character_solver/skeleton_solver_function.h"
#include "momentum/common/checks.h"
#include "momentum/common/log.h"
#include "momentum/math/constants.h"
#include "momentum/math/random.h"
#include "momentum/solver/gauss_newton_solver.h"

#include <numeric>

namespace momentum {

namespace {

momentum::Character createRigidBodySkeleton(const momentum::Character& character) {
  const auto rigidBodyParameters = character.parameterTransform.getRigidParameters();

  std::vector<bool> rigidJoints = character.parametersToActiveJoints(rigidBodyParameters);
  // Note that we need to keep _all_ the parameters that affect the rigid joints.

  // The rigidly transformed joints should appear near the top of the skeleton.
  // Rather than trying to be fancy about this, we'll just find the last affected joint and
  // keep all the joints in the list before it.  This will also allow us to much more easily
  // simplify the parameter transform since we don't need to deal with missing joints in the
  // middle.
  size_t numRigidJoints = 0;
  for (size_t i = 0; i < rigidJoints.size(); ++i) {
    if (rigidJoints[i]) {
      numRigidJoints = i + 1;
    }
  }

  MT_THROW_IF(
      numRigidJoints == 0, "No rigid joints found in character, unable to apply transformation.");

  // Need to include all joints and parameters in the parent chain:
  std::fill(rigidJoints.begin(), rigidJoints.begin() + numRigidJoints, true);

  // Retain all parameters that affect the rigid pose, which means all parameters
  // affecting joints above the root.
  const auto parametersToKeep = character.activeJointsToParameters(rigidJoints);

  // This only changes the parameters:
  auto [intermediateTransform, intermediateLimits] =
      subsetParameterTransform(character.parameterTransform, ParameterLimits{}, parametersToKeep);

  // Construct the new skeleton by just clipping the skeleton to the rigid joints.
  momentum::Skeleton resultSkeleton;
  resultSkeleton.joints.reserve(numRigidJoints);
  for (size_t i = 0; i < numRigidJoints; ++i) {
    resultSkeleton.joints.push_back(character.skeleton.joints.at(i));
  }

  std::vector<size_t> jointMapping(character.skeleton.joints.size(), kInvalidIndex);
  std::iota(jointMapping.begin(), jointMapping.begin() + resultSkeleton.joints.size(), 0);

  return {
      resultSkeleton,
      mapParameterTransformJoints(
          intermediateTransform, resultSkeleton.joints.size(), jointMapping)};
}

} // namespace

template <typename T>
std::vector<ModelParametersT<T>> transformPose(
    const Character& character,
    const std::vector<ModelParametersT<T>>& modelParameters,
    const TransformT<T>& transforms) {
  return transformPose(character, modelParameters, std::vector<TransformT<T>>{transforms}, true);
}

template <typename T>
PoseTransformSolverT<T>::PoseTransformSolverT(const Character& character) {
  numModelParametersFull_ = character.parameterTransform.numAllModelParameters();

  characterSimplified_ = createRigidBodySkeleton(character);
  MT_CHECK(!characterSimplified_.skeleton.joints.empty(), "No joints in simplified character");
  // Use the farthest joint since it should have all parameters applied to it:
  rootJointSimplified_ = characterSimplified_.skeleton.joints.size() - 1;

  parameterTransformSimplified_ = characterSimplified_.parameterTransform.cast<T>();
  rigidParametersSimplified_ = characterSimplified_.parameterTransform.getRigidParameters();

  simplifiedParamToFullParamIdx_.resize(parameterTransformSimplified_.numAllModelParameters());
  for (Eigen::Index iSimplifiedParam = 0;
       iSimplifiedParam < characterSimplified_.parameterTransform.numAllModelParameters();
       ++iSimplifiedParam) {
    const auto fullParamIdx = character.parameterTransform.getParameterIdByName(
        characterSimplified_.parameterTransform.name[iSimplifiedParam]);
    MT_CHECK(fullParamIdx != momentum::kInvalidIndex);
    simplifiedParamToFullParamIdx_.at(iSimplifiedParam) = fullParamIdx;
  }

  positionError_ = std::make_shared<momentum::PositionErrorFunctionT<T>>(characterSimplified_);
  orientationError_ =
      std::make_shared<momentum::OrientationErrorFunctionT<T>>(characterSimplified_);

  solverFunction_ = std::make_unique<momentum::SkeletonSolverFunctionT<T>>(
      characterSimplified_, parameterTransformSimplified_);
  solverFunction_->addErrorFunction(positionError_);
  solverFunction_->addErrorFunction(orientationError_);

  momentum::GaussNewtonSolverOptions solverOptions;
  solverOptions.maxIterations = 1000;
  solverOptions.minIterations = 10;
  // It shouldn't need regularization:
  solverOptions.regularization = std::numeric_limits<T>::epsilon();
  solverOptions.useBlockJtJ = true;

  solver_ = std::make_unique<GaussNewtonSolverT<T>>(solverOptions, solverFunction_.get());
  solver_->setEnabledParameters(rigidParametersSimplified_);

  solvedParametersSimplified_ =
      ModelParametersT<T>::Zero(parameterTransformSimplified_.numAllModelParameters());
}

template <typename T>
PoseTransformSolverT<T>::~PoseTransformSolverT() = default;

template <typename T>
void PoseTransformSolverT<T>::transformPose(
    ModelParametersT<T>& modelParametersFull,
    const TransformT<T>& transform,
    const ModelParametersT<T>& modelParametersPrev) {
  MT_CHECK(
      modelParametersFull.size() == numModelParametersFull_,
      "Model parameters don't match full character");
  MT_CHECK(
      modelParametersPrev.size() == numModelParametersFull_ || modelParametersPrev.size() == 0,
      "Prev model parameters don't match full character");

  // Initialize with params copied from fullParams:
  solvedParametersSimplified_.v.setZero();
  for (size_t iSimplifiedParam = 0; iSimplifiedParam < simplifiedParamToFullParamIdx_.size();
       ++iSimplifiedParam) {
    const auto fullParamIdx = simplifiedParamToFullParamIdx_[iSimplifiedParam];
    solvedParametersSimplified_[iSimplifiedParam] = modelParametersFull[fullParamIdx];
  }

  skelStateSimplified_.set(
      parameterTransformSimplified_.apply(solvedParametersSimplified_),
      characterSimplified_.skeleton,
      false);

  if (modelParametersPrev.size() != 0) {
    // Copy rigid parameters from previous solve to ensure continuity:
    for (size_t iSimplifiedParam = 0; iSimplifiedParam < simplifiedParamToFullParamIdx_.size();
         ++iSimplifiedParam) {
      const auto fullParamIdx = simplifiedParamToFullParamIdx_[iSimplifiedParam];
      if (rigidParametersSimplified_.test(iSimplifiedParam)) {
        solvedParametersSimplified_[iSimplifiedParam] = modelParametersPrev[fullParamIdx];
      }
    }
  }

  const TransformT<T> worldFromRootTarget =
      transform * skelStateSimplified_.jointState[rootJointSimplified_].transform;

  positionError_->clearConstraints();
  positionError_->addConstraint(PositionDataT<T>(
      Eigen::Vector3<T>::Zero(), worldFromRootTarget.translation, rootJointSimplified_, 100.0));

  orientationError_->clearConstraints();
  orientationError_->addConstraint(OrientationDataT<T>(
      Eigen::Quaternion<T>::Identity(), worldFromRootTarget.rotation, rootJointSimplified_, 10.0));

  // The momentum solver has a lot of local minima issues related to the use
  // of Euler angles.  As a result for any given solve there is a chance if will
  // get stuck somewhere non-optimal, which would be really bad in this case because
  // we'd end up with a single pose out of the set which is flipped backward or whatever.
  // To avoid this, we'll do the solve with restarts: each solve will start with a random
  // set of rigid parameters and we'll run to convergence, then exit when the resulting
  // rotation is close enough.

  for (size_t iIter = 0;; ++iIter) {
    if (iIter > 10000) {
      MT_LOGW(
          "Solver failed to converge after {} iterations when solving for rigid transform.", iIter);
      break;
    }

    solver_->solve(solvedParametersSimplified_.v);
    skelStateSimplified_.set(
        parameterTransformSimplified_.apply(solvedParametersSimplified_),
        characterSimplified_.skeleton,
        false);
    if ((skelStateSimplified_.jointState[rootJointSimplified_].rotation().toRotationMatrix() -
         worldFromRootTarget.rotation.toRotationMatrix())
            .norm() < 1e-3) {
      break;
    }

    // Randomize the starting location to deal with Euler angle local minimum issues:
    for (int k = 0; k < characterSimplified_.parameterTransform.numAllModelParameters(); ++k) {
      if (rigidParametersSimplified_.test(k)) {
        solvedParametersSimplified_[k] = rng_.uniform(-pi(), pi());
      }
    }
  }

  for (size_t simplifiedParamIdx = 0; simplifiedParamIdx < simplifiedParamToFullParamIdx_.size();
       ++simplifiedParamIdx) {
    if (rigidParametersSimplified_.test(simplifiedParamIdx)) {
      const auto fullParamIdx = simplifiedParamToFullParamIdx_[simplifiedParamIdx];
      modelParametersFull.v[fullParamIdx] = solvedParametersSimplified_[simplifiedParamIdx];
    }
  }
}

template <typename T>
std::vector<ModelParametersT<T>> transformPose(
    const Character& character,
    const std::vector<ModelParametersT<T>>& modelParameters,
    const std::vector<TransformT<T>>& transforms,
    [[maybe_unused]] bool ensureContinuousOutput) {
  MT_CHECK(
      transforms.size() == 1 || modelParameters.size() == transforms.size(),
      "Mismatched transforms and parameters");

  // Applies a transform to a character by solving for the new set of model
  // parameters such that the new character's root matches up with the transformed
  // version of the old character's root.  This could in principle be done in a closed
  // form but such a solution would need to be hand-coded for each parametrization
  // of character, so for now we'll do it using an IK solve (which is very fast
  // in this case because we're only working with the rigid parameters).
  const auto nPoses = modelParameters.size();

  PoseTransformSolverT<T> transformSolver(character);

  std::vector<momentum::ModelParametersT<T>> result(modelParameters.size());

  momentum::ModelParametersT<T> prevSolve;
  for (size_t iFrame = 0; iFrame < nPoses; ++iFrame) {
    const auto& fullParams_init = modelParameters.at(iFrame);
    if (fullParams_init.size() == 0) {
      continue;
    }

    const TransformT<T>& transform =
        (iFrame < transforms.size()) ? transforms.at(iFrame) : transforms.at(0);

    result[iFrame] = modelParameters[iFrame];
    transformSolver.transformPose(result[iFrame], transform, prevSolve);
    prevSolve = result[iFrame];
  }

  return result;
}

template std::vector<ModelParametersT<float>> transformPose(
    const Character& characterFull,
    const std::vector<ModelParametersT<float>>& modelParametersFull,
    const std::vector<TransformT<float>>& transforms,
    bool ensureContinuousOutput);

template std::vector<ModelParametersT<double>> transformPose(
    const Character& characterFull,
    const std::vector<ModelParametersT<double>>& modelParametersFull,
    const std::vector<TransformT<double>>& transforms,
    bool ensureContinuousOutput);

template std::vector<ModelParametersT<float>> transformPose(
    const Character& character,
    const std::vector<ModelParametersT<float>>& modelParameters,
    const TransformT<float>& transform);

template std::vector<ModelParametersT<double>> transformPose(
    const Character& character,
    const std::vector<ModelParametersT<double>>& modelParameters,
    const TransformT<double>& transform);

template class PoseTransformSolverT<float>;
template class PoseTransformSolverT<double>;

} // namespace momentum
