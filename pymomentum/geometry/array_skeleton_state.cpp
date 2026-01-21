/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/array_skeleton_state.h"

#include <pymomentum/array_utility/array_utility.h>
#include <pymomentum/array_utility/batch_accessor.h>
#include <pymomentum/array_utility/geometry_accessors.h>
#include <pymomentum/geometry/array_parameter_transform.h>

#include <momentum/character/joint.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/math/utility.h>

#include <dispenso/parallel_for.h>

#include <utility>

namespace pymomentum {

namespace {

template <typename T>
py::array_t<T> jointParametersToSkeletonStateImpl(
    const momentum::Character& character,
    const py::array& jointParams,
    const LeadingDimensions& leadingDims,
    JointParamsShape shape) {
  const auto nJoints = static_cast<py::ssize_t>(character.skeleton.joints.size());
  const auto nBatch = leadingDims.totalBatchElements();

  // Create output array with shape [..., nJoints, 8]
  auto result = createOutputArray<T>(leadingDims, {nJoints, 8});

  // Create batch indexer for converting flat indices to multi-dimensional indices
  BatchIndexer indexer(leadingDims);

  // Create accessors using geometry accessors
  JointParametersAccessor<T> inputAcc(jointParams, leadingDims, nJoints, shape);
  SkeletonStateAccessor<T> outputAcc(result, leadingDims, nJoints);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get joint parameters as JointParametersT<T>
      auto jp = inputAcc.get(indices);

      // Compute skeleton state (global transforms)
      const momentum::SkeletonStateT<T> skelState(jp, character.skeleton, /*computeDeriv=*/false);

      // Extract transforms from skeleton state using JointState::transform
      momentum::TransformListT<T> transforms(nJoints);
      for (int64_t iJoint = 0; iJoint < nJoints; ++iJoint) {
        transforms[iJoint] = skelState.jointState[iJoint].transform;
      }

      // Write transforms to output
      outputAcc.setTransforms(indices, transforms);
    });
  }

  return result;
}

template <typename T>
py::array_t<T> jointParametersToLocalSkeletonStateImpl(
    const momentum::Character& character,
    const py::array& jointParams,
    const LeadingDimensions& leadingDims,
    JointParamsShape shape) {
  const auto nJoints = static_cast<py::ssize_t>(character.skeleton.joints.size());
  const auto nBatch = leadingDims.totalBatchElements();

  // Create output array with shape [..., nJoints, 8]
  auto result = createOutputArray<T>(leadingDims, {nJoints, 8});

  // Create batch indexer for converting flat indices to multi-dimensional indices
  BatchIndexer indexer(leadingDims);

  // Create accessors using geometry accessors
  JointParametersAccessor<T> inputAcc(jointParams, leadingDims, nJoints, shape);
  SkeletonStateAccessor<T> outputAcc(result, leadingDims, nJoints);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get joint parameters as JointParametersT<T>
      auto jp = inputAcc.get(indices);

      // Compute skeleton state (includes local transforms)
      const momentum::SkeletonStateT<T> skelState(jp, character.skeleton, /*computeDeriv=*/false);

      // Extract local transforms from skeleton state using JointState::localTransform
      momentum::TransformListT<T> transforms(nJoints);
      for (int64_t iJoint = 0; iJoint < nJoints; ++iJoint) {
        transforms[iJoint] = skelState.jointState[iJoint].localTransform;
      }

      // Write transforms to output
      outputAcc.setTransforms(indices, transforms);
    });
  }

  return result;
}

template <typename T>
py::array_t<T> skeletonStateToJointParametersImpl(
    const momentum::Character& character,
    const py::array& skeletonState,
    const LeadingDimensions& leadingDims) {
  const auto nJoints = static_cast<py::ssize_t>(character.skeleton.joints.size());
  const auto nBatch = leadingDims.totalBatchElements();

  // Create output array with shape [..., nJoints, 7] (structured format)
  auto result = createOutputArray<T>(leadingDims, {nJoints, 7});

  // Create batch indexer for converting flat indices to multi-dimensional indices
  BatchIndexer indexer(leadingDims);

  // Create accessors using geometry accessors
  SkeletonStateAccessor<T> inputAcc(skeletonState, leadingDims, nJoints);
  JointParametersAccessor<T> outputAcc(result, leadingDims, nJoints, JointParamsShape::Structured);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get transforms from skeleton state
      auto transforms = inputAcc.getTransforms(indices);

      // Use momentum's skeletonStateToJointParameters function
      auto jp = momentum::skeletonStateToJointParameters<T>(transforms, character.skeleton);

      // Write joint parameters to output
      outputAcc.set(indices, jp);
    });
  }

  return result;
}

template <typename T>
py::array_t<T> localSkeletonStateToJointParametersImpl(
    const momentum::Character& character,
    const py::array& localSkeletonState,
    const LeadingDimensions& leadingDims) {
  const auto nJoints = static_cast<py::ssize_t>(character.skeleton.joints.size());
  const auto nBatch = leadingDims.totalBatchElements();

  // Create output array with shape [..., nJoints, 7] (structured format)
  auto result = createOutputArray<T>(leadingDims, {nJoints, 7});

  // Create batch indexer for converting flat indices to multi-dimensional indices
  BatchIndexer indexer(leadingDims);

  // Create accessors using geometry accessors
  SkeletonStateAccessor<T> inputAcc(localSkeletonState, leadingDims, nJoints);
  JointParametersAccessor<T> outputAcc(result, leadingDims, nJoints, JointParamsShape::Structured);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get local transforms from skeleton state
      auto localTransforms = inputAcc.getTransforms(indices);

      // Reconstruct joint parameters from local transforms
      // Each joint has 7 parameters: tx, ty, tz, rx, ry, rz, scale
      Eigen::Matrix<T, Eigen::Dynamic, 1> jpVec(nJoints * 7);

      for (int64_t iJoint = 0; iJoint < nJoints; ++iJoint) {
        const auto& joint = character.skeleton.joints[iJoint];
        const auto& localTrans = localTransforms[iJoint];

        // Translation offset: localTrans.translation = joint.translationOffset + params[0:3]
        Eigen::Vector3<T> transPart =
            localTrans.translation - joint.translationOffset.template cast<T>();
        jpVec(iJoint * 7 + 0) = transPart.x();
        jpVec(iJoint * 7 + 1) = transPart.y();
        jpVec(iJoint * 7 + 2) = transPart.z();

        // Rotation: localRot = preRotation * Rz(rz) * Ry(ry) * Rx(rx)
        // We need to extract rx, ry, rz (Euler angles)
        Eigen::Quaternion<T> preRotInv = joint.preRotation.template cast<T>().conjugate();
        Eigen::Quaternion<T> pureRot = preRotInv * localTrans.rotation;

        // Convert quaternion to Euler angles using momentum's utility (Extrinsic XYZ convention)
        Eigen::Matrix3<T> rotMat = pureRot.toRotationMatrix();
        Eigen::Vector3<T> euler =
            momentum::rotationMatrixToEulerXYZ(rotMat, momentum::EulerConvention::Extrinsic);
        jpVec(iJoint * 7 + 3) = euler(0); // rx
        jpVec(iJoint * 7 + 4) = euler(1); // ry
        jpVec(iJoint * 7 + 5) = euler(2); // rz

        // Scale: scale = 2^params[6], so params[6] = log2(scale)
        jpVec(iJoint * 7 + 6) = std::log2(localTrans.scale);
      }

      // Write joint parameters to output
      outputAcc.set(indices, momentum::JointParametersT<T>(jpVec));
    });
  }

  return result;
}

} // namespace

py::array jointParametersToSkeletonStateArray(
    const momentum::Character& character,
    const py::buffer& jointParams) {
  ArrayChecker checker("joint_parameters_to_skeleton_state");
  JointParamsShape shape =
      checker.validateJointParameters(jointParams, "joint_parameters", character);

  if (checker.isFloat64()) {
    return jointParametersToSkeletonStateImpl<double>(
        character, jointParams, checker.getLeadingDimensions(), shape);
  } else {
    return jointParametersToSkeletonStateImpl<float>(
        character, jointParams, checker.getLeadingDimensions(), shape);
  }
}

py::array modelParametersToSkeletonStateArray(
    const momentum::Character& character,
    const py::buffer& modelParams) {
  // First apply parameter transform to get joint parameters
  py::array jointParams = applyParameterTransformArray(character.parameterTransform, modelParams);

  // Then convert joint parameters to skeleton state
  return jointParametersToSkeletonStateArray(character, jointParams);
}

py::array jointParametersToLocalSkeletonStateArray(
    const momentum::Character& character,
    const py::buffer& jointParams) {
  ArrayChecker checker("joint_parameters_to_local_skeleton_state");
  JointParamsShape shape =
      checker.validateJointParameters(jointParams, "joint_parameters", character);

  if (checker.isFloat64()) {
    return jointParametersToLocalSkeletonStateImpl<double>(
        character, jointParams, checker.getLeadingDimensions(), shape);
  } else {
    return jointParametersToLocalSkeletonStateImpl<float>(
        character, jointParams, checker.getLeadingDimensions(), shape);
  }
}

py::array modelParametersToLocalSkeletonStateArray(
    const momentum::Character& character,
    const py::buffer& modelParams) {
  // First apply parameter transform to get joint parameters
  py::array jointParams = applyParameterTransformArray(character.parameterTransform, modelParams);

  // Then convert joint parameters to local skeleton state
  return jointParametersToLocalSkeletonStateArray(character, jointParams);
}

py::array skeletonStateToJointParametersArray(
    const momentum::Character& character,
    const py::buffer& skeletonState) {
  ArrayChecker checker("skeleton_state_to_joint_parameters");
  checker.validateSkeletonState(skeletonState, "skeleton_state", character);

  if (checker.isFloat64()) {
    return skeletonStateToJointParametersImpl<double>(
        character, skeletonState, checker.getLeadingDimensions());
  } else {
    return skeletonStateToJointParametersImpl<float>(
        character, skeletonState, checker.getLeadingDimensions());
  }
}

py::array localSkeletonStateToJointParametersArray(
    const momentum::Character& character,
    const py::buffer& localSkeletonState) {
  ArrayChecker checker("local_skeleton_state_to_joint_parameters");
  checker.validateSkeletonState(localSkeletonState, "local_skeleton_state", character);

  if (checker.isFloat64()) {
    return localSkeletonStateToJointParametersImpl<double>(
        character, localSkeletonState, checker.getLeadingDimensions());
  } else {
    return localSkeletonStateToJointParametersImpl<float>(
        character, localSkeletonState, checker.getLeadingDimensions());
  }
}

} // namespace pymomentum
