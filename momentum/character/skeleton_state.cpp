/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/skeleton_state.h"

#include "momentum/character/joint.h"
#include "momentum/character/skeleton.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/utility.h"

#include <algorithm>
#include <numeric>

namespace momentum {

template <typename T>
SkeletonStateT<T>::SkeletonStateT(
    const JointParametersT<T>& parameters,
    const Skeleton& referenceSkeleton,
    bool computeDeriv)
    : jointParameters(parameters), jointState(referenceSkeleton.joints.size()) {
  set(referenceSkeleton, computeDeriv);
}

template <typename T>
SkeletonStateT<T>::SkeletonStateT(
    JointParametersT<T>&& parameters,
    const Skeleton& referenceSkeleton,
    bool computeDeriv)
    : jointParameters(std::move(parameters)), jointState(referenceSkeleton.joints.size()) {
  set(referenceSkeleton, computeDeriv);
}

template <typename T>
template <typename T2>
SkeletonStateT<T>::SkeletonStateT(const SkeletonStateT<T2>& rhs)
    : jointParameters(rhs.jointParameters.template cast<T>()), jointState(rhs.jointState.size()) {
  copy(rhs);
}

template <typename T>
template <typename T2>
void SkeletonStateT<T>::set(const SkeletonStateT<T2>& rhs) {
  MT_PROFILE_FUNCTION();
  jointParameters = rhs.jointParameters.template cast<T>();
  jointState.resize(rhs.jointState.size());
  copy(rhs);
}

template <typename T>
template <typename T2>
void SkeletonStateT<T>::copy(const SkeletonStateT<T2>& rhs) {
  MT_PROFILE_FUNCTION();
  for (size_t iJoint = 0; iJoint < jointState.size(); ++iJoint) {
    jointState[iJoint].set(rhs.jointState[iJoint]);
  }
}

template <typename T>
void SkeletonStateT<T>::set(
    const JointParametersT<T>& parameters,
    const Skeleton& referenceSkeleton,
    bool computeDeriv) {
  MT_PROFILE_FUNCTION();
  this->jointParameters = parameters;
  this->jointState.resize(referenceSkeleton.joints.size());
  set(referenceSkeleton, computeDeriv);
}

template <typename T>
void SkeletonStateT<T>::set(
    JointParametersT<T>&& parameters,
    const Skeleton& referenceSkeleton,
    bool computeDeriv) {
  MT_PROFILE_FUNCTION();
  this->jointParameters = std::move(parameters);
  this->jointState.resize(referenceSkeleton.joints.size());
  set(referenceSkeleton, computeDeriv);
}

template <typename T>
void SkeletonStateT<T>::set(const Skeleton& referenceSkeleton, bool computeDeriv) {
  MT_PROFILE_FUNCTION();
  const JointListT<T>& joints = ::momentum::cast<T>(referenceSkeleton.joints);
  const size_t numJoints = joints.size();

  MT_CHECK(
      jointParameters.size() == static_cast<Eigen::Index>(numJoints * kParametersPerJoint),
      "Unexpected joint parameter size. Expected '{}' (# of joints '{}' X kParametersPerJoint '{}') but got '{}'.",
      numJoints * kParametersPerJoint,
      numJoints,
      kParametersPerJoint,
      jointParameters.size());

  // IMPORTANT: this single forward pass assumes parents always precede their children in the
  // joint list (i.e. the skeleton is topologically sorted). Each child reads its parent's
  // already-computed JointState, so violating that ordering will silently propagate stale
  // parent transforms into children.
  for (size_t jointID = 0; jointID < numJoints; jointID++) {
    const Eigen::Index parameterOffset = jointID * kParametersPerJoint;
    const JointT<T>& joint = joints[jointID];

    if (joint.parent == kInvalidIndex) {
      jointState[jointID].set(
          joint, jointParameters.v.template middleRows<7>(parameterOffset), nullptr, computeDeriv);
    } else {
      jointState[jointID].set(
          joint,
          jointParameters.v.template middleRows<7>(parameterOffset),
          &jointState[joint.parent],
          computeDeriv);
    }
  }

  MT_CHECK(jointState.size() == numJoints, "{} is not {}", jointState.size(), numJoints);
}

template <typename T>
TransformListT<T> SkeletonStateT<T>::toTransforms() const {
  TransformListT<T> result;
  result.reserve(jointState.size());
  for (const auto& js : jointState) {
    result.push_back(js.transform);
  }
  return result;
}

template <typename T>
StateSimilarity SkeletonStateT<T>::compare(
    const SkeletonStateT<T>& state1,
    const SkeletonStateT<T>& state2) {
  MT_CHECK(
      state1.jointParameters.size() == state2.jointParameters.size(),
      "{} is not {}",
      state1.jointParameters.size(),
      state2.jointParameters.size());
  MT_CHECK(
      state1.jointState.size() == state2.jointState.size(),
      "{} is not {}",
      state1.jointState.size(),
      state2.jointState.size());

  StateSimilarity result;
  result.positionError.resize(state1.jointState.size());
  result.orientationError.resize(state1.jointState.size());

  // Per-joint position/orientation error in world space. Quaternion dot is clamped to [-1, 1]
  // to guard acos against numerical drift just outside the valid domain, and the sign flip on
  // negative dot picks the shorter of the two equivalent quaternion representations (q and -q
  // encode the same rotation) so the angular error stays in [0, pi].
  for (size_t i = 0; i < state1.jointState.size(); i++) {
    result.positionError[i] =
        (state1.jointState[i].translation() - state2.jointState[i].translation()).norm();
    const T dot = std::min(
        std::max(
            state1.jointState[i].rotation().normalized().dot(
                state2.jointState[i].rotation().normalized()),
            T(-1.0)),
        T(1.0));
    const T sgn = dot < T(0) ? T(-1) : T(1);
    result.orientationError[i] = gsl::narrow_cast<float>(T(2) * std::acos(sgn * dot));
  }

  result.positionRMSE = std::sqrt(
      result.positionError.squaredNorm() / static_cast<float>(result.positionError.size()));
  // TODO: orientationRMSE divides by positionError.size() instead of orientationError.size().
  // The sizes are identical today (both resized to jointState.size() above), so this is harmless,
  // but it's a latent bug if the two arrays ever diverge. Use orientationError.size() here.
  result.orientationRMSE = std::sqrt(
      result.orientationError.squaredNorm() / static_cast<float>(result.orientationError.size()));
  result.positionMax = result.positionError.maxCoeff();
  result.orientationMax = result.orientationError.maxCoeff();

  return result;
}

// Because the skeleton is topologically sorted (parent index < child index), the lowest common
// ancestor of A and B can be found by repeatedly walking the joint with the larger index up to
// its parent until both pointers meet. We accumulate the local-transform chain on each side,
// so the final result composes A->ancestor and ancestor->B without ever inverting world-space
// transforms.
template <typename T>
TransformT<T> transformAtoB(
    size_t jointA,
    size_t jointB,
    const Skeleton& referenceSkeleton,
    const SkeletonStateT<T>& skelState) {
  size_t ancestorA = jointA;
  size_t ancestorB = jointB;

  TransformT<T> B_to_ancestorB;
  TransformT<T> A_to_ancestorA;

  while (true) {
    // kInvalidIndex represents "world" and is treated as the lowest index, since world is the
    // topological ancestor of every joint.
    if (ancestorB != kInvalidIndex && (ancestorA == kInvalidIndex || ancestorA < ancestorB)) {
      // ancestorB has the larger index, so it cannot be an ancestor of ancestorA -- walk B up.
      B_to_ancestorB = skelState.jointState[ancestorB].localTransform * B_to_ancestorB;
      if (ancestorB < referenceSkeleton.joints.size()) {
        ancestorB = referenceSkeleton.joints[ancestorB].parent;
      } else {
        break;
      }
    } else if (
        ancestorA != kInvalidIndex && (ancestorB == kInvalidIndex || ancestorB < ancestorA)) {
      A_to_ancestorA = skelState.jointState[ancestorA].localTransform * A_to_ancestorA;
      if (ancestorA < referenceSkeleton.joints.size()) {
        ancestorA = referenceSkeleton.joints[ancestorA].parent;
      } else {
        break;
      }
    } else {
      // Indices are equal -- we've reached the common ancestor.
      break;
    }
  }

  MT_CHECK(ancestorA == ancestorB, "{} is not {}", ancestorA, ancestorB);
  return B_to_ancestorB.inverse() * A_to_ancestorA;
}

namespace {

template <typename T>
[[nodiscard]] Eigen::Vector<T, kParametersPerJoint> localTransformToJointParameters(
    const Joint& joint,
    const TransformT<T>& localTransform) {
  Eigen::Vector<T, kParametersPerJoint> result = Eigen::Vector<T, kParametersPerJoint>::Zero();
  result.template segment<3>(0) =
      localTransform.translation - joint.translationOffset.template cast<T>();

  const Eigen::Quaternion<T> localRotation =
      joint.preRotation.template cast<T>().inverse() * localTransform.rotation;
  result.template segment<3>(3) = quaternionToEuler(localRotation);
  result(6) = std::log2(localTransform.scale);
  return result;
}

} // namespace

template <typename T>
JointParametersT<T> skeletonStateToJointParameters(
    const SkeletonStateT<T>& state,
    const Skeleton& skeleton) {
  JointParametersT<T> result =
      JointParametersT<T>::Zero(skeleton.joints.size() * kParametersPerJoint);
  for (size_t i = 0; i < skeleton.joints.size(); ++i) {
    // TODO: parentJoint and parentXF are computed but never used in this overload -- the local
    // transform is read directly from jointState[i].localTransform. Remove the dead lookup, or
    // assert it matches parentXF.inverse() * jointState[i].transform.
    const auto parentJoint = skeleton.joints[i].parent;
    TransformT<T> parentXF;
    if (parentJoint != kInvalidIndex) {
      parentXF = state.jointState[parentJoint].transform;
    }

    const TransformT<T> localXF = state.jointState[i].localTransform;
    result.v.template segment<kParametersPerJoint>(i * kParametersPerJoint) =
        localTransformToJointParameters(skeleton.joints[i], localXF);
  }

  return result;
}

template <typename T>
JointParametersT<T> skeletonStateToJointParameters(
    const TransformListT<T>& state,
    const Skeleton& skeleton) {
  JointParametersT<T> result =
      JointParametersT<T>::Zero(skeleton.joints.size() * kParametersPerJoint);
  for (size_t i = 0; i < skeleton.joints.size(); ++i) {
    const auto parentJoint = skeleton.joints[i].parent;
    TransformT<T> parentXF;
    if (parentJoint != kInvalidIndex) {
      parentXF = state[parentJoint];
    }

    // parentXF * localXF = jointXF, so localXF = parentXF^-1 * jointXF.
    const TransformT<T> localXF = parentXF.inverse() * state[i];
    result.v.template segment<kParametersPerJoint>(i * kParametersPerJoint) =
        localTransformToJointParameters(skeleton.joints[i], localXF);
  }
  return result;
}

namespace {

// Constrained inverse of localTransformToJointParameters: zero-fill any parameter slot whose
// activeJointParams entry is false, and pick the rotation extraction that matches how many
// rotation axes are active. This avoids the Euler-angle ambiguity that the unconstrained 3-axis
// path suffers from when a joint really only has 1 or 2 free rotation DOFs.
template <typename T>
[[nodiscard]] Eigen::Vector<T, kParametersPerJoint>
localTransformToJointParametersRespectingConstraints(
    const Joint& joint,
    const TransformT<T>& localTransform,
    const VectorX<bool>& activeJointParams,
    size_t jointIndex) {
  const size_t paramOffset = jointIndex * kParametersPerJoint;

  Eigen::Vector<T, kParametersPerJoint> result = Eigen::Vector<T, kParametersPerJoint>::Zero();

  for (int k = 0; k < 3; ++k) {
    if (activeJointParams[paramOffset + k]) {
      result(k) = localTransform.translation(k) - static_cast<T>(joint.translationOffset(k));
    }
  }

  // Strip the joint's pre-rotation so we work in the joint's own rotation frame.
  const Eigen::Quaternion<T> localRotation =
      joint.preRotation.template cast<T>().inverse() * localTransform.rotation;

  const Matrix3<T> rotationMatrix = localRotation.toRotationMatrix();

  std::array<int, 3> activeAxes{};
  size_t numActiveAxes = 0;
  for (int k = 0; k < 3; ++k) {
    if (activeJointParams[paramOffset + 3 + k]) {
      activeAxes[numActiveAxes++] = k;
    }
  }

  if (numActiveAxes == 1) {
    const auto axis = activeAxes[0];
    const T angle = rotationMatrixToOneAxisEuler(rotationMatrix, axis);
    result(3 + axis) = angle;
  } else if (numActiveAxes == 2) {
    const Vector2<T> angles =
        rotationMatrixToTwoAxisEuler(rotationMatrix, activeAxes[0], activeAxes[1]);
    result(3 + activeAxes[0]) = angles[0];
    result(3 + activeAxes[1]) = angles[1];
  } else if (numActiveAxes == 3) {
    // ZYX-then-reverse matches the (Rx * Ry * Rz) convention used elsewhere in momentum and
    // produces the same parameters as the unconstrained quaternionToEuler path.
    result.template segment<3>(3) = rotationMatrixToEulerZYX(rotationMatrix).reverse();
  } else {
    // numActiveAxes == 0: leave rotation parameters at zero.
  }

  if (activeJointParams[paramOffset + 6]) {
    result(6) = std::log2(localTransform.scale);
  }

  return result;
}

} // namespace

template <typename T>
JointParametersT<T> skeletonStateToJointParametersRespectingActiveParameters(
    const SkeletonStateT<T>& state,
    const Skeleton& skeleton,
    const VectorX<bool>& activeJointParams) {
  MT_CHECK(
      activeJointParams.size() ==
          static_cast<Eigen::Index>(skeleton.joints.size() * kParametersPerJoint),
      "activeJointParams size {} does not match expected size {} (skeleton joints: {})",
      activeJointParams.size(),
      skeleton.joints.size() * kParametersPerJoint,
      skeleton.joints.size());

  JointParametersT<T> result =
      JointParametersT<T>::Zero(skeleton.joints.size() * kParametersPerJoint);

  for (size_t i = 0; i < skeleton.joints.size(); ++i) {
    // TODO: parentJoint and parentXF are computed but never used in this overload -- the local
    // transform is read directly from jointState[i].localTransform. Remove the dead lookup, or
    // assert it matches parentXF.inverse() * jointState[i].transform.
    const auto parentJoint = skeleton.joints[i].parent;
    TransformT<T> parentXF;
    if (parentJoint != kInvalidIndex) {
      parentXF = state.jointState[parentJoint].transform;
    }

    const TransformT<T> localXF = state.jointState[i].localTransform;
    result.v.template segment<kParametersPerJoint>(i * kParametersPerJoint) =
        localTransformToJointParametersRespectingConstraints(
            skeleton.joints[i], localXF, activeJointParams, i);
  }

  return result;
}

template <typename T>
JointParametersT<T> skeletonStateToJointParametersRespectingActiveParameters(
    const TransformListT<T>& state,
    const Skeleton& skeleton,
    const VectorX<bool>& activeJointParams) {
  MT_CHECK(
      activeJointParams.size() ==
          static_cast<Eigen::Index>(skeleton.joints.size() * kParametersPerJoint),
      "activeJointParams size {} does not match expected size {} (skeleton joints: {})",
      activeJointParams.size(),
      skeleton.joints.size() * kParametersPerJoint,
      skeleton.joints.size());

  JointParametersT<T> result =
      JointParametersT<T>::Zero(skeleton.joints.size() * kParametersPerJoint);

  for (size_t i = 0; i < skeleton.joints.size(); ++i) {
    const auto parentJoint = skeleton.joints[i].parent;
    TransformT<T> parentXF;
    if (parentJoint != kInvalidIndex) {
      parentXF = state[parentJoint];
    }

    // parentXF * localXF = jointXF, so localXF = parentXF^-1 * jointXF.
    const TransformT<T> localXF = parentXF.inverse() * state[i];
    result.v.template segment<kParametersPerJoint>(i * kParametersPerJoint) =
        localTransformToJointParametersRespectingConstraints(
            skeleton.joints[i], localXF, activeJointParams, i);
  }

  return result;
}

template struct SkeletonStateT<float>;
template struct SkeletonStateT<double>;

template void SkeletonStateT<float>::set(const SkeletonStateT<float>&);
template void SkeletonStateT<float>::set(const SkeletonStateT<double>&);
template void SkeletonStateT<double>::set(const SkeletonStateT<float>&);
template void SkeletonStateT<double>::set(const SkeletonStateT<double>&);

template SkeletonStateT<float>::SkeletonStateT(const SkeletonStateT<double>&);
template SkeletonStateT<double>::SkeletonStateT(const SkeletonStateT<float>&);

template TransformT<float> transformAtoB(
    size_t jointA,
    size_t jointB,
    const Skeleton& referenceSkeleton,
    const SkeletonStateT<float>& skelState);
template TransformT<double> transformAtoB(
    size_t jointA,
    size_t jointB,
    const Skeleton& referenceSkeleton,
    const SkeletonStateT<double>& skelState);

template JointParametersT<float> skeletonStateToJointParameters(
    const SkeletonStateT<float>& state,
    const Skeleton& skeleton);
template JointParametersT<double> skeletonStateToJointParameters(
    const SkeletonStateT<double>& state,
    const Skeleton& skeleton);

template JointParametersT<float> skeletonStateToJointParameters(
    const TransformListT<float>& state,
    const Skeleton& skeleton);
template JointParametersT<double> skeletonStateToJointParameters(
    const TransformListT<double>& state,
    const Skeleton& skeleton);

template JointParametersT<float> skeletonStateToJointParametersRespectingActiveParameters(
    const SkeletonStateT<float>& state,
    const Skeleton& skeleton,
    const VectorX<bool>& activeJointParams);
template JointParametersT<double> skeletonStateToJointParametersRespectingActiveParameters(
    const SkeletonStateT<double>& state,
    const Skeleton& skeleton,
    const VectorX<bool>& activeJointParams);

template JointParametersT<float> skeletonStateToJointParametersRespectingActiveParameters(
    const TransformListT<float>& state,
    const Skeleton& skeleton,
    const VectorX<bool>& activeJointParams);
template JointParametersT<double> skeletonStateToJointParametersRespectingActiveParameters(
    const TransformListT<double>& state,
    const Skeleton& skeleton,
    const VectorX<bool>& activeJointParams);

} // namespace momentum
