/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/skeleton_state.h>

#include <momentum/character/skeleton.h>
#include <momentum/common/checks.h>
#include <momentum/common/profile.h>
#include <momentum/math/types.h>

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
  // get input joints
  const JointListT<T>& joints = ::momentum::cast<T>(referenceSkeleton.joints);

  // initialize array size variables
  const size_t numJoints = joints.size();

  // ensure that all variables are valid
  MT_CHECK(
      jointParameters.size() == gsl::narrow<Eigen::Index>(numJoints * kParametersPerJoint),
      "Unexpected joint parameter size. Expected '{}' (# of joints '{}' X kParametersPerJoint '{}') but got '{}'.",
      numJoints * kParametersPerJoint,
      numJoints,
      kParametersPerJoint,
      jointParameters.size());

  // go over all joint elements and calculate Transformation
  for (size_t jointID = 0; jointID < numJoints; jointID++) {
    const Eigen::Index parameterOffset = jointID * kParametersPerJoint;

    // some reference for quick access
    const JointT<T>& joint = joints[jointID];

    // set joint-state based on parameters
    // IMPORTANT: this all assumes that parent joints always appear before their children in the
    // joint list, so their joint state will already be calculated when processing the children
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

  // ensure arrays are valid
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
  // check that we're actually good to compare the states
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

  // calculate position error and orientation error in global coordinates per joint
  for (size_t i = 0; i < state1.jointState.size(); i++) {
    result.positionError[i] =
        (state1.jointState[i].translation() - state2.jointState[i].translation()).norm();
    const double dot = std::min(
        std::max(
            state1.jointState[i].rotation().normalized().dot(
                state2.jointState[i].rotation().normalized()),
            T(-1.0)),
        T(1.0));
    const double sgn = dot < 0.0 ? -1.0 : 1.0;
    result.orientationError[i] = gsl::narrow_cast<float>(2.0 * std::acos(sgn * dot));
  }

  // derive RMSE/max values
  result.positionRMSE = std::sqrt(
      result.positionError.squaredNorm() / static_cast<float>(result.positionError.size()));
  result.orientationRMSE = std::sqrt(
      result.orientationError.squaredNorm() / static_cast<float>(result.positionError.size()));
  result.positionMax = result.positionError.maxCoeff();
  result.orientationMax = result.orientationError.maxCoeff();

  return result;
}

// It turns out because the body is topologically sorted, there is a pretty simple algorithm for
// computing the common ancestor.
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
    // Note that we treat kInvalidIndex as if it is the _lowest_ index, as the world is the
    // topological ancestor of everything.
    if (ancestorB != kInvalidIndex && (ancestorA == kInvalidIndex || ancestorA < ancestorB)) {
      // parentB can't possible be a parent of parentA, so move parentB up one level in the
      // hierarchy.
      B_to_ancestorB = skelState.jointState[ancestorB].localTransform * B_to_ancestorB;
      ancestorB = referenceSkeleton.joints[ancestorB].parent;
    } else if (
        ancestorA != kInvalidIndex && (ancestorB == kInvalidIndex || ancestorB < ancestorA)) {
      A_to_ancestorA = skelState.jointState[ancestorA].localTransform * A_to_ancestorA;
      ancestorA = referenceSkeleton.joints[ancestorA].parent;
    } else {
      // Reached a common ancestor of A and B so we can stop.
      break;
    }
  }

  // At this point, ancestorA == ancestorB.
  MT_CHECK(ancestorA == ancestorB, "{} is not {}", ancestorA, ancestorB);
  return B_to_ancestorB.inverse() * A_to_ancestorA;
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

} // namespace momentum
