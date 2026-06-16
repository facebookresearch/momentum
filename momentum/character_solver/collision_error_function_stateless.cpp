/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/collision_error_function_stateless.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"
#include "momentum/math/utility.h"

namespace momentum {

template <typename T>
CollisionErrorFunctionStatelessT<T>::CollisionErrorFunctionStatelessT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const CollisionGeometry& cg)
    : SkeletonErrorFunctionT<T>(skel, pt), collisionGeometry(cg) {
  updateCollisionPairs();
}

template <typename T>
CollisionErrorFunctionStatelessT<T>::CollisionErrorFunctionStatelessT(const Character& character)
    : CollisionErrorFunctionStatelessT(
          character.skeleton,
          character.parameterTransform,
          character.collision
              ? *character.collision
              : throw std::invalid_argument(
                    "Attempting to create collision error function with a character that has no collision geometries")) {
  // Do nothing
}

template <typename T>
void CollisionErrorFunctionStatelessT<T>::updateCollisionPairs() {
  validPairs_.clear();

  const SkeletonStateT<T> state(
      this->parameterTransform_.zero().template cast<T>(), this->skeleton_);
  collisionState.update(state, collisionGeometry);

  const auto n = collisionGeometry.size();

  aabbs_.resize(n);
  for (size_t i = 0; i < n; ++i) {
    aabbs_[i].id = gsl::narrow_cast<axel::Index>(i);
    updateAabb(aabbs_[i], collisionState, i);
  }

  if (n == 0) {
    return;
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (isValidCollisionPair(collisionState, collisionGeometry, this->skeleton_, i, j)) {
        const size_t lca = this->skeleton_.commonAncestor(
            collisionGeometry[i].parent, collisionGeometry[j].parent);
        validPairs_.push_back({i, j, lca});
      }
    }
  }
}

template <typename T>
void CollisionErrorFunctionStatelessT<T>::computeBroadPhase(const SkeletonStateT<T>& state) {
  {
    MT_PROFILE_EVENT("Collision: updateState");
    collisionState.update(state, collisionGeometry);
  }

  for (size_t i = 0; i < aabbs_.size(); ++i) {
    updateAabb(aabbs_[i], collisionState, i);
  }
}

template <typename T>
bool CollisionErrorFunctionStatelessT<T>::checkCollision(
    const CollisionGeometryStateT<T>& colState,
    size_t indexA,
    size_t indexB,
    CollisionOverlapResultT<T>& result) const {
  if (!aabbs_[indexA].intersects(aabbs_[indexB])) {
    return false;
  }
  return overlaps(colState, collisionGeometry, indexA, indexB, result);
}

template <typename T>
void CollisionErrorFunctionStatelessT<T>::accumulateGradientAlongChain(
    const SkeletonStateT<T>& state,
    const Vector3<T>& position,
    const Vector3<T>& direction,
    T weight,
    size_t startJoint,
    size_t stopJoint,
    Ref<VectorX<T>> gradient,
    T scaleCorrection) const {
  accumulateChainDerivatives<T>(
      state,
      this->skeleton_,
      this->activeJointParams_,
      position,
      direction,
      weight,
      startJoint,
      stopJoint,
      scaleCorrection,
      [&](T value, size_t jointParamIndex) {
        gradient_jointParams_to_modelParams(
            value, jointParamIndex, this->parameterTransform_, gradient);
      });
}

template <typename T>
void CollisionErrorFunctionStatelessT<T>::accumulateJacobianAlongChain(
    const SkeletonStateT<T>& state,
    const Vector3<T>& position,
    const Vector3<T>& direction,
    T factor,
    size_t startJoint,
    size_t stopJoint,
    Ref<MatrixX<T>> jacobian,
    int row,
    T scaleCorrection) const {
  accumulateChainDerivatives<T>(
      state,
      this->skeleton_,
      this->activeJointParams_,
      position,
      direction,
      factor,
      startJoint,
      stopJoint,
      scaleCorrection,
      [&](T value, size_t jointParamIndex) {
        jacobian_jointParams_to_modelParams(
            value, jointParamIndex, row, this->parameterTransform_, jacobian);
      });
}

template <typename T>
std::vector<Vector2i> CollisionErrorFunctionStatelessT<T>::getCollisionPairs() const {
  std::vector<Vector2i> collidingPairs;
  for (const auto& pair : validPairs_) {
    CollisionOverlapResultT<T> result;
    if (checkCollision(collisionState, pair.indexA, pair.indexB, result)) {
      collidingPairs.emplace_back(pair.indexA, pair.indexB);
    }
  }
  return collidingPairs;
}

template <typename T>
double CollisionErrorFunctionStatelessT<T>::getError(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */) {
  if (state.jointState.empty()) {
    return 0.0;
  }

  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  computeBroadPhase(state);

  double error = 0;
  for (const auto& pair : validPairs_) {
    CollisionOverlapResultT<T> result;
    if (!checkCollision(collisionState, pair.indexA, pair.indexB, result)) {
      continue;
    }
    error += sqr(result.overlap) * kCollisionWeight * this->weight_;
  }

  return error;
}

template <typename T>
double CollisionErrorFunctionStatelessT<T>::getGradient(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<VectorX<T>> gradient) {
  if (state.jointState.empty()) {
    return 0.0;
  }

  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  computeBroadPhase(state);

  double error = 0;
  for (const auto& pair : validPairs_) {
    CollisionOverlapResultT<T> result;
    if (!checkCollision(collisionState, pair.indexA, pair.indexB, result)) {
      continue;
    }

    error += sqr(result.overlap) * kCollisionWeight * this->weight_;

    const Vector3<T> direction = result.positionA - result.positionB;
    const T wgt = -T(2) * kCollisionWeight * this->weight_ * (result.overlap / result.distance);

    accumulateGradientAlongChain(
        state,
        result.positionA,
        direction,
        wgt,
        collisionGeometry[pair.indexA].parent,
        pair.commonAncestor,
        gradient,
        -result.distance * result.radiusA * ln2<T>());
    accumulateGradientAlongChain(
        state,
        result.positionB,
        direction,
        -wgt,
        collisionGeometry[pair.indexB].parent,
        pair.commonAncestor,
        gradient,
        result.distance * result.radiusB * ln2<T>());

    // Scale derivatives above the common ancestor
    {
      const T netScaleVal =
          (direction.squaredNorm() - result.distance * (result.radiusA + result.radiusB)) *
          ln2<T>() * wgt;
      size_t ancestorIndex = pair.commonAncestor;
      while (ancestorIndex != kInvalidIndex) {
        const size_t paramIndex = ancestorIndex * kParametersPerJoint;
        if (this->activeJointParams_[paramIndex + 6]) {
          gradient_jointParams_to_modelParams(
              netScaleVal, paramIndex + 6, this->parameterTransform_, gradient);
        }
        ancestorIndex = this->skeleton_.joints[ancestorIndex].parent;
      }
    }
  }

  return error;
}

template <typename T>
double CollisionErrorFunctionStatelessT<T>::getJacobian(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<MatrixX<T>> jacobian,
    Ref<VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();
  MT_CHECK(
      jacobian.rows() >= gsl::narrow<Eigen::Index>(getJacobianSize()),
      "Jacobian rows mismatch: Actual {}, Expected {}",
      jacobian.rows(),
      gsl::narrow<Eigen::Index>(getJacobianSize()));
  MT_CHECK(
      residual.rows() >= gsl::narrow<Eigen::Index>(getJacobianSize()),
      "Residual rows mismatch: Actual {}, Expected {}",
      residual.rows(),
      gsl::narrow<Eigen::Index>(getJacobianSize()));

  if (state.jointState.empty()) {
    return 0.0;
  }

  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  computeBroadPhase(state);

  const T wgt = std::sqrt(kCollisionWeight * this->weight_);
  int pos = 0;
  double error = 0;

  for (const auto& pair : validPairs_) {
    CollisionOverlapResultT<T> result;
    if (!checkCollision(collisionState, pair.indexA, pair.indexB, result)) {
      continue;
    }

    error += sqr(result.overlap) * kCollisionWeight * this->weight_;

    const Vector3<T> direction = result.positionA - result.positionB;
    const T fac = wgt / result.distance;
    const int row = pos++;

    accumulateJacobianAlongChain(
        state,
        result.positionA,
        direction,
        -fac,
        collisionGeometry[pair.indexA].parent,
        pair.commonAncestor,
        jacobian,
        row,
        -result.distance * result.radiusA * ln2<T>());
    accumulateJacobianAlongChain(
        state,
        result.positionB,
        direction,
        fac,
        collisionGeometry[pair.indexB].parent,
        pair.commonAncestor,
        jacobian,
        row,
        result.distance * result.radiusB * ln2<T>());

    // Scale derivatives above the common ancestor
    {
      const T netScaleVal = -fac * ln2<T>() * direction.squaredNorm() +
          wgt * (result.radiusA + result.radiusB) * ln2<T>();
      size_t ancestorIndex = pair.commonAncestor;
      while (ancestorIndex != kInvalidIndex) {
        const size_t paramIndex = ancestorIndex * kParametersPerJoint;
        if (this->activeJointParams_[paramIndex + 6]) {
          jacobian_jointParams_to_modelParams(
              netScaleVal, paramIndex + 6, row, this->parameterTransform_, jacobian);
        }
        ancestorIndex = this->skeleton_.joints[ancestorIndex].parent;
      }
    }

    residual(row) = result.overlap * wgt;
  }

  usedRows = pos;
  return error;
}

template <typename T>
size_t CollisionErrorFunctionStatelessT<T>::getJacobianSize() const {
  return validPairs_.size();
}

template class CollisionErrorFunctionStatelessT<float>;
template class CollisionErrorFunctionStatelessT<double>;

} // namespace momentum
