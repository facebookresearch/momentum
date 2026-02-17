/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/simd_collision_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/simd/simd.h"

namespace momentum {

namespace {

template <typename T>
[[nodiscard]] drjit::Array<T, 2> toDrJitVec(const Eigen::Vector2<T>& v) {
  return {v.x(), v.y()};
}

template <typename T>
[[nodiscard]] drjit::Array<T, 3> toDrJitVec(const Eigen::Vector3<T>& v) {
  return {v.x(), v.y(), v.z()};
}

template <typename T>
[[nodiscard]] Eigen::Vector3<T> extractSingleElement(const Vector3P<T>& vec, int index) {
  return {vec.x()[index], vec.y()[index], vec.z()[index]};
}

// It is much friendlier to SIMD than the version currently in the momentum code base
// because it includes fewer branches.
//
// Long term we should probably merge this with the other implementation, if
// properly templated we can probably even use the same code for both.
template <typename VecType1, typename VecType2>
[[nodiscard]] std::pair<Packet<typename VecType1::Scalar>, Packet<typename VecType1::Scalar>>
closestPointOnTwoSegments(
    const VecType1& p0,
    const VecType1& d0,
    const VecType2& p1,
    const VecType2& d1) {
  static_assert(std::is_same_v<typename VecType1::Scalar, typename VecType2::Scalar>);

  using T = typename VecType1::Scalar;

  const Vector3P<T> p = p1 - p0;
  const Packet<T> pd0 = drjit::dot(p, d0);
  const Packet<T> pd1 = drjit::dot(p, d1);

  const Packet<T> d00 = drjit::squared_norm(d0);
  const Packet<T> d11 = drjit::squared_norm(d1);
  const Packet<T> d01 = drjit::dot(d0, d1);
  const Packet<T> div = d00 * d11 - d01 * d01;

  auto t0 = drjit::zeros<Packet<T>>();
  auto t1 = drjit::zeros<Packet<T>>();

  // If the segments are nearly parallel, the initial assignment
  // to t0 may be garbage. The following assignments are all stable,
  // however, so after a ping-pong-ping finding the closest point
  // on the other line to the current point, we are guaranteed to
  // have stable output points.
  //
  // Note that the actual returned values are inherently unstable
  // if the lines are near parallel, but the distance from
  // (p0 + t0*d0) to (p1 + t1*d1) will be stable.
  t0 = drjit::clip(drjit::select(div > 0.0f, (pd0 * d11 - pd1 * d01) / div, t0), 0.0f, 1.0f);
  t1 = drjit::clip(drjit::select(d11 > 0.0f, (t0 * d01 - pd1) / d11, t1), 0.0f, 1.0f);
  t0 = drjit::clip(drjit::select(d00 > 0.0f, (t1 * d01 + pd0) / d00, t0), 0.0f, 1.0f);

  return {t0, t1};
}

} // namespace

template <typename T>
SimdCollisionErrorFunctionT<T>::SimdCollisionErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const CollisionGeometry& collisionGeometry)
    : SkeletonErrorFunctionT<T>(skel, pt), collisionGeometry_(collisionGeometry) {
  updateCollisionPairs();
}

template <typename T>
SimdCollisionErrorFunctionT<T>::SimdCollisionErrorFunctionT(const Character& character)
    : SimdCollisionErrorFunctionT(
          character.skeleton,
          character.parameterTransform,
          character.collision
              ? *character.collision
              : throw std::invalid_argument(
                    "Attempting to create collision error function with a character that has no collision geometries")) {
  // Do nothing
}

template <typename T>
void SimdCollisionErrorFunctionT<T>::updateCollisionPairs() {
  // clear collisions
  excludingPairIds_.clear();
  jacobianSize_ = 0;

  // Get the size of collisionGeometry_
  const auto n = collisionGeometry_.size();

  // create zero state
  const SkeletonStateT<T> state(
      this->parameterTransform_.zero().template cast<T>(), this->skeleton_);

  // Update the collision state based on the zero state and collision geometry
  collisionState_.update(state, collisionGeometry_);

  // Build BVH
  std::vector<axel::BoundingBox<T>> aabbs(n);
  for (size_t i = 0; i < n; ++i) {
    auto& aabb = aabbs[i];
    aabb.id = i;
    updateAabb(
        aabb, collisionState_.origin[i], collisionState_.direction[i], collisionState_.radius[i]);
  }
  bvh_.setBoundingBoxes(aabbs);

  // Exit early as there are no geometry objects to process
  if (n == 0) {
    return;
  }

  // Iterate over all possible pairs of geometry objects
  const auto numPotentialPairs = n * (n - 1) / 2;
  excludingPairIds_.reserve(numPotentialPairs);
  T distance = NAN;
  Vector2<T> cp;
  T overlap = NAN;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      const std::pair<size_t, size_t> pairId = std::make_pair(i, j);

      // Check if the pair intersects in the initial pose
      if (!overlaps(
              collisionState_.origin[i],
              collisionState_.direction[i],
              collisionState_.radius[i],
              collisionState_.delta[i],
              collisionState_.origin[j],
              collisionState_.direction[j],
              collisionState_.radius[j],
              collisionState_.delta[j],
              distance,
              cp,
              overlap)) {
        // no, it doesn't. check

        // walk down the this->skeleton_ hierarchy for the collision pair until we find a common
        // ancestor
        size_t p0 = collisionGeometry_[i].parent;
        size_t p1 = collisionGeometry_[j].parent;
        size_t count = 0;

        // Find the common ancestor. We can always just walk up the higher joint index because the
        // parent always has to have a lower index than the child joint.
        while (p0 != p1) {
          if (p0 > p1) {
            p0 = this->skeleton_.joints[p0].parent;
          } else {
            p1 = this->skeleton_.joints[p1].parent;
          }
          count++;
        }

        // Store collision pairs excluding those with the same parent or adjacent joints
        // If there's an overlap, add the pair ID to the set of excluding pair IDs
        if (count <= 1) {
          excludingPairIds_.insert(pairId);
        }

        jacobianSize_++;
        continue;
      }

      excludingPairIds_.insert(pairId);
    }
  }
  // TODO: Consider using BVH for faster intersection checks
}

template <typename T>
void SimdCollisionErrorFunctionT<T>::computeBroadPhase(const SkeletonStateT<T>& state) {
  // Update collision state
  {
    MT_PROFILE_EVENT("Collision: updateState");
    collisionState_.update(state, collisionGeometry_);
  }

  // Update the AABBs for each geometry object
  for (auto& aabb : bvh_.getPrimitives()) {
    const auto& index = aabb.id;
    updateAabb(
        aabb,
        collisionState_.origin[index],
        collisionState_.direction[index],
        collisionState_.radius[index]);
  }

  // Update BVH
  bvh_.setBoundingBoxes(bvh_.getPrimitives());

  // Update overlapping collision pairs
  collisionPairs_.clear();
  collisionPairs_.resize(collisionGeometry_.size());
  bvh_.traverseOverlappingPairs([&](auto indexA, auto indexB) -> bool {
    // Ensure indexA is always less than indexB
    if (indexA > indexB) {
      std::swap(indexA, indexB);
    }

    // Skip excluded pairs
    const std::pair<size_t, size_t> pairId = std::make_pair(indexA, indexB);
    if (excludingPairIds_.find(pairId) != excludingPairIds_.end()) {
      return true; // continue traversing
    }

    collisionPairs_[indexA].push_back(indexB);

    return true; // continue traversal
  });
}

template <typename T>
double SimdCollisionErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /*unused*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */) {
  if (state.jointState.empty()) {
    return 0.0;
  }

  // check all is valid
  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  computeBroadPhase(state);

  DoubleP error = drjit::zeros<DoubleP>();

  // For each joint, consider all the other joints:
  for (size_t iCol = 0; iCol < collisionPairs_.size(); ++iCol) {
    const auto& candidatesCur = collisionPairs_[iCol];

    const Vector3P<T> p_i = toDrJitVec(collisionState_.origin[iCol]);
    const Vector3P<T> d_i = toDrJitVec(collisionState_.direction[iCol]);
    const Vector2P<T> radius_i = toDrJitVec(collisionState_.radius[iCol]);

    for (auto [candidateIndices, mask] : drjit::range<IntP>(candidatesCur.size())) {
      const IntP jCol = drjit::gather<IntP>(candidatesCur.data(), candidateIndices, mask);
      const auto p_j = drjit::gather<Vector3P<T>>(collisionState_.origin.data(), jCol, mask);
      const auto d_j = drjit::gather<Vector3P<T>>(collisionState_.direction.data(), jCol, mask);
      const auto radius_j = drjit::gather<Vector2P<T>>(collisionState_.radius.data(), jCol, mask);

      const auto [t_i, t_j] = closestPointOnTwoSegments(p_i, d_i, p_j, d_j);

      const Vector3P<T> closestPoint_i = p_i + t_i * d_i;
      const Vector3P<T> closestPoint_j = p_j + t_j * d_j;
      const Packet<T> distance = drjit::norm(closestPoint_i - closestPoint_j);

      // TODO check these:
      const Packet<T> radius = (radius_i.x() * (1.0f - t_i) + radius_i.y() * t_i) +
          (radius_j.x() * (1.0f - t_j) + radius_j.y() * t_j);

      drjit::masked(error, mask) += drjit::square(drjit::maximum(radius - distance, 0));
    }
  }

  return drjit::sum(error) * kCollisionWeight * this->weight_;
}

template <typename T>
double SimdCollisionErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /*unused*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<VectorX<T>> gradient) {
  if (state.jointState.empty()) {
    return 0.0;
  }

  computeBroadPhase(state);

  DoubleP error = drjit::zeros<DoubleP>();

  // For each joint, consider all the other joints:
  for (size_t iCol = 0; iCol < collisionPairs_.size(); ++iCol) {
    const auto& candidatesCur = collisionPairs_[iCol];

    Vector3P<T> p_i = toDrJitVec(collisionState_.origin[iCol]);
    Vector3P<T> d_i = toDrJitVec(collisionState_.direction[iCol]);
    Vector2P<T> radius_i = toDrJitVec(collisionState_.radius[iCol]);

    for (auto [candidateIndices, mask] : drjit::range<IntP>(candidatesCur.size())) {
      const IntP jCol = drjit::gather<IntP>(candidatesCur.data(), candidateIndices, mask);
      const auto p_j = drjit::gather<Vector3P<T>>(collisionState_.origin.data(), jCol, mask);
      const auto d_j = drjit::gather<Vector3P<T>>(collisionState_.direction.data(), jCol, mask);
      const auto radius_j = drjit::gather<Vector2P<T>>(collisionState_.radius.data(), jCol, mask);

      const auto [t_i, t_j] = closestPointOnTwoSegments(p_i, d_i, p_j, d_j);

      const Vector3P<T> position_i = p_i + t_i * d_i;
      const Vector3P<T> position_j = p_j + t_j * d_j;
      const Vector3P<T> direction = position_i - position_j;
      const Packet<T> distance = drjit::norm(direction);

      // TODO check these:
      const Packet<T> radius = (radius_i.x() * (1.0f - t_i) + radius_i.y() * t_i) +
          (radius_j.x() * (1.0f - t_j) + radius_j.y() * t_j);
      const Packet<T> overlap = radius - distance;

      const auto finalMask = drjit::PacketMask<int, kSimdPacketSize>(mask) && distance < radius;

      if (!drjit::any(finalMask)) {
        continue;
      }

      // Use clamped distance to avoid division by zero in SIMD lanes
      constexpr T kMinSafeDistance = T(1e-8);
      const Packet<T> safeDistance = drjit::maximum(distance, Packet<T>(kMinSafeDistance));

      // calculate weight
      const Packet<T> wgt = -2.0f * kCollisionWeight * this->weight_ * overlap / safeDistance;

      drjit::masked(error, finalMask) +=
          kCollisionWeight * this->weight_ * drjit::square(radius - distance);

      const size_t iJoint = collisionGeometry_[iCol].parent;
      for (uint32_t k = 0; k < kSimdPacketSize; ++k) {
        if (!finalMask[k]) {
          continue;
        }

        const size_t jJoint = collisionGeometry_[jCol[k]].parent;
        const auto commonAncestor = this->skeleton_.commonAncestor(iJoint, jJoint);

        const Eigen::Vector3<T> direction_k = extractSingleElement(direction, k);
        const Eigen::Vector3<T> position_ik = extractSingleElement(position_i, k);
        const Eigen::Vector3<T> position_jk = extractSingleElement(position_j, k);
        const T wgt_k = wgt[k];

        // -----------------------------------
        //  process first joint
        // -----------------------------------
        size_t jointIndex = iJoint;
        while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
          const auto& jointState = state.jointState[jointIndex];
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const Vector3<T> posd = position_ik - jointState.translation();

          // calculate derivatives based on active joints
          for (size_t d = 0; d < 3; d++) {
            if (this->activeJointParams_[paramIndex + d]) {
              // calculate joint gradient
              const T val = direction_k.dot(jointState.getTranslationDerivative(d)) * wgt_k;
              // explicitly multiply with the parameter transform to generate parameter space
              // gradients
              for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                   index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                   ++index) {
                gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                    val * this->parameterTransform_.transform.valuePtr()[index];
              }
            }
            if (this->activeJointParams_[paramIndex + 3 + d]) {
              // calculate joint gradient
              const T val = direction_k.dot(jointState.getRotationDerivative(d, posd)) * wgt_k;
              // explicitly multiply with the parameter transform to generate parameter space
              // gradients
              for (auto index =
                       this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
                   index <
                   this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
                   ++index) {
                gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                    val * this->parameterTransform_.transform.valuePtr()[index];
              }
            }
          }
          if (this->activeJointParams_[paramIndex + 6]) {
            // calculate joint gradient
            const T val = direction_k.dot(jointState.getScaleDerivative(posd)) * wgt_k;
            // explicitly multiply with the parameter transform to generate parameter space
            // gradients
            for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
                 index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
                 ++index) {
              gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                  val * this->parameterTransform_.transform.valuePtr()[index];
            }
          }

          // go to the next joint
          jointIndex = this->skeleton_.joints[jointIndex].parent;
        }

        // -----------------------------------
        //  process second joint
        // -----------------------------------
        jointIndex = jJoint;
        while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
          const auto& jointState = state.jointState[jointIndex];
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const Vector3<T> posd = position_jk - jointState.translation();

          // calculate derivatives based on active joints
          for (size_t d = 0; d < 3; d++) {
            if (this->activeJointParams_[paramIndex + d]) {
              // calculate joint gradient
              const T val = direction_k.dot(jointState.getTranslationDerivative(d)) * -wgt_k;
              // explicitly multiply with the parameter transform to generate parameter space
              // gradients
              for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                   index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                   ++index) {
                gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                    val * this->parameterTransform_.transform.valuePtr()[index];
              }
            }
            if (this->activeJointParams_[paramIndex + 3 + d]) {
              // calculate joint gradient
              const T val = direction_k.dot(jointState.getRotationDerivative(d, posd)) * -wgt_k;
              // explicitly multiply with the parameter transform to generate parameter space
              // gradients
              const auto maxIndex =
                  this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
              for (auto index =
                       this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
                   index < maxIndex;
                   ++index) {
                gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                    val * this->parameterTransform_.transform.valuePtr()[index];
              }
            }
          }
          if (this->activeJointParams_[paramIndex + 6]) {
            // calculate joint gradient
            const T val = direction_k.dot(jointState.getScaleDerivative(posd)) * -wgt_k;
            // explicitly multiply with the parameter transform to generate parameter space
            // gradients
            const auto maxIndex =
                this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
            for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
                 index < maxIndex;
                 ++index) {
              gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                  val * this->parameterTransform_.transform.valuePtr()[index];
            }
          }

          // go to the next joint
          jointIndex = this->skeleton_.joints[jointIndex].parent;
        }
      }
    }
  }

  // check all is valid
  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  // return error
  return drjit::sum(error);
}

template <typename T>
double SimdCollisionErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /*unused*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<MatrixX<T>> jacobian,
    Ref<VectorX<T>> residual,
    int& usedRows) {
  computeBroadPhase(state);

  const T wgt = std::sqrt(kCollisionWeight * this->weight_);

  DoubleP error = drjit::zeros<DoubleP>();

  // For each joint, consider all the other joints:
  int row = 0;
  for (size_t iCol = 0; iCol < collisionPairs_.size(); ++iCol) {
    const auto& candidatesCur = collisionPairs_[iCol];

    Vector3P<T> p_i = toDrJitVec(collisionState_.origin[iCol]);
    Vector3P<T> d_i = toDrJitVec(collisionState_.direction[iCol]);
    Vector2P<T> radius_i = toDrJitVec(collisionState_.radius[iCol]);

    for (auto [candidateIndices, mask] : drjit::range<IntP>(candidatesCur.size())) {
      const IntP jCol = drjit::gather<IntP>(candidatesCur.data(), candidateIndices, mask);
      const auto p_j = drjit::gather<Vector3P<T>>(collisionState_.origin.data(), jCol, mask);
      const auto d_j = drjit::gather<Vector3P<T>>(collisionState_.direction.data(), jCol, mask);
      const auto radius_j = drjit::gather<Vector2P<T>>(collisionState_.radius.data(), jCol, mask);

      const auto [t_i, t_j] = closestPointOnTwoSegments(p_i, d_i, p_j, d_j);

      const Vector3P<T> position_i = p_i + t_i * d_i;
      const Vector3P<T> position_j = p_j + t_j * d_j;
      const Vector3P<T> direction = position_i - position_j;
      const Packet<T> distance = drjit::norm(direction);

      // TODO check these:
      const Packet<T> radius = (radius_i.x() * (1.0f - t_i) + radius_i.y() * t_i) +
          (radius_j.x() * (1.0f - t_j) + radius_j.y() * t_j);
      const Packet<T> overlap = radius - distance;

      const auto finalMask = drjit::PacketMask<int, kSimdPacketSize>(mask) && distance < radius;

      if (!drjit::any(finalMask)) {
        continue;
      }

      drjit::masked(error, finalMask) +=
          kCollisionWeight * this->weight_ * drjit::square(radius - distance);

      // Use clamped distance to avoid division by zero in SIMD lanes
      constexpr T kMinSafeDistance = T(1e-8);
      const Packet<T> safeDistance = drjit::maximum(distance, Packet<T>(kMinSafeDistance));

      // calculate collision resolve direction. this is what we need to push joint parent i in.
      // the direction for joint parent j is the inverse
      const Packet<T> inverseDistance = T(1.0) / safeDistance;

      // calculate constant factor
      const Packet<T> fac = 2.0f * inverseDistance * wgt;

      const size_t iJoint = collisionGeometry_[iCol].parent;
      for (uint32_t k = 0; k < kSimdPacketSize; ++k) {
        if (!finalMask[k]) {
          continue;
        }

        const size_t jJoint = collisionGeometry_[jCol[k]].parent;
        const auto commonAncestor = this->skeleton_.commonAncestor(iJoint, jJoint);

        const Eigen::Vector3<T> direction_k = extractSingleElement(direction, k);
        const T fac_k = fac[k];

        // -----------------------------------
        //  process first joint
        // -----------------------------------
        size_t jointIndex = iJoint;
        while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
          const auto& jointState = state.jointState[jointIndex];
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const Eigen::Vector3<T> posd =
              extractSingleElement(position_i, k) - jointState.translation();

          // calculate derivatives based on active joints
          for (size_t d = 0; d < 3; d++) {
            if (this->activeJointParams_[paramIndex + d]) {
              // calculate joint gradient
              const T val = direction_k.dot(jointState.getTranslationDerivative(d)) * -fac_k;
              // explicitly multiply with the parameter transform to generate parameter space
              // gradients
              for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                   index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                   ++index) {
                jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
                    val * this->parameterTransform_.transform.valuePtr()[index];
              }
            }
            if (this->activeJointParams_[paramIndex + 3 + d]) {
              // calculate joint gradient
              const T val = direction_k.dot(jointState.getRotationDerivative(d, posd)) * -fac_k;
              // explicitly multiply with the parameter transform to generate parameter space
              // gradients
              for (auto index =
                       this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
                   index <
                   this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
                   ++index) {
                jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
                    val * this->parameterTransform_.transform.valuePtr()[index];
              }
            }
          }
          if (this->activeJointParams_[paramIndex + 6]) {
            // calculate joint gradient
            const T val = direction_k.dot(jointState.getScaleDerivative(posd)) * -fac_k;
            // explicitly multiply with the parameter transform to generate parameter space
            // gradients
            for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
                 index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
                 ++index) {
              jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
                  val * this->parameterTransform_.transform.valuePtr()[index];
            }
          }

          // go to the next joint
          jointIndex = this->skeleton_.joints[jointIndex].parent;
        }

        // -----------------------------------
        //  process second joint
        // -----------------------------------
        jointIndex = jJoint;
        while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
          const auto& jointState = state.jointState[jointIndex];
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const Vector3<T> posd = extractSingleElement(position_j, k) - jointState.translation();

          // calculate derivatives based on active joints
          for (size_t d = 0; d < 3; d++) {
            if (this->activeJointParams_[paramIndex + d]) {
              // calculate joint gradient
              const T val = direction_k.dot(jointState.getTranslationDerivative(d)) * fac_k;
              // explicitly multiply with the parameter transform to generate parameter space
              // gradients
              for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                   index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                   ++index) {
                jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
                    val * this->parameterTransform_.transform.valuePtr()[index];
              }
            }
            if (this->activeJointParams_[paramIndex + 3 + d]) {
              // calculate joint gradient
              const T val = direction_k.dot(jointState.getRotationDerivative(d, posd)) * fac_k;
              // explicitly multiply with the parameter transform to generate parameter space
              // gradients
              for (auto index =
                       this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
                   index <
                   this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
                   ++index) {
                jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
                    val * this->parameterTransform_.transform.valuePtr()[index];
              }
            }
          }
          if (this->activeJointParams_[paramIndex + 6]) {
            // calculate joint gradient
            const T val = direction_k.dot(jointState.getScaleDerivative(posd)) * fac_k;
            // explicitly multiply with the parameter transform to generate parameter space
            // gradients
            for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
                 index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
                 ++index) {
              jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
                  val * this->parameterTransform_.transform.valuePtr()[index];
            }
          }

          // go to the next joint
          jointIndex = this->skeleton_.joints[jointIndex].parent;
        }

        residual(row) = overlap[k] * wgt;
        row++;
      } // for each element of the packet k
    } // for each simd packet j
  } // for each collision geometry i

  usedRows = row;

  // return error
  return drjit::sum(error);
}

template <typename T>
size_t SimdCollisionErrorFunctionT<T>::getJacobianSize() const {
  return jacobianSize_;
}

template class SimdCollisionErrorFunctionT<float>;
template class SimdCollisionErrorFunctionT<double>;

} // namespace momentum
